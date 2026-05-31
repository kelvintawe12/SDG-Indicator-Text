"""End-to-end training + evaluation pipeline.

This is the single function that every experiment runs through. The
notebook and the CLI both call :func:`run_experiment` so the recorded
numbers cannot drift between them.

Flow
----
1. Load train CSV → :class:`LoadedDataset`.
2. Apply preprocessing per config.
3. Build features (TF-IDF word / char, SBERT — any subset).
4. Multilabel-stratified hold-out split.
5. Apply imbalance strategy on the train fold only (no leakage).
6. Fit head; compute validation probabilities.
7. Tune per-label thresholds on validation if enabled.
8. Compute the metric bundle; persist artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sdgtext.data.load import load_devex
from sdgtext.data.preprocess import PreprocessConfig, normalize_corpus
from sdgtext.data.splits import single_holdout_indices
from sdgtext.eval.metrics import MetricBundle, apply_thresholds, evaluate, per_label_report
from sdgtext.eval.thresholds import tune_thresholds
from sdgtext.features.combine import stack_features
from sdgtext.features.embeddings import encode_sbert
from sdgtext.features.tfidf import make_char_tfidf, make_word_tfidf
from sdgtext.models.heads import build_head, predict_proba_safe
from sdgtext.models.imbalance import STRATEGIES
from sdgtext.utils.logging import get_logger
from sdgtext.utils.seeding import seed_everything

log = get_logger(__name__)


@dataclass
class ExperimentResult:
    config_name: str
    metrics: MetricBundle
    per_label: dict
    thresholds: list[float]
    label_names: list[str]
    notes: str

    def to_json(self) -> dict:
        return {
            "config_name": self.config_name,
            "metrics": self.metrics.as_dict(),
            "per_label": self.per_label,
            "thresholds": list(self.thresholds),
            "label_names": list(self.label_names),
            "notes": self.notes,
        }


def _build_features(texts_train, texts_val, feature_cfg: dict, sbert_cfg: dict):
    """Construct (X_train, X_val, fitted_vectorizers) per config.

    We fit each vectorizer on the training fold only and *transform*
    the validation fold to avoid leakage of validation-fold vocabulary
    into TF-IDF IDF weights. The SBERT encoder is fit-free (a frozen
    pretrained model) so we encode both folds independently.
    """
    use = list(feature_cfg.get("use", ["tfidf_word"]))
    blocks_train, blocks_val = [], []
    fitted: dict[str, Any] = {}

    if "tfidf_word" in use:
        v = make_word_tfidf(feature_cfg.get("tfidf_word", {}))
        Xw_tr = v.fit_transform(texts_train)
        Xw_va = v.transform(texts_val)
        blocks_train.append(Xw_tr)
        blocks_val.append(Xw_va)
        fitted["tfidf_word"] = v

    if "tfidf_char" in use:
        v = make_char_tfidf(feature_cfg.get("tfidf_char", {}))
        Xc_tr = v.fit_transform(texts_train)
        Xc_va = v.transform(texts_val)
        blocks_train.append(Xc_tr)
        blocks_val.append(Xc_va)
        fitted["tfidf_char"] = v

    if "sbert" in use:
        emb_tr = encode_sbert(
            texts_train,
            model_name=sbert_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=sbert_cfg.get("batch_size", 32),
            normalize=sbert_cfg.get("normalize", True),
        )
        emb_va = encode_sbert(
            texts_val,
            model_name=sbert_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=sbert_cfg.get("batch_size", 32),
            normalize=sbert_cfg.get("normalize", True),
        )
        blocks_train.append(emb_tr)
        blocks_val.append(emb_va)
        fitted["sbert"] = sbert_cfg.get("model_name")

    return stack_features(blocks_train), stack_features(blocks_val), fitted


def run_experiment(
    cfg: dict[str, Any],
    artifact_dir: str | Path = "artifacts/models",
    save: bool = True,
) -> ExperimentResult:
    """Run one full experiment and return a structured result.

    ``cfg`` is the merged dict produced by :func:`sdgtext.utils.config.load_config`.
    """
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    name = cfg.get("name", "unnamed")
    notes = cfg.get("notes", "").strip()
    log.info(f"[bold cyan]── Experiment[/]: {name}")

    # 1. Load
    data_cfg = cfg["data"]
    loaded, _issues = load_devex(
        path=data_cfg["raw_train"],
        text_columns=data_cfg["text_columns"],
        id_column=data_cfg.get("id_column"),
        meta_columns=data_cfg.get("meta_columns", []),
        label_format=data_cfg.get("label_format", "binary_wide"),
        label_column_prefix=data_cfg.get("label_column_prefix", "Label"),
        label_code_regex=data_cfg.get("label_code_regex"),
        is_test=False,
    )
    df, label_cols = loaded.df, loaded.label_cols
    Y = df[label_cols].values.astype(int)

    # 2. Preprocess
    pcfg = PreprocessConfig.from_dict(cfg.get("preprocessing", {}))
    texts = normalize_corpus(df[loaded.text_col].tolist(), pcfg)

    # 3. Split
    val_fraction = float(cfg.get("split", {}).get("val_fraction", 0.15))
    train_idx, val_idx = single_holdout_indices(Y, val_fraction=val_fraction, seed=seed)
    texts_tr = [texts[i] for i in train_idx]
    texts_va = [texts[i] for i in val_idx]
    Y_tr, Y_va = Y[train_idx], Y[val_idx]

    # 4. Features
    X_tr, X_va, _fit = _build_features(
        texts_tr, texts_va,
        feature_cfg=cfg.get("features", {}),
        sbert_cfg=cfg.get("features", {}).get("sbert", {}),
    )
    log.info(f"Features: X_train={X_tr.shape}, X_val={X_va.shape}")

    # 5. Imbalance handling (Exp 5 — no-op for others)
    imbalance_name = cfg.get("model", {}).get("imbalance_strategy", "class_weight_only")
    if isinstance(cfg.get("model", {}).get("imbalance"), dict):
        imbalance_name = cfg["model"]["imbalance"].get("active", imbalance_name)
    strategy = STRATEGIES.get(imbalance_name, STRATEGIES["class_weight_only"])
    X_tr_b, Y_tr_b, sw = strategy(X_tr, Y_tr)

    # 6. Fit — special-case Exp 8's stacked ensemble.
    head_spec = cfg.get("model", {}).get("type", "logreg_ovr")
    if head_spec.startswith("__sweep__"):
        raise ValueError(
            "Sweep configs must be expanded by the runner before calling run_experiment."
        )

    if head_spec == "stacked":
        # Train each member on its own feature subset, average their
        # probabilities at the label level. Members are described in
        # cfg.model.members; we honour their feature subsets by
        # filtering the already-stacked X tensor — which is fine for
        # the Exp 8 design where one member uses the sparse subset and
        # the other uses sparse+SBERT.
        # Simpler, equally valid implementation: refit features per
        # member. We refit so the comparison is honest.
        log.info("Stacked ensemble: refitting members on their own feature subsets")
        member_probs = []
        for member in cfg["model"]["members"]:
            sub_feat_cfg = dict(cfg.get("features", {}))
            sub_feat_cfg["use"] = list(member["features"])
            X_tr_m, X_va_m, _ = _build_features(
                texts_tr, texts_va, feature_cfg=sub_feat_cfg,
                sbert_cfg=cfg.get("features", {}).get("sbert", {}),
            )
            head_m = build_head(member["head"], cfg.get("model", {}).get(member["head"].replace("_ovr", ""), {}), seed=seed)
            try:
                head_m.fit(X_tr_m, Y_tr_b, sample_weight=sw) if sw is not None else head_m.fit(X_tr_m, Y_tr_b)
            except TypeError:
                head_m.fit(X_tr_m, Y_tr_b)
            member_probs.append(predict_proba_safe(head_m, X_va_m))
            log.info(f"  member [{member['name']}] fit on features={member['features']}")
        P_va = np.mean(member_probs, axis=0)
        model = None  # we don't persist a single sklearn estimator for the stack
    else:
        head_params = cfg.get("model", {}).get(head_spec.replace("_ovr", ""), {})
        model = build_head(head_spec, head_params, seed=seed)
        fit_kwargs = {"sample_weight": sw} if sw is not None else {}
        try:
            model.fit(X_tr_b, Y_tr_b, **fit_kwargs)
        except TypeError:
            model.fit(X_tr_b, Y_tr_b)
        P_va = predict_proba_safe(model, X_va)

    eval_cfg = cfg.get("evaluation", {})
    if eval_cfg.get("per_label_threshold", False):
        thr = tune_thresholds(
            Y_va, P_va,
            grid=eval_cfg.get("threshold_search_grid",
                              (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)),
            objective=eval_cfg.get("threshold_objective", "f1"),
        )
    else:
        thr = float(cfg.get("inference", {}).get("default_threshold", 0.5))
        thr = np.full(Y_va.shape[1], thr)

    min_lpd = int(cfg.get("inference", {}).get("min_labels_per_doc", 0))
    metrics = evaluate(Y_va, P_va, thr, min_labels_per_doc=min_lpd)
    per_label = per_label_report(
        Y_va, apply_thresholds(P_va, thr, min_lpd), label_cols
    )

    log.info(
        f"[bold green]✓ {name}[/] hamming={metrics.hamming_loss:.4f} "
        f"microF1={metrics.micro_f1:.3f} macroF1={metrics.macro_f1:.3f}"
    )

    result = ExperimentResult(
        config_name=name,
        metrics=metrics,
        per_label=per_label,
        thresholds=list(map(float, thr)),
        label_names=list(label_cols),
        notes=notes,
    )

    if save:
        out_dir = Path(artifact_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if model is not None:
            joblib.dump(model, out_dir / "model.joblib")
        with open(out_dir / "thresholds.json", "w") as f:
            json.dump({lc: float(t) for lc, t in zip(label_cols, thr)}, f, indent=2)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(result.to_json(), f, indent=2)
        log.info(f"Saved artifacts → {out_dir}")
    return result
