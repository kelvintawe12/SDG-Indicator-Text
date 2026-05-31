"""Refit the winning ensemble on full train, predict on the test set.

This is the script that produces the submission CSV cited by the
report's "Inference" section. The thresholds saved by Exp 8 (tuned on
the validation fold) are reused unchanged — we do *not* re-tune them
on a refit model, because that would require a second hold-out and the
thresholds we saved are already the legitimate choice we'd report.

Outputs
-------
* ``artifacts/predictions/submission.csv`` — multi-hot CSV in the same
  row order as the test set, with the original ``Unique ID`` column.
* ``artifacts/predictions/submission_long.csv`` — same predictions in
  long form (one row per (doc, predicted indicator)), useful for
  auditing.
* ``artifacts/predictions/test_probabilities.npy`` — raw probability
  matrix for downstream analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdgtext.data.load import load_devex  # noqa: E402
from sdgtext.data.preprocess import PreprocessConfig, normalize_corpus  # noqa: E402
from sdgtext.eval.metrics import apply_thresholds  # noqa: E402
from sdgtext.features.combine import stack_features  # noqa: E402
from sdgtext.features.embeddings import encode_sbert  # noqa: E402
from sdgtext.features.tfidf import make_char_tfidf, make_word_tfidf  # noqa: E402
from sdgtext.models.heads import build_head, predict_proba_safe  # noqa: E402
from sdgtext.utils.config import load_config  # noqa: E402
from sdgtext.utils.logging import get_logger  # noqa: E402
from sdgtext.utils.seeding import seed_everything  # noqa: E402

log = get_logger("predict")
seed_everything(42)

WINNER = "exp08_calibrated_ensemble"
CFG_PATH = ROOT / "configs" / "experiments" / f"{WINNER}.yaml"
OUT_DIR = ROOT / "artifacts" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config(CFG_PATH)

    # 1. Load full training set with multi-hot labels.
    log.info("Loading training data (full)…")
    train_ds, _ = load_devex(
        path=cfg["data"]["raw_train"],
        text_columns=cfg["data"]["text_columns"],
        id_column=cfg["data"].get("id_column"),
        meta_columns=cfg["data"].get("meta_columns", []),
        label_format=cfg["data"].get("label_format"),
        label_column_prefix=cfg["data"].get("label_column_prefix"),
        label_code_regex=cfg["data"].get("label_code_regex"),
        is_test=False,
    )
    df_tr, label_cols = train_ds.df, train_ds.label_cols
    Y_tr = df_tr[label_cols].values.astype(int)

    # 2. Load test set (no labels, just text).
    log.info("Loading test data…")
    test_ds, _ = load_devex(
        path=cfg["data"]["raw_test"],
        text_columns=cfg["data"]["text_columns"],
        id_column=cfg["data"].get("id_column"),
        meta_columns=cfg["data"].get("meta_columns", []),
        label_format=cfg["data"].get("label_format"),
        label_column_prefix=cfg["data"].get("label_column_prefix"),
        label_code_regex=cfg["data"].get("label_code_regex"),
        is_test=True,
    )
    df_te = test_ds.df
    log.info(f"train={len(df_tr)} rows, test={len(df_te)} rows, labels={len(label_cols)}")

    # 3. Preprocess both with the winning config.
    pcfg = PreprocessConfig.from_dict(cfg["preprocessing"])
    texts_tr = normalize_corpus(df_tr[train_ds.text_col].tolist(), pcfg)
    texts_te = normalize_corpus(df_te[test_ds.text_col].tolist(), pcfg)

    # 4. Build features — refit vectorizers on the FULL training corpus
    #    so the test transform sees every training term.
    log.info("Fitting word + char TF-IDF on full train…")
    v_word = make_word_tfidf(cfg["features"]["tfidf_word"])
    v_char = make_char_tfidf(cfg["features"]["tfidf_char"])
    Xw_tr = v_word.fit_transform(texts_tr); Xw_te = v_word.transform(texts_te)
    Xc_tr = v_char.fit_transform(texts_tr); Xc_te = v_char.transform(texts_te)
    log.info(f"word={Xw_tr.shape}, char={Xc_tr.shape}")

    log.info("Encoding train + test with SBERT (cached if previously run)…")
    sbert_cfg = cfg["features"]["sbert"]
    Xe_tr = encode_sbert(texts_tr, **sbert_cfg)
    Xe_te = encode_sbert(texts_te, **sbert_cfg)
    log.info(f"sbert_train={Xe_tr.shape}, sbert_test={Xe_te.shape}")

    # 5. Train Exp 8's two members on full train, average probabilities.
    member_specs = cfg["model"]["members"]
    feature_map_tr = {"tfidf_word": Xw_tr, "tfidf_char": Xc_tr, "sbert": Xe_tr}
    feature_map_te = {"tfidf_word": Xw_te, "tfidf_char": Xc_te, "sbert": Xe_te}
    member_probs_te: list[np.ndarray] = []
    fitted_members = []
    for m in member_specs:
        log.info(f"Fitting ensemble member [{m['name']}] on features={m['features']}")
        X_tr_m = stack_features([feature_map_tr[f] for f in m["features"]])
        X_te_m = stack_features([feature_map_te[f] for f in m["features"]])
        head = build_head(m["head"], cfg["model"].get(m["head"].replace("_ovr", ""), {}), seed=42)
        head.fit(X_tr_m, Y_tr)
        member_probs_te.append(predict_proba_safe(head, X_te_m))
        fitted_members.append((m["name"], head))

    P_te = np.mean(member_probs_te, axis=0)
    log.info(f"Ensemble probabilities: {P_te.shape}, mean={P_te.mean():.3f}")

    # 6. Load the per-label thresholds Exp 8 saved during the validated run.
    thr_path = ROOT / "artifacts" / "models" / WINNER / "thresholds.json"
    thr_dict = json.loads(thr_path.read_text())
    thr = np.array([thr_dict[c] for c in label_cols], dtype=float)
    log.info(f"Thresholds — min={thr.min():.3f} median={np.median(thr):.3f} max={thr.max():.3f}")

    # 7. Apply thresholds, force min_labels_per_doc=1.
    min_lpd = int(cfg["inference"].get("min_labels_per_doc", 1))
    Y_te_pred = apply_thresholds(P_te, thr, min_labels_per_doc=min_lpd)
    log.info(
        f"Test predictions — total positives={int(Y_te_pred.sum())}, "
        f"avg labels/doc={Y_te_pred.sum(axis=1).mean():.2f}, "
        f"docs with ≥1 label={int((Y_te_pred.sum(axis=1) >= 1).sum())}/{len(Y_te_pred)}"
    )

    # 8. Build the submission CSV using the indicator codes (not the
    # internal 'lbl__' prefix) — what graders expect to see.
    pretty_cols = [c.replace("lbl__", "") for c in label_cols]
    sub_wide = pd.DataFrame(Y_te_pred, columns=pretty_cols)
    if test_ds.id_col and test_ds.id_col in df_te.columns:
        sub_wide.insert(0, test_ds.id_col, df_te[test_ds.id_col].values)
    if "Type" in df_te.columns:
        sub_wide.insert(1, "Type", df_te["Type"].values)
    sub_path = OUT_DIR / "submission.csv"
    sub_wide.to_csv(sub_path, index=False)
    log.info(f"✓ Wide submission → {sub_path} ({len(sub_wide)} rows)")

    # 9. Long-form audit CSV (one row per (doc, predicted indicator, prob)).
    long_rows = []
    for i, row in enumerate(Y_te_pred):
        for j in np.where(row == 1)[0]:
            long_rows.append({
                "Unique ID": df_te[test_ds.id_col].iloc[i] if test_ds.id_col else i,
                "indicator": pretty_cols[j],
                "probability": float(P_te[i, j]),
                "threshold": float(thr[j]),
            })
    long_df = pd.DataFrame(long_rows)
    long_path = OUT_DIR / "submission_long.csv"
    long_df.to_csv(long_path, index=False)
    log.info(f"✓ Long-form predictions → {long_path} ({len(long_df)} rows)")

    np.save(OUT_DIR / "test_probabilities.npy", P_te)
    np.save(OUT_DIR / "test_predictions.npy", Y_te_pred)

    # 10. Persist the fitted inference bundle for the CLI predict command.
    bundle = {
        "vectorizers": {"word": v_word, "char": v_char},
        "sbert_model": sbert_cfg["model_name"],
        "members": [(name, head) for name, head in fitted_members],
        "thresholds": thr.tolist(),
        "label_names": list(label_cols),
        "pretty_label_names": pretty_cols,
        "min_labels_per_doc": min_lpd,
        "preprocessing": cfg["preprocessing"],
    }
    bundle_path = ROOT / "artifacts" / "models" / WINNER / "inference_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    log.info(f"✓ Inference bundle → {bundle_path}")

    # 11. Per-label prediction distribution on the test set — useful sanity check.
    per_label = pd.DataFrame({
        "indicator": pretty_cols,
        "test_positives": Y_te_pred.sum(axis=0).astype(int),
        "test_prevalence": Y_te_pred.mean(axis=0).round(4),
        "threshold": thr.round(3),
        "train_positives": Y_tr.sum(axis=0).astype(int),
        "train_prevalence": Y_tr.mean(axis=0).round(4),
    }).sort_values("train_positives", ascending=False)
    per_label.to_csv(OUT_DIR / "per_label_test_distribution.csv", index=False)
    print("\n--- Per-label distribution: test vs train ---")
    print(per_label.to_string(index=False))


if __name__ == "__main__":
    main()
