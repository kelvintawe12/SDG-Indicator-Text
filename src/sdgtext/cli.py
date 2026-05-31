"""Command-line entry point.

Usage examples (after ``pip install -e .``):

    sdgtext train --config configs/experiments/exp01_baseline_tfidf_lr.yaml
    sdgtext run-all
    sdgtext predict \\
        --model artifacts/models/exp08_calibrated_ensemble \\
        --test data/raw/Devex_test_questions.csv \\
        --out artifacts/predictions/submission.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from sdgtext.utils.config import CONFIG_ROOT, load_config
from sdgtext.utils.logging import get_logger

log = get_logger("sdgtext.cli")


@click.group()
def main() -> None:
    """SDG-3 indicator multi-label text classifier."""


@main.command("train")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--artifact-dir", default="artifacts/models", type=click.Path())
def train_cmd(config_path: str, artifact_dir: str) -> None:
    """Train a single experiment from a YAML config."""
    from sdgtext.models.pipeline import run_experiment  # late import to keep CLI snappy

    cfg = load_config(config_path)
    result = run_experiment(cfg, artifact_dir=artifact_dir, save=True)
    click.echo(json.dumps(result.metrics.as_dict(), indent=2))


@main.command("run-all")
@click.option("--artifact-dir", default="artifacts/models", type=click.Path())
def run_all_cmd(artifact_dir: str) -> None:
    """Run every experiment under configs/experiments/ in order.

    Sweep configs (Exp 4 classifier sweep, Exp 5 imbalance sweep) are
    expanded into one run per arm so the results table reflects each
    arm independently.
    """
    from sdgtext.models.pipeline import run_experiment

    exp_dir = CONFIG_ROOT / "experiments"
    paths = sorted(p for p in exp_dir.glob("exp*.yaml"))
    results = []
    for path in paths:
        cfg = load_config(path)
        # Expand Exp 4 sweep
        if cfg.get("model", {}).get("type") == "__sweep__":
            for arm in cfg["model"]["sweep"]:
                cfg_arm = json_clone(cfg)
                cfg_arm["model"]["type"] = arm
                cfg_arm["name"] = f"{cfg['name']}__{arm}"
                results.append(run_experiment(cfg_arm, artifact_dir=artifact_dir))
            continue
        # Expand Exp 5 imbalance sweep
        if isinstance(cfg.get("model", {}).get("imbalance"), dict) and "sweep" in cfg["model"]["imbalance"]:
            for arm in cfg["model"]["imbalance"]["sweep"]:
                cfg_arm = json_clone(cfg)
                cfg_arm["model"]["imbalance"] = {"active": arm}
                cfg_arm["name"] = f"{cfg['name']}__{arm}"
                results.append(run_experiment(cfg_arm, artifact_dir=artifact_dir))
            continue
        results.append(run_experiment(cfg, artifact_dir=artifact_dir))

    # Final summary table to stdout (the notebook also renders this)
    rows = [
        {"experiment": r.config_name, **r.metrics.as_dict()} for r in results
    ]
    click.echo(json.dumps(rows, indent=2))
    summary_path = Path(artifact_dir) / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    log.info(f"Summary → {summary_path}")


@main.command("predict")
@click.option("--model", "model_dir", required=True, type=click.Path(exists=True))
@click.option("--test", "test_csv", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_csv", required=True, type=click.Path())
def predict_cmd(model_dir: str, test_csv: str, config_path: str, out_csv: str) -> None:
    """Generate a submission CSV using a saved model + thresholds."""
    import joblib
    import numpy as np
    import pandas as pd

    from sdgtext.data.load import load_devex
    from sdgtext.data.preprocess import PreprocessConfig, normalize_corpus
    from sdgtext.eval.metrics import apply_thresholds
    from sdgtext.features.tfidf import make_word_tfidf, make_char_tfidf  # noqa: F401
    from sdgtext.models.heads import predict_proba_safe

    cfg = load_config(config_path)
    loaded, _ = load_devex(
        path=test_csv,
        text_columns=cfg["data"]["text_columns"],
        id_column=cfg["data"].get("id_column"),
        meta_columns=cfg["data"].get("meta_columns", []),
        label_format=cfg["data"].get("label_format", "binary_wide"),
        label_column_prefix=cfg["data"].get("label_column_prefix", "Label"),
        label_code_regex=cfg["data"].get("label_code_regex"),
        is_test=True,
    )
    pcfg = PreprocessConfig.from_dict(cfg.get("preprocessing", {}))
    texts = normalize_corpus(loaded.df[loaded.text_col].tolist(), pcfg)

    # We expect the model artifact to ship its own fitted feature
    # pipeline. For the scaffold we save just the head; the notebook
    # demonstrates the full pipeline pickle. This command currently
    # accepts a (vectorizer, model, thresholds) bundle if one exists.
    bundle_path = Path(model_dir) / "pipeline.joblib"
    if not bundle_path.exists():
        raise click.ClickException(
            f"{bundle_path} not found. Train via the notebook export-pipeline "
            "step to produce a self-contained inference bundle."
        )
    bundle = joblib.load(bundle_path)
    X = bundle["features"].transform(texts)
    P = predict_proba_safe(bundle["model"], X)
    thr = np.array(bundle["thresholds"])
    y_pred = apply_thresholds(P, thr, min_labels_per_doc=cfg["inference"].get("min_labels_per_doc", 0))

    out_df = pd.DataFrame(y_pred, columns=bundle["label_names"])
    if loaded.id_col and loaded.id_col in loaded.df.columns:
        out_df.insert(0, loaded.id_col, loaded.df[loaded.id_col].values)
    out_df.to_csv(out_csv, index=False)
    click.echo(f"Wrote {len(out_df)} predictions → {out_csv}")


def json_clone(d):
    """Deep clone via JSON; safe because configs are plain data."""
    return json.loads(json.dumps(d))


if __name__ == "__main__":
    main()
