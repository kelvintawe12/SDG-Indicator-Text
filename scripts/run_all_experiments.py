"""Run every experiment under configs/experiments/ and produce the
artifacts the report needs:

* ``reports/experiment_summary.csv`` — comparison table (all metrics).
* ``reports/figures/experiment_progression.png`` — bar+line plot.
* ``reports/figures/per_label_f1.png`` — head/tail comparison.
* ``reports/figures/eda_overview.png`` + ``cooccurrence.png`` — EDA.
* ``artifacts/models/<exp>/{model.joblib, metrics.json, thresholds.json}``.

Sweep configs (Exp 4: classifier sweep; Exp 5: imbalance sweep) are
expanded into one run per arm so each appears as its own row in the
summary table.

This script is what the notebook calls under the hood; running it from
the command line is just a more debuggable entry point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdgtext.data.load import load_devex  # noqa: E402
from sdgtext.eval import eda as eda_mod  # noqa: E402
from sdgtext.models.pipeline import run_experiment  # noqa: E402
from sdgtext.utils.config import load_config  # noqa: E402
from sdgtext.utils.logging import get_logger  # noqa: E402
from sdgtext.utils.seeding import seed_everything  # noqa: E402

log = get_logger("run_all")
seed_everything(42)

EXP_DIR = ROOT / "configs" / "experiments"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

sns.set_context("notebook")
sns.set_style("whitegrid")


def _expand(cfg: dict) -> list[dict]:
    """Expand sweep configs into one config per arm."""
    if cfg.get("model", {}).get("type") == "__sweep__":
        out = []
        for arm in cfg["model"]["sweep"]:
            c = json.loads(json.dumps(cfg))
            c["model"]["type"] = arm
            c["name"] = f"{cfg['name']}__{arm}"
            out.append(c)
        return out
    if isinstance(cfg.get("model", {}).get("imbalance"), dict) and "sweep" in cfg["model"]["imbalance"]:
        out = []
        for arm in cfg["model"]["imbalance"]["sweep"]:
            c = json.loads(json.dumps(cfg))
            c["model"]["imbalance"] = {"active": arm}
            c["name"] = f"{cfg['name']}__{arm}"
            out.append(c)
        return out
    return [cfg]


def write_eda_figures():
    cfg = load_config("default.yaml")
    ds, _ = load_devex(
        path=cfg["data"]["raw_train"],
        text_columns=cfg["data"]["text_columns"],
        id_column=cfg["data"].get("id_column"),
        meta_columns=cfg["data"].get("meta_columns", []),
        label_format=cfg["data"].get("label_format", "binary_wide"),
        label_column_prefix=cfg["data"].get("label_column_prefix", "Label"),
        label_code_regex=cfg["data"].get("label_code_regex"),
        is_test=False,
    )
    df, lc = ds.df, ds.label_cols
    freq = eda_mod.label_frequency_table(df, lc)
    card = eda_mod.cardinality_stats(df, lc)
    log.info(f"EDA: {card}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.barplot(data=freq, x="label", y="positives", ax=axes[0], color="steelblue")
    axes[0].set_title("Per-label positive counts (head→tail)")
    axes[0].tick_params(axis="x", rotation=70)
    axes[1].hist([len(t.split()) for t in df[ds.text_col]], bins=40, color="steelblue")
    axes[1].set(title="Document length (words)", xlabel="tokens", ylabel="documents")
    plt.tight_layout()
    plt.savefig(FIGS / "eda_overview.png", dpi=140, bbox_inches="tight")
    plt.close()

    cooc = eda_mod.cooccurrence_matrix(df, lc)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cooc, annot=False, cmap="magma", ax=ax)
    ax.set_title("Label co-occurrence (joint positive counts)")
    plt.tight_layout()
    plt.savefig(FIGS / "cooccurrence.png", dpi=140, bbox_inches="tight")
    plt.close()
    log.info(f"EDA figures → {FIGS}/eda_overview.png + cooccurrence.png")


def write_results_figures(results):
    summary = pd.DataFrame([
        {"experiment": r.config_name, **r.metrics.as_dict()} for r in results
    ]).sort_values("hamming_loss").reset_index(drop=True)
    summary.to_csv(REPORTS / "experiment_summary.csv", index=False)

    # Figure 1: bar (Hamming) + line (Macro-F1) per experiment.
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    order = summary["experiment"].tolist()
    ax1.bar(order, summary["hamming_loss"], color="#4c72b0", label="Hamming Loss (↓)")
    ax1.set_ylabel("Hamming Loss")
    ax1.tick_params(axis="x", rotation=70)
    ax1.set_ylim(0, max(summary["hamming_loss"].max() * 1.25, 0.05))
    ax2 = ax1.twinx()
    ax2.plot(order, summary["macro_f1"], color="#dd8452", marker="o", label="Macro-F1 (↑)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Macro-F1")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGS / "experiment_progression.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Figure 2: per-label F1 — baseline vs best (by Hamming).
    by_name = {r.config_name: r for r in results}
    best_name = summary.iloc[0]["experiment"]
    baseline_name = next((n for n in by_name if n.startswith("exp01")), None)
    if baseline_name and best_name != baseline_name:
        rows = []
        for lc in by_name[baseline_name].label_names:
            rows.append({
                "label": lc.replace("lbl__", ""),
                "baseline_f1": by_name[baseline_name].per_label[lc]["f1"],
                "best_f1": by_name[best_name].per_label[lc]["f1"],
                "support": by_name[best_name].per_label[lc]["support"],
            })
        perlbl = pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        x = np.arange(len(perlbl))
        ax.bar(x - 0.2, perlbl["baseline_f1"], width=0.4, label=f"Baseline (Exp 1)")
        ax.bar(x + 0.2, perlbl["best_f1"], width=0.4, label=f"Best ({best_name})")
        ax.set_xticks(x); ax.set_xticklabels(perlbl["label"], rotation=70)
        ax.set_ylabel("F1"); ax.set_title("Per-label F1: baseline vs best (sorted by support)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGS / "per_label_f1.png", dpi=140, bbox_inches="tight")
        plt.close()
        perlbl.to_csv(REPORTS / "per_label_comparison.csv", index=False)
    log.info(f"Results figures + summary → {REPORTS}")
    return summary


def main():
    write_eda_figures()
    paths = sorted(EXP_DIR.glob("exp*.yaml"))
    results = []
    for p in paths:
        cfg = load_config(p)
        for sub in _expand(cfg):
            log.info(f"[bold magenta]>>>[/] {sub['name']}")
            results.append(run_experiment(sub))
    summary = write_results_figures(results)
    print("\n=== SUMMARY (sorted by Hamming Loss ↓) ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {REPORTS / 'experiment_summary.csv'}")


if __name__ == "__main__":
    main()
