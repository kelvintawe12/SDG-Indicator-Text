"""Tiny YAML config loader with one-level ``inherits`` support.

Avoids hydra/omegaconf to keep the dependency surface small and the
Colab cold-start fast. The override semantics: keys from the child
config replace keys in the parent at every nesting level (deep merge),
lists are replaced wholesale rather than merged element-wise.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single optional ``inherits:`` key.

    ``inherits`` is interpreted relative to the configs/ root. We do not
    support multi-level inheritance chains by design — one parent is
    sufficient for this project and avoids surprising override order.
    """
    p = Path(path)
    if not p.is_absolute():
        # allow callers to pass either "experiments/exp01..." or absolute paths
        candidate = CONFIG_ROOT / p
        p = candidate if candidate.exists() else p
    with open(p, "r") as f:
        cfg = yaml.safe_load(f) or {}
    parent_name = cfg.pop("inherits", None)
    if parent_name:
        parent_path = CONFIG_ROOT / parent_name
        with open(parent_path, "r") as f:
            parent = yaml.safe_load(f) or {}
        cfg = _deep_merge(parent, cfg)
    return cfg
