"""Tiny end-to-end smoke test on synthetic data.

Run this *without* the Devex CSVs to confirm the package imports cleanly
and the pipeline plumbing works. Useful for graders who want to sanity-
check the repo before downloading the dataset.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdgtext.data.preprocess import PreprocessConfig, normalize_corpus  # noqa: E402
from sdgtext.eval.metrics import evaluate  # noqa: E402
from sdgtext.eval.thresholds import tune_thresholds  # noqa: E402
from sdgtext.features.tfidf import make_char_tfidf, make_word_tfidf  # noqa: E402
from sdgtext.features.combine import stack_features  # noqa: E402
from sdgtext.models.heads import build_head, predict_proba_safe  # noqa: E402

rng = np.random.default_rng(0)
N, L = 120, 5
texts = [
    "Vaccine campaign reduces measles incidence in five districts." if i % 3 == 0
    else "Programme on maternal mortality reduction in rural clinics." if i % 3 == 1
    else "Antiretroviral therapy access expansion in West Africa." for i in range(N)
]
Y = rng.integers(0, 2, size=(N, L))
Y[:, 0] = (np.arange(N) % 3 == 0).astype(int)  # signal on label 0

texts = normalize_corpus(texts, PreprocessConfig())

vw = make_word_tfidf({"min_df": 1, "max_features": 2000, "stop_words": None})
vc = make_char_tfidf({"min_df": 1, "max_features": 2000})
Xw = vw.fit_transform(texts)
Xc = vc.fit_transform(texts)
X = stack_features([Xw, Xc])

cut = int(0.8 * N)
Xtr, Xva = X[:cut], X[cut:]
Ytr, Yva = Y[:cut], Y[cut:]

model = build_head("logreg_ovr", {"C": 1.0, "max_iter": 500})
model.fit(Xtr, Ytr)
P = predict_proba_safe(model, Xva)
thr = tune_thresholds(Yva, P)
m = evaluate(Yva, P, thr)
print("smoke-test OK:", m.as_dict())
