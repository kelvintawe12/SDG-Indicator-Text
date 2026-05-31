"""Stack sparse TF-IDF and dense SBERT into a single design matrix.

scipy ``hstack`` handles mixed sparse + dense if we wrap the dense
block in a ``csr_matrix``. We L2-normalize the dense block first so its
columns don't dominate the L2 regularizer of downstream Logistic
Regression — the SBERT vector arrives unit-norm but its columns have a
much larger per-feature scale than the sparse TF-IDF columns.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp


def stack_features(blocks: Iterable[sp.spmatrix | np.ndarray]) -> sp.csr_matrix:
    """Concatenate feature blocks horizontally.

    Dense numpy blocks are wrapped in a CSR matrix; sparse blocks are
    cast to CSR if they aren't already. Result is always CSR.
    """
    csr_blocks: list[sp.csr_matrix] = []
    for b in blocks:
        if b is None:
            continue
        if sp.issparse(b):
            csr_blocks.append(b.tocsr())
        else:
            arr = np.asarray(b)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            csr_blocks.append(sp.csr_matrix(arr))
    if not csr_blocks:
        raise ValueError("stack_features called with no usable blocks.")
    if len(csr_blocks) == 1:
        return csr_blocks[0]
    return sp.hstack(csr_blocks, format="csr")
