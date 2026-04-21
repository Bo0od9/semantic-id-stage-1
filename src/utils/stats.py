from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as _sps


def wilcoxon_paired(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    assert arr_a.shape == arr_b.shape, "paired arrays must have the same shape"
    diff = arr_a - arr_b
    if np.all(diff == 0.0):
        return 0.0, 1.0
    result = _sps.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def bonferroni(
    pvalues: Sequence[float],
    alpha: float = 0.05,
    n: int | None = None,
) -> list[bool]:
    pvs = list(pvalues)
    if n is None:
        n = len(pvs)
    threshold = alpha / n
    return [p < threshold for p in pvs]


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    assert arr.ndim == 1
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi
