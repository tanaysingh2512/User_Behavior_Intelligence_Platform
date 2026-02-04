from __future__ import annotations

import numpy as np
from scipy import stats


def _safe_bins(values: np.ndarray, bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([values.min() - 1e-6, values.max() + 1e-6])
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return edges


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected.astype(float)
    actual = actual.astype(float)
    edges = _safe_bins(expected, bins)
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    return np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))


def psi_from_bins(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)
    return float(np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct)))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    return float(stats.ks_2samp(expected, actual).statistic)
