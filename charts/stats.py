"""Seed aggregation and paired intervals.

Every level plot is mean +/- 1 SD across seeds, with the SD taken within one scenario: the
seeds are the replicates, the scenarios are different conditions. Comparison plots use the
CRN pairing instead: two arms on the same (scenario, seed) saw the identical environment,
so the per-pair difference isolates the algorithm and a t-interval on it is the right bar.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

# Two-sided 95% Student-t critical values by degrees of freedom (scipy is not a dependency).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
         34: 2.032, 40: 2.021, 60: 2.000, 120: 1.980}


def t_crit(df: int) -> float:
    """95% two-sided t critical value; conservative (next lower df) between table rows."""
    if df < 1:
        return math.nan
    if df in _T975:
        return _T975[df]
    lower = [k for k in _T975 if k < df]
    return _T975[max(lower)] if df <= 120 else 1.96


def mean_sd(values) -> tuple[float, float, int]:
    """Mean, sample SD (ddof=1; 0 for a single value) and n of the finite values."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, 0
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), sd, int(arr.size)


def paired_ci(differences) -> tuple[float, float, float, int]:
    """Mean paired difference with its 95% t-interval: (mean, low, high, n)."""
    mean, sd, n = mean_sd(differences)
    if n < 2:
        return mean, math.nan, math.nan, n
    half = t_crit(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half, n


def summarise(grid: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    """mean / sd / n of ``metric`` across seeds for each group in ``by``."""
    rows = []
    for key, group in grid.groupby(by, sort=False, dropna=False):
        mean, sd, n = mean_sd(group[metric].tolist())
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "mean": mean, "sd": sd, "n": n})
    return pd.DataFrame(rows)


def paired_differences(grid: pd.DataFrame, arm: str, reference: str, metric: str,
                       scenario: int | None = None) -> list[float]:
    """arm - reference on ``metric`` for every (scenario, seed) both completed."""
    subset = grid if scenario is None else grid[grid["scenario"] == scenario]
    a = subset[subset["arm"] == arm].set_index(["scenario", "seed"])[metric]
    b = subset[subset["arm"] == reference].set_index(["scenario", "seed"])[metric]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return (joined["a"] - joined["b"]).tolist()


def seed_band(series_per_seed: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-timestep mean and SD (ddof=1) across seeds; NaN-aware, SD 0 for one seed."""
    stack = np.vstack(series_per_seed)
    counts = np.sum(np.isfinite(stack), axis=0)
    with warnings.catch_warnings():
        # All-NaN columns (e.g. t=0 of gamma.txt) legitimately give NaN; don't warn.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1])
    sd = np.where(counts > 1, sd, 0.0)
    return mean, sd


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average over the last ``window`` steps (min_periods=1)."""
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()
