"""Correlation with human scores: SRCC, KRCC, PLCC (raw and after VQEG logistic fit)."""

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from humper.metrics import METRICS, aligned


def logistic5(x, b1, b2, b3, b4, b5):
    """VQEG 5-parameter monotonic logistic mapping (Sheikh et al., TIP 2006)."""
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Map metric scores x onto the MOS scale; None if the fit does not converge.
    x is z-scored first so one init works for every metric."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = (x - x.mean()) / (x.std() or 1.0)
    p0 = [np.ptp(y), 1.0, 0.0, 0.0, y.mean()]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (RuntimeWarning, optimize.OptimizeWarning))
            params, _ = optimize.curve_fit(logistic5, z, y, p0=p0, maxfev=20000)
    except RuntimeError:
        return None
    return logistic5(z, *params)


def srcc(x, y) -> float:
    return float(stats.spearmanr(x, y).statistic)


def krcc(x, y) -> float:
    return float(stats.kendalltau(x, y).statistic)


def plcc(x, y, fit: str | None = "logistic5") -> float:
    """Pearson correlation; with fit='logistic5' the metric is first mapped by `fit_logistic`
    (falls back to raw x if the fit does not converge; see `correlations` for the flag)."""
    x = np.asarray(x, float)
    if fit == "logistic5":
        fitted = fit_logistic(x, y)
        x = x if fitted is None else fitted
    return float(stats.pearsonr(x, y).statistic)


def correlations(x, y) -> dict:
    fitted = fit_logistic(x, y)
    return {
        "srcc": srcc(x, y),
        "krcc": krcc(x, y),
        "plcc_raw": plcc(x, y, fit=None),
        "plcc": plcc(x, y, fit=None) if fitted is None else float(stats.pearsonr(fitted, y).statistic),
        "fit_ok": fitted is not None,
        "n": len(y),
    }


def table(scores: pd.DataFrame, by: str | None = None, metrics: list[str] | None = None) -> pd.DataFrame:
    """Long table: one row per (group, metric). Metrics are direction-aligned before correlating."""
    names = metrics or [m.name for m in METRICS if m.name in scores]
    groups = [("all", scores)] if by is None else scores.groupby(by, sort=True)
    rows = []
    for key, g in groups:
        for name in names:
            rows.append({by or "subset": key, "metric": name, **correlations(aligned(g, name), g["mos"])})
    return pd.DataFrame(rows)
