"""Realised volatility estimators.

The Quant Guild builds compute no realised volatility at all -- the "volatility
dashboard" regresses implied vol on future implied vol, which measures the
persistence of a forecast, not the quality of it.  The variance risk premium,
which is the thing that actually pays a short-vol position, is
``implied - realised``.  You cannot see it without this module.

All estimators return **annualised** volatility (decimal) given OHLC bars, with
the annualisation factor supplied explicitly.  ``periods_per_year=252`` for
daily bars.  Nothing here guesses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "close_to_close",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "yang_zhang",
    "realised_vol",
    "ESTIMATORS",
    "estimator_efficiency",
]

_MIN_OBS = 2


def _log(x):
    x = np.asarray(x, dtype=float)
    return np.log(np.where(x > 0, x, np.nan))


def close_to_close(df: pd.DataFrame, window: int, periods_per_year: int = 252,
                   demean: bool = False) -> pd.Series:
    """Classic estimator: rolling std of log returns.

    ``demean=False`` (the default) uses the zero-mean form
    ``sqrt(mean(r^2))``, which is the right choice at short horizons where the
    drift is unidentifiable and subtracting a noisy sample mean adds variance
    to the estimate.
    """
    r = _log(df["close"]).astype(float)
    r = pd.Series(r, index=df.index).diff()
    if demean:
        v = r.rolling(window, min_periods=_MIN_OBS).var(ddof=1)
    else:
        v = (r**2).rolling(window, min_periods=_MIN_OBS).mean()
    return np.sqrt(v * periods_per_year)


def parkinson(df: pd.DataFrame, window: int, periods_per_year: int = 252) -> pd.Series:
    """High-low range estimator. ~5x more efficient than close-to-close for a
    pure driftless GBM, but biased **down** when the price is not observed
    continuously (discrete trading truncates the true high and low)."""
    hl = pd.Series(_log(df["high"]) - _log(df["low"]), index=df.index)
    v = (hl**2).rolling(window, min_periods=_MIN_OBS).mean() / (4.0 * np.log(2.0))
    return np.sqrt(v * periods_per_year)


def garman_klass(df: pd.DataFrame, window: int, periods_per_year: int = 252) -> pd.Series:
    """Uses the whole OHLC bar. More efficient than Parkinson, still assumes
    zero drift and no overnight gap."""
    hl = pd.Series(_log(df["high"]) - _log(df["low"]), index=df.index)
    co = pd.Series(_log(df["close"]) - _log(df["open"]), index=df.index)
    per_bar = 0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2
    v = per_bar.rolling(window, min_periods=_MIN_OBS).mean()
    return np.sqrt(v.clip(lower=0) * periods_per_year)


def rogers_satchell(df: pd.DataFrame, window: int, periods_per_year: int = 252) -> pd.Series:
    """Drift-robust OHLC estimator: unbiased when the process has a non-zero
    drift, which matters over an earnings window where the drift is not zero."""
    h, l_, c, o = (_log(df[k]) for k in ("high", "low", "close", "open"))
    per_bar = pd.Series((h - c) * (h - o) + (l_ - c) * (l_ - o), index=df.index)
    v = per_bar.rolling(window, min_periods=_MIN_OBS).mean()
    return np.sqrt(v.clip(lower=0) * periods_per_year)


def yang_zhang(df: pd.DataFrame, window: int, periods_per_year: int = 252) -> pd.Series:
    """Yang-Zhang: the only common estimator that handles **both** drift and
    overnight gaps.  For single stocks around earnings, where the entire move is
    an overnight gap, the gap term is not a nuisance -- it is the signal -- so
    this is the estimator to prefer for event work.

    ``V = V_overnight + k*V_open_to_close + (1-k)*V_rogers_satchell`` with
    ``k = 0.34 / (1.34 + (n+1)/(n-1))``.
    """
    n = int(window)
    if n < 3:
        raise ValueError("yang_zhang needs window >= 3")
    o, h, l_, c = (_log(df[k]) for k in ("open", "high", "low", "close"))
    o = pd.Series(o, index=df.index)
    c = pd.Series(c, index=df.index)
    overnight = o - c.shift(1)
    open_to_close = c - o
    v_on = overnight.rolling(n, min_periods=_MIN_OBS).var(ddof=1)
    v_oc = open_to_close.rolling(n, min_periods=_MIN_OBS).var(ddof=1)
    v_rs = (rogers_satchell(df, n, periods_per_year=1) ** 2)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    v = v_on + k * v_oc + (1.0 - k) * v_rs
    return np.sqrt(v.clip(lower=0) * periods_per_year)


ESTIMATORS = {
    "close_to_close": close_to_close,
    "parkinson": parkinson,
    "garman_klass": garman_klass,
    "rogers_satchell": rogers_satchell,
    "yang_zhang": yang_zhang,
}


def realised_vol(df: pd.DataFrame, window: int, method: str = "yang_zhang",
                 periods_per_year: int = 252) -> pd.Series:
    """Dispatch to a named estimator. Raises on an unknown name rather than
    falling back to a default -- a typo must not silently change the estimator."""
    try:
        fn = ESTIMATORS[method]
    except KeyError:
        raise ValueError(
            f"unknown estimator {method!r}; choose from {sorted(ESTIMATORS)}"
        ) from None
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if method != "close_to_close" and missing:
        raise ValueError(f"{method} needs columns {sorted(required)}, missing {sorted(missing)}")
    return fn(df, window, periods_per_year=periods_per_year)


def estimator_efficiency(df: pd.DataFrame, window: int, true_vol: float,
                         periods_per_year: int = 252) -> pd.DataFrame:
    """Compare estimators against a known volatility.

    Only meaningful on simulated data where ``true_vol`` is actually known --
    which is exactly why the synthetic data provider exists.  Reports bias and
    relative variance so you can justify your estimator choice with a number
    instead of a preference.
    """
    rows = []
    for name in ESTIMATORS:
        try:
            s = realised_vol(df, window, method=name, periods_per_year=periods_per_year).dropna()
        except ValueError:
            continue
        if s.empty:
            continue
        rows.append(
            {
                "estimator": name,
                "mean": float(s.mean()),
                "bias": float(s.mean() - true_vol),
                "bias_pct": float(100.0 * (s.mean() / true_vol - 1.0)),
                "std": float(s.std(ddof=1)),
                "rmse": float(np.sqrt(np.mean((s - true_vol) ** 2))),
                "n": int(s.size),
            }
        )
    out = pd.DataFrame(rows).set_index("estimator")
    if not out.empty:
        out["rel_efficiency"] = out["rmse"].min() / out["rmse"]
    return out.sort_values("rmse")
