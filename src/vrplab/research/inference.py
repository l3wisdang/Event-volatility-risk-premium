"""Inference tools for overlapping, autocorrelated financial samples.

This is the module that would have changed the conclusion of the Quant Guild
volatility dashboard.  That build regresses a 30-day forward average of a
30-day implied vol index on the same index, sampled daily, and reads the
``scipy.stats.linregress`` p-value off the result.  Two independent overlaps
compound there:

  1. consecutive forward windows share 29 of 30 days;
  2. each observation of the regressor is itself a ~30-day forward-looking
     quantity, so consecutive *levels* are already ~97% the same information.

The residuals are therefore massively autocorrelated, the OLS standard errors
are too small by roughly sqrt(30) ~ 5.5x, and the reported p-value is not a
p-value.  Everything here exists to stop that happening.

Nothing in this module reports a t-statistic without also reporting the
effective sample size behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "HACResult",
    "ols_hac",
    "newey_west_lag",
    "effective_sample_size",
    "circular_block_bootstrap",
    "stationary_bootstrap",
    "bootstrap_ci",
    "deflated_sharpe_ratio",
    "min_track_record_length",
]


# --------------------------------------------------------------------------- #
#  HAC regression
# --------------------------------------------------------------------------- #
@dataclass
class HACResult:
    params: np.ndarray
    names: list[str]
    se_ols: np.ndarray
    se_hac: np.ndarray
    tstat_hac: np.ndarray
    pvalue_hac: np.ndarray
    r2: float
    r2_adj: float
    nobs: int
    lags: int
    resid: np.ndarray = field(repr=False)
    fitted: np.ndarray = field(repr=False)
    ess: float = np.nan

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.params,
                "se_ols": self.se_ols,
                "se_hac": self.se_hac,
                "inflation": self.se_hac / np.where(self.se_ols > 0, self.se_ols, np.nan),
                "t_hac": self.tstat_hac,
                "p_hac": self.pvalue_hac,
            },
            index=self.names,
        )

    def __str__(self) -> str:
        head = (
            f"HAC (Newey-West, lags={self.lags}) OLS | n={self.nobs} "
            f"effective n~{self.ess:.0f} | R2={self.r2:.4f} adj={self.r2_adj:.4f}"
        )
        return head + "\n" + self.summary().to_string(float_format=lambda v: f"{v: .5f}")


def newey_west_lag(nobs: int, horizon: int | None = None) -> int:
    """Choose a truncation lag.

    With a known overlap horizon ``h`` (e.g. a 30-day forward window), the
    residual MA order is exactly ``h - 1`` and that is the lag to use --
    Hansen-Hodrick.  Without one, fall back to the standard
    ``floor(4 * (n/100)^(2/9))`` rule.
    """
    if horizon is not None and horizon > 1:
        return int(horizon - 1)
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    """Sample size adjusted for serial correlation,
    ``n_eff = n / (1 + 2*sum_k rho_k)``.

    Floored at 1 and capped at n.  This is the number to quote next to any
    t-statistic computed on overlapping data; quoting ``n = 500`` when the
    series contains 7 independent blocks is the core dishonesty in most retail
    backtests, and it is almost always unintentional.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float(n)
    if max_lag is None:
        max_lag = min(n - 2, int(10 * np.log10(n)))
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom <= 0:
        return float(n)
    total = 0.0
    for k in range(1, max_lag + 1):
        rho = np.dot(xc[:-k], xc[k:]) / denom
        if rho <= 0:          # initial-positive-sequence truncation (Geyer)
            break
        total += rho
    return float(np.clip(n / (1.0 + 2.0 * total), 1.0, n))


def ols_hac(y, X, names: Sequence[str] | None = None, add_const: bool = True,
            horizon: int | None = None, lags: int | None = None) -> HACResult:
    """OLS with Newey-West heteroskedasticity- and autocorrelation-consistent
    standard errors.

    Parameters
    ----------
    horizon:
        Overlap horizon of the dependent variable in observations.  Pass it
        whenever ``y`` is a forward-looking window; it sets ``lags = horizon-1``.
    lags:
        Explicit truncation lag, overrides ``horizon``.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if names is None:
        names = [f"x{i+1}" for i in range(X.shape[1])]
    names = list(names)
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
        names = ["const"] + names

    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[good], X[good]
    n, k = X.shape
    if n <= k:
        raise ValueError(f"not enough observations: n={n}, k={k}")

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - k

    s2 = resid @ resid / dof
    se_ols = np.sqrt(np.diag(s2 * XtX_inv))

    L = lags if lags is not None else newey_west_lag(n, horizon)
    L = int(np.clip(L, 0, n - 1))
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)                 # Bartlett kernel
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov_hac = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.clip(np.diag(cov_hac), 0.0, np.inf))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se_hac
    p = 2.0 * stats.t.sf(np.abs(t), df=dof)

    ss_res = resid @ resid
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / dof if np.isfinite(r2) else np.nan

    return HACResult(
        params=beta, names=names, se_ols=se_ols, se_hac=se_hac,
        tstat_hac=t, pvalue_hac=p, r2=float(r2), r2_adj=float(r2_adj),
        nobs=n, lags=L, resid=resid, fitted=X @ beta,
        ess=effective_sample_size(resid),
    )


# --------------------------------------------------------------------------- #
#  Bootstraps
# --------------------------------------------------------------------------- #
def circular_block_bootstrap(x: np.ndarray, block: int, n_boot: int = 2000,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    """Resample a series in wrapped blocks, preserving local dependence.

    Returns an ``(n_boot, n)`` array of resampled paths.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = x.size
    block = int(np.clip(block, 1, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return x[idx.reshape(n_boot, -1)[:, :n]]


def stationary_bootstrap(x: np.ndarray, mean_block: float, n_boot: int = 2000,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths, so the
    resampled series is itself stationary.  Preferred over fixed blocks when
    you cannot defend a particular block length."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = x.size
    p = 1.0 / max(mean_block, 1.0)
    out = np.empty((n_boot, n))
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(0, n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(0, n) if rng.random() < p else (i + 1) % n
        out[b] = x[idx]
    return out


def bootstrap_ci(x, statistic: Callable[[np.ndarray], float], block: int | None = None,
                 n_boot: int = 2000, alpha: float = 0.05, method: str = "circular",
                 rng: np.random.Generator | None = None) -> dict:
    """Block-bootstrap confidence interval and one-sided p-value for a statistic.

    The returned ``p_le_zero`` is the bootstrap probability that the statistic
    is <= 0, i.e. the natural one-sided p-value for "is this edge real".
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "p_le_zero": np.nan,
                "n": int(x.size), "block": np.nan, "n_boot": 0}
    if block is None:
        block = max(1, int(round(x.size / max(effective_sample_size(x), 1.0))))
    sampler = circular_block_bootstrap if method == "circular" else stationary_bootstrap
    paths = sampler(x, block, n_boot=n_boot, rng=rng)
    dist = np.array([statistic(p) for p in paths], dtype=float)
    dist = dist[np.isfinite(dist)]
    return {
        "point": float(statistic(x)),
        "lo": float(np.quantile(dist, alpha / 2)),
        "hi": float(np.quantile(dist, 1 - alpha / 2)),
        "p_le_zero": float(np.mean(dist <= 0.0)),
        "n": int(x.size),
        "block": int(block),
        "n_boot": int(dist.size),
    }


# --------------------------------------------------------------------------- #
#  Multiple-testing aware performance statistics
# --------------------------------------------------------------------------- #
def deflated_sharpe_ratio(returns, n_trials: int = 1, benchmark_sr: float = 0.0,
                          periods_per_year: int = 252) -> dict:
    """Bailey & Lopez de Prado deflated Sharpe ratio.

    An annualised Sharpe of 1.2 found after trying 200 parameter combinations is
    not the same evidence as a Sharpe of 1.2 from a single pre-registered test.
    This adjusts for the number of trials and for the skew and kurtosis of the
    return distribution -- both of which matter enormously for short-vol
    strategies, whose returns are left-skewed and fat-tailed, exactly the shape
    that inflates a naive Sharpe.

    Returns the observed SR, the expected maximum SR under the null of no skill
    across ``n_trials``, and the deflated probability that the true SR > 0.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 8 or r.std(ddof=1) == 0:
        return {"sr": np.nan, "sr_annual": np.nan, "sr0": np.nan, "dsr": np.nan, "n": n}

    sr = r.mean() / r.std(ddof=1)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))

    # Expected maximum of n_trials independent standard normals.
    e_max = 0.0
    if n_trials > 1:
        gamma = 0.5772156649015329
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        e_max = (1.0 - gamma) * z1 + gamma * z2
    sr0 = benchmark_sr + e_max / np.sqrt(n)

    denom = np.sqrt(1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2)
    if not np.isfinite(denom) or denom <= 0:
        dsr = np.nan
    else:
        dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))

    return {
        "sr": float(sr),
        "sr_annual": float(sr * np.sqrt(periods_per_year)),
        "sr0": float(sr0),
        "dsr": dsr,
        "skew": skew,
        "kurtosis": kurt,
        "n": int(n),
        "n_trials": int(n_trials),
    }


def min_track_record_length(returns, target_sr: float = 0.0, confidence: float = 0.95) -> float:
    """How many observations you would need before a Sharpe this size is
    distinguishable from ``target_sr`` at the given confidence.

    If this comes back larger than your sample, you do not have a result -- you
    have a hypothesis.  For event strategies the unit is *events*, and the
    number is usually humbling.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 8 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    if sr <= target_sr:
        return np.inf
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    z = stats.norm.ppf(confidence)
    return float(1.0 + (1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2) * (z / (sr - target_sr)) ** 2)
