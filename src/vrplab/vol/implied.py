"""Black-Scholes-Merton pricing, Greeks, implied-vol inversion and implied move.

Everything here is vectorised over numpy arrays and works on scalars too.

Conventions used throughout vrplab
----------------------------------
* ``sigma`` is ALWAYS an annualised volatility expressed as a decimal
  (0.35 == 35% annualised).  There is exactly one place in the codebase where
  a raw vendor number is converted into this convention -- the data layer --
  and it is required to declare which convention it received.  This is the
  single most common source of silent error in retail vol code: multiplying an
  already-annualised number by sqrt(252).
* ``T`` is year fraction to expiry, ACT/365F.
* ``r`` is a continuously compounded risk-free rate, ``q`` a continuous
  dividend yield.  Both are decimals.
* Prices are per share.  The contract multiplier is applied exactly once, in
  the backtest layer, never here.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "d1_d2",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_vanna",
    "bs_volga",
    "straddle_price",
    "straddle_greeks",
    "implied_vol",
    "implied_move",
    "atm_straddle_to_implied_move",
    "atm_straddle_to_expected_abs_move",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)   # 0.797885: E|Z| for a standard normal
SQRT_PI_OVER_2 = np.sqrt(np.pi / 2.0)   # 1.253314: its reciprocal


def _as_arrays(*args):
    out = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
    return out


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Return ``(d1, d2)``.

    Degenerate inputs (``T <= 0`` or ``sigma <= 0``) return ``+/-inf`` with the
    correct sign so that ``bs_price`` collapses to the discounted intrinsic
    value rather than raising.  Silent NaNs are worse than a wrong-but-obvious
    number, and an exception in a GUI callback thread is worse than both.
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_t = sigma * np.sqrt(T)
        moneyness = np.log(np.where(S > 0, S, np.nan) / np.where(K > 0, K, np.nan))
        d1 = (moneyness + (r - q + 0.5 * sigma**2) * T) / vol_t
        d2 = d1 - vol_t

    degenerate = (T <= 0) | (sigma <= 0)
    if np.any(degenerate):
        fwd = S * np.exp((r - q) * T)
        sign = np.where(fwd > K, np.inf, np.where(fwd < K, -np.inf, 0.0))
        d1 = np.where(degenerate, sign, d1)
        d2 = np.where(degenerate, sign, d2)
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes-Merton price of a European option, per share."""
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    if kind == "call":
        px = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    elif kind == "put":
        px = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return np.maximum(px, 0.0)


def bs_delta(S, K, T, r, sigma, q=0.0, kind="call"):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    df_q = np.exp(-np.asarray(q, float) * np.asarray(T, float))
    return df_q * (norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return np.where(np.isfinite(g), g, 0.0)


def bs_vega(S, K, T, r, sigma, q=0.0, per_vol_point=False):
    """Vega.

    Returned per **1.00 change in sigma** by default (i.e. dPrice/dSigma), which
    is the only definition that composes correctly with the rest of the maths.
    Pass ``per_vol_point=True`` for the trading-desk convention of price change
    per 1 volatility *point* (0.01), which is what a broker screen shows.
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    v = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    v = np.where(np.isfinite(v), v, 0.0)
    return v / 100.0 if per_vol_point else v


def bs_theta(S, K, T, r, sigma, q=0.0, kind="call", per_day=True):
    """Theta. Per calendar day by default (ACT/365F), matching ``T``."""
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        carry = -(S * df_q * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    carry = np.where(np.isfinite(carry), carry, 0.0)
    if kind == "call":
        th = carry - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
    elif kind == "put":
        th = carry + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return th / 365.0 if per_day else th


def bs_vanna(S, K, T, r, sigma, q=0.0):
    """d(Vega)/d(S) == d(Delta)/d(sigma). Matters for a short straddle: it is
    the term that makes the delta of your position move when vol moves."""
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma
    return np.where(np.isfinite(v), v, 0.0)


def bs_volga(S, K, T, r, sigma, q=0.0):
    """d(Vega)/d(sigma). The convexity of the position in vol -- the reason a
    linear 'vega x delta-IV' P&L estimate understates a large IV crush."""
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = bs_vega(S, K, T, r, sigma, q) * d1 * d2 / sigma
    return np.where(np.isfinite(v), v, 0.0)


def straddle_price(S, K, T, r, sigma, q=0.0):
    return bs_price(S, K, T, r, sigma, q, "call") + bs_price(S, K, T, r, sigma, q, "put")


def straddle_greeks(S, K, T, r, sigma, q=0.0):
    """Greeks of a **long** straddle, per share. Negate for a short."""
    return {
        "price": straddle_price(S, K, T, r, sigma, q),
        "delta": bs_delta(S, K, T, r, sigma, q, "call") + bs_delta(S, K, T, r, sigma, q, "put"),
        "gamma": 2.0 * bs_gamma(S, K, T, r, sigma, q),
        "vega": 2.0 * bs_vega(S, K, T, r, sigma, q),
        "theta": bs_theta(S, K, T, r, sigma, q, "call") + bs_theta(S, K, T, r, sigma, q, "put"),
        "vanna": bs_vanna(S, K, T, r, sigma, q) * 2.0,
        "volga": 2.0 * bs_volga(S, K, T, r, sigma, q),
    }


def implied_vol(price, S, K, T, r, q=0.0, kind="call", lo=1e-4, hi=8.0):
    """Invert Black-Scholes for sigma by Brent's method on a bracketed root.

    Returns ``np.nan`` when the quoted price violates the no-arbitrage bounds,
    rather than returning a number that happens to be ``lo`` or ``hi``.  A
    silent clamp at the boundary is how a 'vol' of 800% ends up in a backtest.
    """
    price, S, K, T, r, q = (float(price), float(S), float(K), float(T), float(r), float(q))
    if T <= 0 or S <= 0 or K <= 0:
        return np.nan

    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if kind == "call":
        lower, upper = max(S * df_q - K * df_r, 0.0), S * df_q
    else:
        lower, upper = max(K * df_r - S * df_q, 0.0), K * df_r
    # Strictly inside the bounds, with a tick of tolerance for rounding.
    if not (lower - 1e-10 <= price <= upper + 1e-10):
        return np.nan
    if price <= lower + 1e-10:
        return np.nan

    def f(sig):
        return float(bs_price(S, K, T, r, sig, q, kind)) - price

    try:
        if f(lo) > 0 or f(hi) < 0:
            return np.nan
        return float(brentq(f, lo, hi, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return np.nan


def implied_move(sigma, T):
    """One-standard-deviation move implied by an annualised vol over horizon T,
    as a fraction of spot: ``sigma * sqrt(T)``."""
    return np.asarray(sigma, float) * np.sqrt(np.asarray(T, float))


def atm_straddle_to_expected_abs_move(straddle_px, S):
    """Expected **absolute** move implied by an ATM straddle, as a fraction of
    spot.

    Derivation, driftless, r = q = 0, x = sigma*sqrt(T):

        ATM call = ATM put = S[N(x/2) - N(-x/2)],  so
        straddle = 2S[2N(x/2) - 1]
                 ~= 2S * x * phi(0)            (first order in x)
                  = S * x * sqrt(2/pi)

    and for a normal with standard deviation x, ``E|move| = x * sqrt(2/pi)``.
    The two right-hand sides are identical, so

        E|move| = straddle / S

    exactly at first order.  The desk shorthand is therefore not a fudge -- it
    is the correct leading-order expression, and it is the number to compare
    against a mean of realised absolute event moves.
    """
    return np.asarray(straddle_px, float) / np.asarray(S, float)


def atm_straddle_to_implied_move(straddle_px, S, T=None):
    """One-**standard-deviation** move implied by an ATM straddle, as a
    fraction of spot.

    From the derivation in :func:`atm_straddle_to_expected_abs_move`,
    ``straddle/S = x * sqrt(2/pi)`` with ``x = sigma*sqrt(T)``, so inverting:

        sigma*sqrt(T) = sqrt(pi/2) * straddle / S  ~=  1.2533 * straddle / S

    Note this is *larger* than ``straddle/S``, because a one-sigma move is
    larger than the mean absolute move of a normal (by 1/sqrt(2/pi)).  Use
    :func:`atm_straddle_to_expected_abs_move` when comparing against realised
    absolute moves and this function when comparing against a standard
    deviation; conflating the two is a 25% error in either direction.

    ``T`` is accepted for signature symmetry and ignored.
    """
    return SQRT_PI_OVER_2 * np.asarray(straddle_px, float) / np.asarray(S, float)
