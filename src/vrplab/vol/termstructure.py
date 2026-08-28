"""Extracting the *event* component of implied variance from a term structure.

Why this module exists
----------------------
Every retail treatment of the "IV crush" trade uses the **level** of ATM implied
vol as the signal: 93% looks high, so sell it.  That is not a signal, it is a
level with no reference distribution.  A high-beta software name into earnings
*should* have high ATM IV; the question is whether the part of that IV which is
attributable to the scheduled event is expensive relative to the move the event
actually produces.

The standard decomposition (Dubinsky-Johannes; used on every equity vol desk)
splits total implied variance into a diffusive part that accrues with calendar
time and a one-off event part that does not:

    total variance to expiry i:   V_i = sigma_i^2 * T_i
    model:                        V_i = sigma_d^2 * T_i + J^2      for every
                                        expiry i that brackets the event

with ``sigma_d`` the annualised diffusive (non-event) vol and ``J`` the
one-standard-deviation event move as a fraction of spot.  Two expiries that both
contain the event identify both unknowns exactly:

    sigma_d^2 = (V_2 - V_1) / (T_2 - T_1)
    J^2       = V_1 - sigma_d^2 * T_1

``J`` is the number to trade against.  ``J`` versus the realised event move is
the entire earnings-vol strategy; the IV crush itself is close to deterministic
and tells you almost nothing.

With three or more bracketing expiries the system is overdetermined and we fit
it by least squares, which also gives a residual you can use as a
specification check: a large residual means a single flat diffusive vol does
not describe the curve (term-structure slope from something other than the
event) and the extracted ``J`` should not be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["EventVariance", "extract_event_variance", "event_premium_ratio"]


@dataclass(frozen=True)
class EventVariance:
    """Result of the decomposition.

    Attributes
    ----------
    diffusive_vol:
        Annualised non-event volatility, decimal.
    event_move:
        One-standard-deviation event move as a fraction of spot (``J``).
        ``nan`` when the decomposition is infeasible.
    event_variance:
        ``J**2``.
    residual_rms:
        RMS of the fit residual in variance units.  Zero by construction when
        exactly two expiries are supplied.
    n_expiries:
        How many expiries entered the fit.
    feasible:
        False when the implied ``J**2`` or ``sigma_d**2`` came out negative,
        which happens with stale or crossed quotes and with an inverted curve
        that the model cannot represent.  Never silently clipped to zero.
    """

    diffusive_vol: float
    event_move: float
    event_variance: float
    residual_rms: float
    n_expiries: int
    feasible: bool

    def implied_move_pct(self) -> float:
        return 100.0 * self.event_move


def extract_event_variance(sigmas, tenors, clip_infeasible: bool = False) -> EventVariance:
    """Decompose a set of bracketing ATM implied vols into diffusive + event.

    Parameters
    ----------
    sigmas:
        Annualised ATM implied vols (decimals) for expiries that **all** bracket
        the same scheduled event.  Feeding an expiry that does not contain the
        event silently breaks the identification.
    tenors:
        Year fractions to those expiries, same order, strictly increasing.
    clip_infeasible:
        If True, a negative fitted ``J**2`` or ``sigma_d**2`` is floored at zero
        and ``feasible`` is still reported False.  Default False returns ``nan``
        so an infeasible fit cannot leak into a P&L series unnoticed.
    """
    sigmas = np.asarray(sigmas, dtype=float).ravel()
    tenors = np.asarray(tenors, dtype=float).ravel()
    if sigmas.shape != tenors.shape:
        raise ValueError("sigmas and tenors must have the same shape")
    ok = np.isfinite(sigmas) & np.isfinite(tenors) & (tenors > 0) & (sigmas > 0)
    sigmas, tenors = sigmas[ok], tenors[ok]
    n = sigmas.size
    if n < 2:
        return EventVariance(np.nan, np.nan, np.nan, np.nan, n, False)
    order = np.argsort(tenors)
    sigmas, tenors = sigmas[order], tenors[order]
    if not np.all(np.diff(tenors) > 0):
        raise ValueError("tenors must be distinct")

    total_var = sigmas**2 * tenors                       # V_i
    design = np.column_stack([tenors, np.ones(n)])       # [T_i, 1] @ [sigma_d^2, J^2]
    coef, *_ = np.linalg.lstsq(design, total_var, rcond=None)
    diff_var, event_var = float(coef[0]), float(coef[1])
    resid = total_var - design @ coef
    residual_rms = float(np.sqrt(np.mean(resid**2)))

    feasible = (diff_var >= 0.0) and (event_var >= 0.0)
    if not feasible:
        if clip_infeasible:
            diff_var = max(diff_var, 0.0)
            event_var = max(event_var, 0.0)
        else:
            return EventVariance(np.nan, np.nan, np.nan, residual_rms, n, False)

    return EventVariance(
        diffusive_vol=float(np.sqrt(diff_var)),
        event_move=float(np.sqrt(event_var)),
        event_variance=float(event_var),
        residual_rms=residual_rms,
        n_expiries=n,
        feasible=feasible,
    )


def event_premium_ratio(realised_move: float, implied_move: float) -> float:
    """``|realised event move| / implied event move``.

    This single number is the strategy.  A short straddle held through the event
    makes money broadly when the ratio is below ~1 and loses when it is above,
    and because the payoff is convex in the ratio the *mean* of the ratio is
    close to useless -- report the whole distribution, especially the right
    tail.  ``nan`` propagates rather than dividing by zero.
    """
    if implied_move is None or not np.isfinite(implied_move) or implied_move <= 0:
        return np.nan
    return abs(float(realised_move)) / float(implied_move)
