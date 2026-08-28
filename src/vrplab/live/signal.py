"""Live screening against Interactive Brokers.

This is the only module that touches a live market, and it deliberately stops
short of sending an order.  What it produces is a **decision packet**: the
implied event move the market is charging right now, the same name's own
historical distribution of realised event moves, where today's price sits in
that distribution, the size that keeps a tail loss survivable, and an explicit
list of reasons to stand down.

Two design choices worth defending in an interview:

* **The screen is a percentile against the name's own history, not a level.**
  "93% IV looks inflated" is not a signal; 93% on a name whose last twelve
  earnings moves averaged 11% is cheap.  The comparison has to be
  implied-event-move versus that name's realised-event-move distribution.
* **Size is set by the tail, not by the premium.**  A short straddle is short a
  convex function of the move.  Sizing off expected profit is how a
  positive-expectancy strategy still ruins you; sizing off a survivable
  worst-case is how it does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..vol.implied import SQRT_2_OVER_PI, straddle_price
from ..vol.termstructure import extract_event_variance

__all__ = ["RiskLimits", "Signal", "screen_event", "size_position"]


@dataclass(frozen=True)
class RiskLimits:
    """Hard limits, checked before anything is called a trade."""

    account_equity: float = 25_000.0
    max_loss_per_event: float = 0.02       # fraction of equity in a stress move
    stress_move_sigmas: float = 4.0        # the move you must survive
    min_events_in_history: int = 8         # below this, no percentile is meaningful
    max_spread_pct_of_mid: float = 0.08    # refuse illiquid chains
    min_implied_move: float = 0.02         # nothing to sell below this
    max_premium_fraction: float = 0.05     # cap notional premium per event


@dataclass
class Signal:
    symbol: str
    asof: pd.Timestamp
    spot: float
    implied_event_move: float
    diffusive_vol: float
    front_tenor: float
    history_n: int
    history_mean_abs_move: float
    implied_vs_history: float              # implied / mean realised
    percentile_of_history: float           # where implied sits in realised dist
    straddle_mid: float
    max_contracts: int
    stress_loss: float
    action: str
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"{self.symbol}  {self.asof:%Y-%m-%d}  spot {self.spot:.2f}",
            f"  implied event move   {self.implied_event_move:8.2%}",
            f"  diffusive vol        {self.diffusive_vol:8.2%}",
            f"  history ({self.history_n} events) mean |move| "
            f"{self.history_mean_abs_move:.2%}",
            f"  implied / realised   {self.implied_vs_history:8.2f}x",
            f"  implied sits at      {self.percentile_of_history:8.1%} of realised history",
            f"  ATM straddle mid     {self.straddle_mid:8.2f}",
            f"  max size             {self.max_contracts} contracts "
            f"(stress loss {self.stress_loss:,.0f})",
            f"  ACTION: {self.action}",
        ]
        lines += [f"    - {r}" for r in self.reasons]
        return "\n".join(lines)


def size_position(spot: float, strike: float, tenor: float, iv: float,
                  implied_move: float, limits: RiskLimits, r: float = 0.04,
                  post_iv: float | None = None) -> tuple[int, float]:
    """Contracts such that a ``stress_move_sigmas`` move stays inside the loss cap.

    The stress move is measured in units of the **implied event move**, not of
    the diffusive vol, because the event jump is the risk.  At 4x the implied
    move the position is deep in the money on one leg and the loss is close to
    linear in the overshoot, which is exactly the regime where a
    premium-based sizing rule fails.
    """
    if implied_move <= 0 or spot <= 0:
        return 0, 0.0
    credit = float(straddle_price(spot, strike, tenor, r, iv))
    stressed_spot = spot * np.exp(limits.stress_move_sigmas * implied_move)
    # After the event the event variance is gone, so the position is repriced on
    # the DIFFUSIVE vol. Pass it in from the term-structure decomposition; the
    # 0.6*iv fallback below is only for callers that have no term structure, and
    # it is a guess, not an estimate.
    if post_iv is None or not np.isfinite(post_iv) or post_iv <= 0:
        post_iv = max(iv * 0.6, 0.05)
    stressed_value = float(straddle_price(stressed_spot, strike,
                                          max(tenor - 1 / 365, 1 / 365), r, post_iv))
    loss_per_share = max(stressed_value - credit, 0.0)
    if loss_per_share <= 0:
        return 0, 0.0
    budget = limits.account_equity * limits.max_loss_per_event
    by_stress = int(np.floor(budget / (loss_per_share * 100.0)))
    by_premium = int(np.floor(limits.account_equity * limits.max_premium_fraction
                              / max(credit * 100.0, 1e-9)))
    n = max(min(by_stress, by_premium), 0)
    return n, loss_per_share * 100.0 * n


def screen_event(
    *,
    symbol: str,
    asof: pd.Timestamp,
    spot: float,
    front_iv: float,
    back_iv: float,
    front_tenor: float,
    back_tenor: float,
    historical_moves: np.ndarray,
    limits: RiskLimits | None = None,
    spread_pct_of_mid: float = np.nan,
    r: float = 0.04,
    strike_increment: float = 2.5,
) -> Signal:
    """Turn a live term structure into a decision packet.

    ``historical_moves`` is that symbol's own past absolute event moves as
    fractions of spot -- produced by
    :func:`vrplab.research.eventstudy.build_event_panel` on its earnings
    history.  Screening without it is guessing, so the function will say so
    rather than emitting a number.
    """
    limits = limits or RiskLimits()
    reasons: list[str] = []

    term = extract_event_variance([front_iv, back_iv], [front_tenor, back_tenor])
    hist = np.asarray(historical_moves, dtype=float)
    hist = np.abs(hist[np.isfinite(hist)])
    n_hist = hist.size
    mean_abs = float(hist.mean()) if n_hist else np.nan

    strike = float(np.round(spot / strike_increment) * strike_increment)
    mid = float(straddle_price(spot, strike, front_tenor, r, front_iv))

    if not term.feasible:
        reasons.append("term structure does not admit a non-negative event "
                       "variance; quotes are stale, crossed, or the curve slope "
                       "is not coming from the event")
        return Signal(symbol, pd.Timestamp(asof), spot, np.nan, np.nan, front_tenor,
                      n_hist, mean_abs, np.nan, np.nan, mid, 0, 0.0,
                      "STAND DOWN", reasons)

    implied = term.event_move
    # `implied` is a one-standard-deviation move; `hist` is a mean of absolute
    # moves. For a normal, E|X| = sigma * sqrt(2/pi), so convert before
    # comparing. Mixing the two is a 25% error in either direction -- see the
    # note in vrplab.vol.implied.atm_straddle_to_implied_move.
    expected_abs = SQRT_2_OVER_PI * implied
    ratio = expected_abs / mean_abs if n_hist and mean_abs > 0 else np.nan
    pct = float(np.mean(hist <= expected_abs)) if n_hist else np.nan

    n_contracts, stress_loss = size_position(spot, strike, front_tenor, front_iv,
                                             implied, limits, r=r,
                                             post_iv=term.diffusive_vol)

    if n_hist < limits.min_events_in_history:
        reasons.append(f"only {n_hist} historical events; a percentile on this "
                       f"sample is noise (need >= {limits.min_events_in_history})")
    if implied < limits.min_implied_move:
        reasons.append(f"implied event move {implied:.2%} is below the "
                       f"{limits.min_implied_move:.2%} floor; nothing to sell")
    if np.isfinite(spread_pct_of_mid) and spread_pct_of_mid > limits.max_spread_pct_of_mid:
        reasons.append(f"quoted spread {spread_pct_of_mid:.1%} of mid exceeds the "
                       f"{limits.max_spread_pct_of_mid:.1%} limit; the edge is "
                       f"inside the spread")
    if term.residual_rms > 5e-4:
        reasons.append(f"term-structure fit residual {term.residual_rms:.2e} is "
                       f"large; a single diffusive vol does not describe this curve")
    if n_contracts < 1:
        reasons.append("risk limits permit zero contracts at this size of tail")

    if reasons:
        action = "STAND DOWN"
    elif np.isfinite(ratio) and ratio > 1.25:
        action = f"SELL VOL: implied is {ratio:.2f}x this name's own realised mean"
    elif np.isfinite(ratio) and ratio < 0.85:
        action = f"BUY VOL: implied is only {ratio:.2f}x realised mean"
    else:
        action = "NO EDGE: implied is within noise of this name's realised history"

    return Signal(symbol, pd.Timestamp(asof), spot, implied, term.diffusive_vol,
                  front_tenor, n_hist, mean_abs, ratio, pct, mid,
                  n_contracts, stress_loss, action, reasons)


def screen_from_ib(provider, symbol: str, earnings_date, historical_moves,
                   limits: RiskLimits | None = None, r: float = 0.04) -> Signal:
    """Pull the two bracketing expiries from a live IB connection and screen.

    Requires a connected :class:`vrplab.data.ib.IBProvider`.  Uses
    ``reqSecDefOptParams`` to find the expiries either side of the announcement
    and ``reqMktData`` with generic tick 106 for a per-contract implied vol,
    rather than IB's aggregate underlying IV series of unstated tenor.
    """
    earnings_date = pd.Timestamp(earnings_date)
    params = provider.option_params(symbol)
    if params.empty:
        raise RuntimeError(f"no option parameters returned for {symbol}")
    smart = params[params["exchange"] == "SMART"]
    row = (smart if not smart.empty else params).iloc[0]

    expiries = pd.to_datetime(pd.Series(row["expirations"]), format="%Y%m%d")
    after = expiries[expiries > earnings_date].sort_values()
    if len(after) < 2:
        raise RuntimeError(f"need two expiries after {earnings_date:%Y-%m-%d}, "
                           f"found {len(after)}")
    front, back = after.iloc[0], after.iloc[1]

    strikes = np.asarray(row["strikes"], dtype=float)
    # A snapshot on the front ATM call gives spot and a per-contract IV.
    probe = provider.option_snapshot(symbol, front.strftime("%Y%m%d"),
                                     float(strikes[len(strikes) // 2]), "C")
    spot = float(probe.get("und_price", np.nan))
    if not np.isfinite(spot):
        raise RuntimeError("no underlying price in the option snapshot; check "
                           "market data subscriptions")
    atm = float(strikes[np.argmin(np.abs(strikes - spot))])

    f = provider.option_snapshot(symbol, front.strftime("%Y%m%d"), atm, "C")
    b = provider.option_snapshot(symbol, back.strftime("%Y%m%d"), atm, "C")
    asof = pd.Timestamp.utcnow().tz_localize(None)
    spread_pct = np.nan
    if np.isfinite(f.get("bid", np.nan)) and np.isfinite(f.get("ask", np.nan)):
        m = 0.5 * (f["bid"] + f["ask"])
        spread_pct = (f["ask"] - f["bid"]) / m if m > 0 else np.nan

    return screen_event(
        symbol=symbol, asof=asof, spot=spot,
        front_iv=float(f["iv"]), back_iv=float(b["iv"]),
        front_tenor=max((front - asof).days, 1) / 365.0,
        back_tenor=max((back - asof).days, 2) / 365.0,
        historical_moves=historical_moves, limits=limits,
        spread_pct_of_mid=spread_pct, r=r,
    )
