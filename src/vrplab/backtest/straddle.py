"""Event-driven short-straddle P&L, with the costs that decide the answer.

The three numbers that turn a profitable earnings-vol study into an
unprofitable one, in order of magnitude:

  1. **the bid-ask spread**, paid twice, on an ATM straddle into a print, where
     it is at its widest;
  2. **the contract multiplier**, which the source builds omit entirely (their
     "$267 profit" is a per-share number);
  3. **theta**, which they also omit by holding ``T`` fixed across the event.

None of these is a rounding error.  On a 7-day ATM straddle at 60% IV the
round-trip spread alone is routinely 3-6% of the premium, against a planted
variance premium of maybe 20-30% of *variance*, which is a far smaller number
in premium terms than it sounds.

The convention here: a short straddle is opened at the bid and closed at the
ask, priced from a mid computed by Black-Scholes at the quoted implied vol.
Every cost is explicit and can be set to zero to see exactly how much of the
result it was carrying.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..vol.implied import straddle_greeks, straddle_price

__all__ = ["CostModel", "StraddleTrade", "price_event_straddle", "run_event_backtest"]


@dataclass(frozen=True)
class CostModel:
    """Explicit, auditable transaction costs.

    Attributes
    ----------
    half_spread_pct:
        Half the quoted bid-ask as a fraction of the option mid, per leg.
        0.02 means a 4%-of-mid round trip on each leg.
    commission_per_contract:
        Broker commission per option contract per side, in currency.
    slippage_pct:
        Additional adverse fill beyond the quoted touch, as a fraction of mid.
        Set this above zero for anything you intend to trade in size.
    multiplier:
        Contract multiplier. 100 for US equity options.
    """

    half_spread_pct: float = 0.02
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.0
    multiplier: int = 100

    def entry_fill(self, mid: float) -> float:
        """Price received for selling one straddle (two legs), per share."""
        return float(mid) * (1.0 - self.half_spread_pct - self.slippage_pct)

    def exit_fill(self, mid: float) -> float:
        """Price paid to buy one straddle back, per share."""
        return float(mid) * (1.0 + self.half_spread_pct + self.slippage_pct)

    def commissions(self, n_contracts: int = 1) -> float:
        """Round trip, two legs, both sides."""
        return 4.0 * self.commission_per_contract * n_contracts


@dataclass
class StraddleTrade:
    """One short-straddle event trade, fully decomposed."""

    symbol: str
    event_date: pd.Timestamp
    entry_date: pd.Timestamp
    """The session whose close the position is opened at. For an AMC reporter
    this is the event date; for a BMO reporter it is the previous session. Any
    conditioning variable must be evaluated here, not at `event_date`."""
    strike: float
    spot_entry: float
    spot_exit: float
    iv_entry: float
    iv_exit: float
    tenor_entry: float
    tenor_exit: float
    mid_entry: float
    mid_exit: float
    credit: float          # received, per share, net of half-spread
    debit: float           # paid, per share, net of half-spread
    gross_pnl: float       # per share, mid-to-mid
    net_pnl: float         # per contract-set, after spread, slippage, commission
    net_pnl_per_share: float
    realised_move: float
    implied_move: float
    move_vs_implied_sd: float
    """|realised| / (sigma_front * sqrt(T)). NOTE this is a different quantity
    from the event panel's `move_ratio`, which divides by the extracted event
    move J. This one uses the whole front-expiry vol including its diffusive
    part, so it is always the smaller of the two. Named distinctly on purpose:
    the two frames are routinely joined."""
    return_on_credit: float
    vega_entry: float
    gamma_entry: float
    theta_entry: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _round_strike(spot: float, increment: float = 2.5) -> float:
    """Snap to a listed strike. Pricing a straddle at a strike that does not
    exist is how a study produces a P&L nobody could have earned."""
    if increment <= 0:
        return float(spot)
    return float(np.round(spot / increment) * increment)


def price_event_straddle(
    *,
    symbol: str,
    event_date: pd.Timestamp,
    spot_entry: float,
    spot_exit: float,
    iv_entry: float,
    iv_exit: float,
    tenor_entry: float,
    hold_days: float = 1.0,
    entry_date: pd.Timestamp | None = None,
    r: float = 0.04,
    q: float = 0.0,
    costs: CostModel | None = None,
    n_contracts: int = 1,
    strike_increment: float = 2.5,
) -> StraddleTrade:
    """Price one short ATM straddle through an event.

    ``tenor_exit = tenor_entry - hold_days/365`` so theta is charged, unlike the
    source builds which reprice at the same ``T``.  Sign convention: positive
    ``net_pnl`` is a profit for the **seller**.
    """
    costs = costs or CostModel()
    K = _round_strike(spot_entry, strike_increment)
    tenor_exit = max(tenor_entry - hold_days / 365.0, 1.0 / 365.0)

    mid_entry = float(straddle_price(spot_entry, K, tenor_entry, r, iv_entry, q))
    mid_exit = float(straddle_price(spot_exit, K, tenor_exit, r, iv_exit, q))

    credit = costs.entry_fill(mid_entry)
    debit = costs.exit_fill(mid_exit)

    gross = mid_entry - mid_exit
    net_per_share = credit - debit
    net = net_per_share * costs.multiplier * n_contracts - costs.commissions(n_contracts)

    g = straddle_greeks(spot_entry, K, tenor_entry, r, iv_entry, q)
    realised = float(np.log(spot_exit / spot_entry))
    implied = float(iv_entry * np.sqrt(tenor_entry))

    return StraddleTrade(
        symbol=symbol,
        event_date=pd.Timestamp(event_date),
        entry_date=pd.Timestamp(entry_date if entry_date is not None else event_date),
        strike=K,
        spot_entry=float(spot_entry),
        spot_exit=float(spot_exit),
        iv_entry=float(iv_entry),
        iv_exit=float(iv_exit),
        tenor_entry=float(tenor_entry),
        tenor_exit=float(tenor_exit),
        mid_entry=mid_entry,
        mid_exit=mid_exit,
        credit=credit,
        debit=debit,
        gross_pnl=float(gross),
        net_pnl=float(net),
        net_pnl_per_share=float(net_per_share),
        realised_move=realised,
        implied_move=implied,
        move_vs_implied_sd=float(abs(realised) / implied) if implied > 0 else np.nan,
        return_on_credit=float(net_per_share / credit) if credit > 0 else np.nan,
        vega_entry=float(g["vega"]),
        gamma_entry=float(g["gamma"]),
        theta_entry=float(g["theta"]),
    )


def run_event_backtest(panel: pd.DataFrame, costs: CostModel | None = None,
                       r: float = 0.04, q: float = 0.0, hold_days: float = 1.0,
                       n_contracts: int = 1, strike_increment: float = 2.5) -> pd.DataFrame:
    """Run the short-straddle trade over an event panel.

    ``panel`` must carry: ``symbol``, ``spot_entry``, ``spot_exit``,
    ``iv_entry``, ``iv_exit``, ``tenor_entry``, indexed by event date.  Rows with
    any missing input are dropped and counted, never imputed -- the source
    builds substitute ``pre_iv = VIX/100*1.5`` when data is missing, which
    fabricates the headline number.
    """
    required = {"symbol", "spot_entry", "spot_exit", "iv_entry", "iv_exit", "tenor_entry"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    clean = panel.dropna(subset=sorted(required - {"symbol"}))
    dropped = len(panel) - len(clean)

    trades = [
        price_event_straddle(
            symbol=row["symbol"], event_date=idx,
            spot_entry=row["spot_entry"], spot_exit=row["spot_exit"],
            iv_entry=row["iv_entry"], iv_exit=row["iv_exit"],
            tenor_entry=row["tenor_entry"], hold_days=hold_days,
            entry_date=row.get("entry_date"),
            r=r, q=q, costs=costs, n_contracts=n_contracts,
            strike_increment=strike_increment,
        ).as_dict()
        for idx, row in clean.iterrows()
    ]
    out = pd.DataFrame(trades)
    if not out.empty:
        out = out.set_index("event_date").sort_index()
    out.attrs["dropped_rows"] = int(dropped)
    out.attrs["costs"] = costs or CostModel()
    return out
