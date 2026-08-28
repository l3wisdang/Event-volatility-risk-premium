"""A market simulator with known ground truth.

Why bother
----------
Every estimator in this package makes a claim: the term-structure decomposition
claims to recover the event move; the HMM claims to recover regimes; the event
study claims to measure a variance risk premium net of costs.  On real data you
cannot check any of those, because you never observe the truth.  On simulated
data you can *plant* the truth and then demand that the estimator recover it.

This is the same instinct as a look-ahead harness, pushed one step further: a
harness proves your pipeline does not cheat, a ground-truth simulator proves
your estimator actually measures the thing it is named after.  If the event
study cannot find a variance risk premium that was deliberately planted at
25%, it will certainly not find a real one of 5%.

The generator plants, simultaneously:

* a two-state (or k-state) hidden volatility regime with a known transition
  matrix -- ground truth for :mod:`vrplab.research.regime`;
* scheduled quarterly earnings events, each with a known implied event move and
  a realised move drawn so that implied variance exceeds true variance by a
  chosen premium -- ground truth for :mod:`vrplab.research.eventstudy`;
* a fat left tail, so short-vol P&L has the correct shape and a naive Sharpe
  ratio is correctly flattering;
* an option term structure at each event that is internally consistent with the
  planted event variance -- ground truth for
  :mod:`vrplab.vol.termstructure`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import BarRequest, VolUnits, validate_bars

__all__ = ["SyntheticConfig", "SyntheticProvider"]

_TRADING_DAYS = 252


@dataclass
class SyntheticConfig:
    """Ground truth. Every field here is something an estimator must recover."""

    symbol: str = "SYNTH"
    start: str = "2015-01-02"
    n_days: int = 2600
    s0: float = 100.0

    # --- hidden volatility regimes (annualised diffusive vol per state) ---
    regime_vols: tuple[float, ...] = (0.14, 0.32)
    transition: tuple[tuple[float, ...], ...] = ((0.985, 0.015), (0.045, 0.955))

    # --- scheduled events ---
    days_between_events: int = 63          # ~quarterly
    implied_event_move: float = 0.070      # one-sd move the options price in
    implied_event_move_sd: float = 0.020   # cross-event dispersion of that
    variance_risk_premium: float = 0.25    # implied event var = (1+vrp) * true
    tail_prob: float = 0.06                # probability an event is a fat draw
    tail_multiple: float = 3.2             # how much bigger the fat draw is

    # --- option market micro-structure ---
    front_tenor_days: int = 7
    back_tenor_days: int = 35
    iv_quote_noise: float = 0.004          # additive noise on quoted ATM vols
    diffusive_vrp: float = 0.15            # options overprice diffusive vol too

    drift: float = 0.06
    seed: int = 7

    # filled in by the generator so callers can assert against it
    truth: dict = field(default_factory=dict)


class SyntheticProvider:
    """A :class:`~vrplab.data.base.MarketDataProvider` with known answers."""

    name = "synthetic"
    vol_units = VolUnits.ANNUAL_DECIMAL

    def __init__(self, config: SyntheticConfig | None = None):
        self.config = config or SyntheticConfig()
        self._generate()

    # ------------------------------------------------------------------ #
    def _generate(self) -> None:
        c = self.config
        rng = np.random.default_rng(c.seed)
        n = c.n_days
        P = np.asarray(c.transition, dtype=float)
        vols = np.asarray(c.regime_vols, dtype=float)
        k = len(vols)

        # ---- hidden regime path ----
        states = np.zeros(n, dtype=int)
        states[0] = rng.integers(0, k)
        for t in range(1, n):
            states[t] = rng.choice(k, p=P[states[t - 1]])
        sigma_d = vols[states]                       # annualised diffusive vol

        dt = 1.0 / _TRADING_DAYS
        dates = pd.bdate_range(start=c.start, periods=n)

        # ---- event schedule ----
        first = c.days_between_events // 2
        event_idx = np.arange(first, n - c.back_tenor_days - 2, c.days_between_events)
        is_event = np.zeros(n, dtype=bool)
        is_event[event_idx] = True                   # event realises overnight INTO this bar

        # ---- planted event moves ----
        m = event_idx.size
        implied_moves = np.clip(
            rng.normal(c.implied_event_move, c.implied_event_move_sd, m), 0.015, 0.40)
        # `variance_risk_premium` is the NET planted premium, inclusive of the
        # fat tail.  A mixture with probability p of a draw `mult` times larger
        # has second moment (1 - p + p*mult^2) times the base variance, so the
        # base is scaled down by that factor.  Without this correction the tail
        # silently eats the premium and the config field means something other
        # than its name -- which is precisely the class of error this whole
        # package exists to catch.
        tail_moment = 1.0 - c.tail_prob + c.tail_prob * c.tail_multiple**2
        true_sd = implied_moves / np.sqrt((1.0 + c.variance_risk_premium) * tail_moment)
        fat = rng.random(m) < c.tail_prob
        realised_jump = rng.normal(0.0, true_sd)
        realised_jump[fat] *= c.tail_multiple

        # ---- price path: diffusion + overnight event jumps ----
        z = rng.normal(size=n)
        log_ret = (c.drift - 0.5 * sigma_d**2) * dt + sigma_d * np.sqrt(dt) * z
        jump_series = np.zeros(n)
        jump_series[event_idx] = realised_jump
        log_ret = log_ret + jump_series

        close = c.s0 * np.exp(np.cumsum(log_ret))
        prev_close = np.concatenate([[c.s0], close[:-1]])

        # Intraday OHLC consistent with the same diffusive vol. The event jump
        # is an overnight gap, so it lands in the open, not the intraday range.
        daily_sd = sigma_d * np.sqrt(dt)
        open_ = prev_close * np.exp(jump_series + rng.normal(0.0, 1.0, n) * daily_sd * 0.25)
        intraday = np.abs(rng.normal(0.0, 1.0, size=(n, 2)) * daily_sd[:, None])
        high = np.maximum(open_, close) * np.exp(intraday[:, 0])
        low = np.minimum(open_, close) * np.exp(-intraday[:, 1])
        volume = rng.integers(1_000_000, 9_000_000, n).astype(float)
        volume[event_idx] *= 3.0

        self._bars = validate_bars(
            pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=dates,
            ),
            BarRequest(c.symbol, dates[0], dates[-1]),
        )

        # ---- daily 30-day ATM implied vol series (what IB would hand you) ----
        # Options overprice diffusive vol by `diffusive_vrp`, and the 30-day
        # quote absorbs the event variance of any event inside the window.
        horizon = 30
        iv30 = np.zeros(n)
        for t in range(n):
            hi = min(n, t + horizon)
            fwd_diff_var = np.mean(sigma_d[t:hi] ** 2) if hi > t else sigma_d[t] ** 2
            fwd_diff_var *= (1.0 + c.diffusive_vrp)
            ev = implied_moves[(event_idx >= t) & (event_idx < hi)]
            total_var = fwd_diff_var * (horizon / _TRADING_DAYS) + float(np.sum(ev**2))
            iv30[t] = np.sqrt(total_var / (horizon / _TRADING_DAYS))
        iv30 = np.clip(iv30 + rng.normal(0, c.iv_quote_noise, n), 0.02, None)
        self._iv30 = pd.Series(iv30, index=dates, name="iv30")

        # ---- option term structure quoted the day BEFORE each event ----
        t_front = c.front_tenor_days / 365.0
        t_back = c.back_tenor_days / 365.0
        quote_idx = event_idx - 1
        sd_at_quote = sigma_d[quote_idx] * np.sqrt(1.0 + c.diffusive_vrp)
        front_iv = np.sqrt((sd_at_quote**2 * t_front + implied_moves**2) / t_front)
        back_iv = np.sqrt((sd_at_quote**2 * t_back + implied_moves**2) / t_back)
        front_iv = front_iv + rng.normal(0, c.iv_quote_noise, m)
        back_iv = back_iv + rng.normal(0, c.iv_quote_noise, m)

        # The announcement is made after the close of day `quote_idx`, so the
        # event date IS the quote date and the move realises in the following
        # session -- an AMC reporter, the most common case.  Labelling the event
        # by the day the move appears instead would put the jump inside the
        # entry price, which is exactly the off-by-one-session error that
        # inverts the trade for half the earnings universe.
        self._events = pd.DataFrame(
            {
                "symbol": c.symbol,
                "event_date": dates[quote_idx],
                "quote_date": dates[quote_idx],
                "move_date": dates[event_idx],
                "session": "AMC",
                "spot_at_quote": close[quote_idx],
                "front_expiry": dates[quote_idx] + pd.Timedelta(days=c.front_tenor_days),
                "back_expiry": dates[quote_idx] + pd.Timedelta(days=c.back_tenor_days),
                "front_tenor": t_front,
                "back_tenor": t_back,
                "front_iv": front_iv,
                "back_iv": back_iv,
                # ---- ground truth, for assertions only ----
                "true_implied_move": implied_moves,
                "true_diffusive_vol": sd_at_quote,
                "true_realised_jump": realised_jump,
                "true_is_tail": fat,
            }
        ).set_index("event_date")

        c.truth = {
            "regime_states": pd.Series(states, index=dates, name="state"),
            "regime_vols": vols,
            "transition": P,
            "diffusive_vol": pd.Series(sigma_d, index=dates, name="sigma_d"),
            "variance_risk_premium": c.variance_risk_premium,
            "n_events": int(m),
            "mean_implied_move": float(implied_moves.mean()),
            "mean_abs_realised": float(np.abs(realised_jump).mean()),
        }

    # ------------------------------------------------------------------ #
    #  MarketDataProvider interface
    # ------------------------------------------------------------------ #
    def bars(self, request: BarRequest) -> pd.DataFrame:
        if request.what == "OPTION_IMPLIED_VOLATILITY":
            s = self._iv30.loc[request.start:request.end]
            return pd.DataFrame(
                {"open": s, "high": s, "low": s, "close": s, "volume": 0.0}, index=s.index)
        return self._bars.loc[request.start:request.end].copy()

    def option_chain(self, symbol: str, asof: pd.Timestamp) -> pd.DataFrame:
        """The two bracketing expiries quoted at ``asof``, if it is a quote date."""
        asof = pd.Timestamp(asof)
        row = self._events[self._events["quote_date"] == asof]
        if row.empty:
            return pd.DataFrame()
        r = row.iloc[0]
        return pd.DataFrame(
            [
                {"expiry": r["front_expiry"], "tenor": r["front_tenor"],
                 "atm_iv": r["front_iv"], "underlying": r["spot_at_quote"]},
                {"expiry": r["back_expiry"], "tenor": r["back_tenor"],
                 "atm_iv": r["back_iv"], "underlying": r["spot_at_quote"]},
            ]
        )

    # ------------------------------------------------------------------ #
    def full_bars(self) -> pd.DataFrame:
        """The whole generated price history, without going through BarRequest."""
        return self._bars.copy()

    def events(self) -> pd.DataFrame:
        """Scheduled events with their quoted term structure."""
        return self._events.copy()

    def iv30(self) -> pd.Series:
        return self._iv30.copy()

    @property
    def truth(self) -> dict:
        return self.config.truth
