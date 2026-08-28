"""The one command that answers the question.

``run_study`` takes a provider, an event calendar and a cost assumption, and
returns a single report: is the event variance risk premium there, does it
survive the spread, does conditioning on a volatility regime improve it, and is
the sample large enough to say so.

It is deliberately opinionated about what gets printed.  A research tool that
lets you read a favourable number without also reading the effective sample
size behind it will, eventually, let you trade one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.straddle import CostModel
from ..data.base import BarRequest
from .eventstudy import build_event_panel, premium_report, summarise_events
from .inference import bootstrap_ci
from .regime import (GaussianHMM, homogeneity_test, markov_order_test,
                     select_n_states)

__all__ = ["StudyConfig", "StudyResult", "run_study", "fit_regime",
           "HOLD_DAYS_BY_EXIT"]

# Calendar-day fractions from the entry close to the exit mark. 16:00 to the
# next 09:30 is 17.5 hours; 16:00 to the next 16:00 is 24.
HOLD_DAYS_BY_EXIT = {"open": 17.5 / 24.0, "close": 1.0}


@dataclass
class StudyConfig:
    costs: CostModel = field(default_factory=CostModel)
    exit_on: str = "open"                 # pure overnight hold
    hold_days: float | None = None
    """Calendar days of theta charged across the hold. ``None`` derives it from
    ``exit_on``, which is what you want: a close-to-next-open hold is 17.5 hours
    (0.73 days), not 24. Charging a full day on an overnight hold flatters the
    short straddle by roughly a quarter of a day's decay."""
    risk_free: float = 0.04
    dividend_yield: float = 0.0
    strike_increment: float = 2.5
    # regime overlay
    regime_observable: str = "log_range"  # or "abs_return"
    regime_candidates: tuple[int, ...] = (1, 2, 3)
    regime_lookback: int = 63
    # honesty knobs
    n_trials: int = 1                     # how many variants you searched
    n_boot: int = 2000
    seed: int = 0


@dataclass
class StudyResult:
    panel: pd.DataFrame
    coverage: str
    distribution: pd.DataFrame
    performance: pd.DataFrame
    verdict: str
    regime: dict
    conditional: pd.DataFrame
    trades: pd.DataFrame

    def report(self) -> str:
        lines = [
            "=" * 78,
            "EVENT VARIANCE RISK PREMIUM STUDY",
            "=" * 78,
            f"Coverage: {self.coverage}",
            "",
            "-- Implied vs realised event move " + "-" * 44,
            self.distribution.to_string(float_format=lambda v: f"{v: .4f}"),
            "",
            "-- Short straddle, net of costs " + "-" * 46,
            self.performance.to_string(float_format=lambda v: f"{v: .4f}"),
            "",
            "-- Volatility regime overlay " + "-" * 49,
            self._regime_block(),
            "",
            "-- Conditional performance " + "-" * 51,
            self._conditional_block(),
            "",
            "VERDICT: " + self.verdict,
            "=" * 78,
        ]
        return "\n".join(lines)

    def _conditional_block(self) -> str:
        """The table plus the caveats that make it readable.

        These warnings used to live in ``DataFrame.attrs``, which ``to_string``
        silently drops -- so the one number they exist to defuse (a regime with
        a small p-value on a couple of dozen trades) printed with no caveat at
        all. They are printed inline now.
        """
        if self.conditional.empty:
            return "  (no regime split applied)"
        k = len(self.conditional)
        out = [self.conditional.to_string(float_format=lambda v: f"{v: .4f}")]
        if k > 1:
            smallest = int(self.conditional["n"].min())
            out += [
                "",
                "  Read this table with three caveats:",
                "  1. A regime is only useful if the intervals do NOT overlap. "
                "Overlapping",
                "     intervals mean you split a small sample, not that you found a "
                "conditional edge.",
                f"  2. Splitting {k} ways is {k} more tests. The p_le_zero column is "
                "NOT corrected",
                f"     for that; a Bonferroni-style reading multiplies each by {k}. "
                f"The smallest cell here holds {smallest} trades.",
                "  3. The HMM parameters were fitted on the whole bar history, so the "
                "state",
                "     labels are in-sample at the parameter level even though the "
                "filter itself",
                "     uses no future data. See fit_regime(fit_through=...) for the "
                "walk-forward version.",
            ]
        return "\n".join(out)

    def _regime_block(self) -> str:
        r = self.regime
        if not r.get("fitted"):
            return "  regime model not fitted: " + r.get("reason", "unknown")
        # BIC on the OBSERVATIONS leads, because it is the evidence that is not
        # contaminated by the filter. The LR test below runs on filtered labels,
        # whose serial dependence is partly induced by the model's own sticky
        # transition matrix -- structurally the same criticism this module makes
        # of the tutorial build it replaces, so it is stated rather than hidden.
        out = [
            f"  states selected by BIC: {r['n_states']}  (candidates {r['candidates']})",
            f"  primary evidence -- BIC(k=1) - BIC(k={r['n_states']}) = "
            f"{r.get('delta_bic', float('nan')):.1f}  "
            f"(positive favours regimes; this is computed on the observations, "
            f"not on labels)",
        ]
        if r.get("fit_through") is not None:
            out.append(f"  parameters estimated on data through "
                       f"{r['fit_through']:%Y-%m-%d} ({r.get('n_train')} obs), "
                       f"then filtered forward -- no parameter look-ahead")
        else:
            out.append("  parameters estimated on the FULL sample: state labels are "
                       "in-sample at the")
            out.append("  parameter level. Pass fit_regime(fit_through=...) for the "
                       "walk-forward version.")
        out.append(
            f"  secondary -- Markov-order LR on filtered labels: "
            f"stat={r['markov_lr']:.1f} df={r['markov_df']} p={r['markov_p']:.3g}  "
            f"[confounded: the filter's own sticky prior induces label dependence, "
            f"so a small p here is partly the machinery]")
        if r.get("homogeneity_p") is not None:
            out.append(
                f"  time-homogeneity LR (2 splits): p={r['homogeneity_p']:.3g}"
                + ("  -> transition matrix is NOT stable across the sample; a single "
                   "Markov chain is misspecified" if r["homogeneity_p"] < 0.05 else ""))
        if r["markov_p"] > 0.05:
            out.append("  -> cannot reject serial independence even with the filter's "
                       "help: the overlay is decoration, not structure.")
        short = [k for k, d in enumerate(r["durations"]) if d < 3.0]
        if short:
            out.append(
                f"  -> states {short} have an expected duration below 3 bars. "
                "A state you leave almost immediately is not a regime, it is a "
                "flexible density; do not condition a trade on it.")
        out.append("  " + r["summary"].replace("\n", "\n  "))
        return "\n".join(out)


def _regime_observable(bars: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "log_range":
        # log high-low range: strictly positive, far better behaved than the
        # raw (h-l)/c ratio, whose skew makes a Gaussian emission misspecified.
        return np.log((bars["high"] / bars["low"]).clip(lower=1 + 1e-9))
    if kind == "abs_return":
        return np.log(bars["close"]).diff().abs()
    raise ValueError(f"unknown regime observable {kind!r}")


def fit_regime(bars: pd.DataFrame, config: StudyConfig | None = None,
               fit_through: pd.Timestamp | str | None = None) -> dict:
    """Fit the volatility-regime overlay once, so it can be reused across cost
    scenarios instead of being refitted (and re-randomised) each time.

    Parameters
    ----------
    fit_through:
        Estimate the HMM parameters using bars up to and including this date
        only, then *filter* the whole sample under those frozen parameters.
        This removes the parameter-level look-ahead: with ``fit_through=None``
        (the default) the EM sees the entire history, so a state label in 2015
        was produced by parameters that had seen 2026.

        The filtered probabilities are honest either way at fixed parameters --
        that is what ``test_filtered_probabilities_use_no_future_information``
        proves -- but the parameters themselves are not, and for anything you
        intend to present as tradeable you want this set.

    Returns the dict :func:`run_study` accepts as ``regime=``.
    """
    cfg = config or StudyConfig()
    out: dict = {"fitted": False, "candidates": cfg.regime_candidates,
                 "fit_through": pd.Timestamp(fit_through) if fit_through else None}
    try:
        obs = _regime_observable(bars, cfg.regime_observable).dropna()
        train = obs.loc[:pd.Timestamp(fit_through)] if fit_through else obs
        if len(train) < 30 * max(cfg.regime_candidates):
            out["reason"] = (f"only {len(train)} observations before "
                             f"{fit_through}; too few to estimate a regime model")
            return out

        sel = select_n_states(train.to_numpy(), candidates=cfg.regime_candidates,
                              n_init=3, random_state=cfg.seed)
        k = sel["best_bic"]
        out.update({"n_states": k, "selection": sel["table"],
                    "bic_table": sel["table"], "n_train": int(len(train))})
        if k == 1:
            out["reason"] = ("BIC selects a single state; there are no regimes "
                             "in this observable.")
            return out

        model = GaussianHMM(n_states=k, n_init=4, random_state=cfg.seed).fit(train.to_numpy())
        # Parameters frozen on `train`; filtering runs over the whole sample.
        filtered = model.filter(obs.to_numpy())
        state = pd.Series(filtered.argmax(axis=1), index=obs.index, name="regime")

        mo = markov_order_test(state.to_numpy(), n_states=k)
        hom = homogeneity_test(state.to_numpy(), n_splits=2, n_states=k)
        bic1 = float(sel["table"].loc[1, "bic"]) if 1 in sel["table"].index else np.nan
        bick = float(sel["table"].loc[k, "bic"])

        out.update({
            "fitted": True, "model": model, "state": state,
            "markov_lr": mo["lr_stat"], "markov_df": mo["df"], "markov_p": mo["pvalue"],
            "homogeneity_p": hom["pvalue"], "homogeneity_lr": hom["lr_stat"],
            "bic_1": bic1, "bic_k": bick, "delta_bic": bic1 - bick,
            "summary": model.summary(), "durations": model.expected_durations(),
        })
    except (ValueError, RuntimeError) as exc:
        out["reason"] = str(exc)
    return out


def run_study(provider, events: pd.DataFrame, symbol: str,
              config: StudyConfig | None = None,
              regime: dict | None = None) -> StudyResult:
    """Run the whole pipeline against any provider satisfying the data protocol.

    Pass ``regime`` from :func:`fit_regime` to reuse a single fitted overlay
    across several cost assumptions -- refitting per scenario would let the
    random restarts, not the costs, move the answer.
    """
    cfg = config or StudyConfig()
    hold_days = cfg.hold_days
    if hold_days is None:
        hold_days = HOLD_DAYS_BY_EXIT[cfg.exit_on]
    start = pd.Timestamp(events.index.min()) - pd.Timedelta(days=600)
    end = pd.Timestamp(events.index.max()) + pd.Timedelta(days=30)
    bars = provider.bars(BarRequest(symbol, start, end))

    panel_obj = build_event_panel(events, bars, exit_on=cfg.exit_on)
    panel = panel_obj.frame
    if panel.empty:
        raise RuntimeError(f"no usable events: {panel_obj.coverage()}")

    distribution = summarise_events(panel, n_boot=cfg.n_boot, seed=cfg.seed)
    rep = premium_report(
        panel, costs=cfg.costs, n_trials=cfg.n_trials, n_boot=cfg.n_boot,
        seed=cfg.seed, r=cfg.risk_free, q=cfg.dividend_yield,
        hold_days=hold_days, strike_increment=cfg.strike_increment,
    )

    # ---------------- regime overlay ---------------- #
    regime = regime if regime is not None else fit_regime(bars, cfg)
    conditional = pd.DataFrame()
    if regime.get("fitted"):
        conditional = _conditional_performance(rep["trades"], regime["state"])

    return StudyResult(
        panel=panel, coverage=panel_obj.coverage(), distribution=distribution,
        performance=rep["stats"], verdict=rep["verdict"], regime=regime,
        conditional=conditional, trades=rep["trades"],
    )


def _conditional_performance(trades: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    """P&L split by the regime filtered as at the **entry** session.

    Two distinct look-aheads have to be avoided here and only one of them is
    obvious.

    The obvious one: use ``filter`` rather than ``smooth`` or ``viterbi``, so
    the state at t conditions on nothing after t.

    The subtle one: align on ``entry_date``, not on the event date.  For a
    before-the-open reporter the event-date bar is the *post*-announcement
    session, and the regime observable is that bar's own high-low range -- so
    aligning on the event date would condition a position opened at the previous
    close on a state computed partly from the announcement move itself.  The
    trade frame carries ``entry_date`` for exactly this reason.
    """
    if trades.empty or state.empty:
        return pd.DataFrame()
    if "entry_date" not in trades.columns:
        raise ValueError(
            "trades frame has no entry_date column; conditioning on the event "
            "date leaks the announcement move into the regime for BMO reporters"
        )
    keys = pd.DatetimeIndex(trades["entry_date"])
    aligned = state.reindex(state.index.union(keys)).ffill().reindex(keys)
    rows = []
    for k, grp in trades.assign(regime=aligned.to_numpy()).groupby("regime"):
        pnl = grp["net_pnl"].to_numpy(float)
        ci = bootstrap_ci(pnl, np.mean, block=1, n_boot=800)
        rows.append({
            "regime": int(k), "n": len(pnl), "hit_rate": float((pnl > 0).mean()),
            "mean_pnl": ci["point"], "boot_lo": ci["lo"], "boot_hi": ci["hi"],
            "p_le_zero": ci["p_le_zero"], "worst": float(pnl.min()),
        })
    out = pd.DataFrame(rows).set_index("regime")
    return out
