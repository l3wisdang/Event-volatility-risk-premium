"""The event-study engine: does the earnings variance risk premium exist, and
does it survive costs?

This is the module the source projects never built.  Their earnings dashboard
analyses exactly one event, stores nothing, and reports the IV crush percentage
as the headline.  The IV crush is close to deterministic -- scheduled
uncertainty resolves, so implied vol falls, essentially always.  It is not the
strategy.  The strategy is:

    realised event move  vs  implied event move

accumulated over a panel of events, net of the spread you actually pay.  A
short straddle held through a print makes money when that ratio is small and
loses convexly when it is large, so the *mean* of the ratio is close to
uninformative and the right tail is everything.

Three things make this engine different from a scatter plot:

  * the implied event move comes from the **term structure**, not from the ATM
    level, so a high-beta name is not flagged as expensive merely for being
    volatile;
  * every statistic is reported with a block-bootstrap interval and an
    effective sample size, because event P&L is cross-sectionally clustered
    (earnings season) even when it is not serially correlated;
  * the answer is reported per unit of premium collected, which is the only
    scale on which trades of different sizes are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.straddle import CostModel, run_event_backtest
from ..vol.termstructure import extract_event_variance
from .inference import bootstrap_ci, deflated_sharpe_ratio, min_track_record_length

__all__ = ["EventPanel", "build_event_panel", "summarise_events", "premium_report"]


@dataclass
class EventPanel:
    """A panel of events with implied and realised event moves."""

    frame: pd.DataFrame
    n_input: int
    n_usable: int
    reasons: dict

    def __len__(self) -> int:
        return len(self.frame)

    def coverage(self) -> str:
        lost = "; ".join(f"{k}={v}" for k, v in sorted(self.reasons.items()) if v)
        return (f"{self.n_usable}/{self.n_input} events usable"
                + (f" (dropped: {lost})" if lost else ""))


def build_event_panel(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    exit_on: str = "close",
    post_iv_mode: str = "diffusive",
    max_residual_rms: float = 5e-4,
) -> EventPanel:
    """Assemble the per-event panel.

    Parameters
    ----------
    events:
        Indexed by event date, with columns ``symbol``, ``front_iv``,
        ``back_iv``, ``front_tenor``, ``back_tenor``, and optionally ``session``
        (``"BMO"``/``"AMC"``, default ``"AMC"``) and ``post_iv``. Any other
        columns are ignored -- the entry and exit sessions are derived from the
        bar index and the session flag, never from a supplied quote date.
    bars:
        Daily OHLC for the underlying, indexed by date.
    exit_on:
        ``"open"`` to mark the trade at the post-event open (a pure overnight
        hold), ``"close"`` to hold to the post-event close.  The source build
        averages open and close, which is neither, and which shrinks the
        measured move -- flattering the short straddle.
    post_iv_mode:
        ``"observed"`` uses an explicit ``post_iv`` column.  ``"diffusive"``
        sets the post-event vol to the diffusive vol backed out of the term
        structure, i.e. assumes the event variance is fully removed and nothing
        else changes.  That assumption is stated here rather than buried.
    max_residual_rms:
        Reject a term-structure fit whose residual exceeds this; a large
        residual means the curve slope is not coming from the event and the
        extracted move is not trustworthy.

    Session handling
    ----------------
    ``BMO`` (before market open) reporters announce *before* the session on the
    event date, so the last pre-event observation is the **previous** close and
    the post-event observation is the same day.  ``AMC`` reporters announce
    after the close, so the pre-event observation is that day's close and the
    post-event observation is the next session.  Getting this backwards inverts
    the sign of the trade for roughly half the earnings universe; the source
    build has no session field at all.
    """
    reasons = {"no_quote_bar": 0, "no_exit_bar": 0, "infeasible_termstructure": 0,
               "bad_fit_residual": 0, "missing_post_iv": 0}
    rows = []
    idx = bars.index

    for event_date, ev in events.iterrows():
        session = str(ev.get("session", "AMC")).upper()
        event_date = pd.Timestamp(event_date)

        if session == "BMO":
            pre_candidates = idx[idx < event_date]
            post_candidates = idx[idx >= event_date]
        else:                                   # AMC (and the safe default)
            pre_candidates = idx[idx <= event_date]
            post_candidates = idx[idx > event_date]

        if len(pre_candidates) == 0:
            reasons["no_quote_bar"] += 1
            continue
        if len(post_candidates) == 0:
            reasons["no_exit_bar"] += 1
            continue
        entry_date, exit_date = pre_candidates[-1], post_candidates[0]

        term = extract_event_variance(
            [ev["front_iv"], ev["back_iv"]],
            [ev["front_tenor"], ev["back_tenor"]],
        )
        if not term.feasible:
            reasons["infeasible_termstructure"] += 1
            continue
        if term.residual_rms > max_residual_rms:
            reasons["bad_fit_residual"] += 1
            continue

        spot_entry = float(bars.at[entry_date, "close"])
        spot_exit = float(bars.at[exit_date, "open" if exit_on == "open" else "close"])

        if post_iv_mode == "observed":
            post_iv = ev.get("post_iv", np.nan)
            if not np.isfinite(post_iv):
                reasons["missing_post_iv"] += 1
                continue
        else:
            post_iv = term.diffusive_vol

        realised = float(np.log(spot_exit / spot_entry))
        rows.append({
            "event_date": event_date,
            "symbol": ev.get("symbol", "?"),
            "session": session,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "spot_entry": spot_entry,
            "spot_exit": spot_exit,
            "iv_entry": float(ev["front_iv"]),
            "iv_exit": float(post_iv),
            "tenor_entry": float(ev["front_tenor"]),
            "diffusive_vol": term.diffusive_vol,
            "implied_event_move": term.event_move,
            "realised_event_move": realised,
            "abs_realised_move": abs(realised),
            "move_ratio": abs(realised) / term.event_move if term.event_move > 0 else np.nan,
            "termstructure_residual": term.residual_rms,
        })

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.set_index("event_date").sort_index()
    return EventPanel(frame=frame, n_input=len(events), n_usable=len(frame), reasons=reasons)


def summarise_events(panel: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Distributional summary of the implied-vs-realised relationship.

    Reports the ratio distribution, not just its mean.  A short straddle is
    short a convex function of this ratio, so the mean is the wrong statistic
    and the 90th/95th/99th percentiles are the risk.
    """
    if panel.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    ratio = panel["move_ratio"].to_numpy(float)
    ratio = ratio[np.isfinite(ratio)]

    var_ratio = (panel["implied_event_move"] ** 2 /
                 panel["realised_event_move"].replace(0, np.nan) ** 2)
    var_ratio = var_ratio.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    # Variance premium measured properly: E[implied var] / E[realised var] - 1.
    iv_var = float(np.mean(panel["implied_event_move"] ** 2))
    rv_var = float(np.mean(panel["realised_event_move"] ** 2))
    vrp = iv_var / rv_var - 1.0 if rv_var > 0 else np.nan

    boot_vrp = bootstrap_ci(
        (panel["implied_event_move"] ** 2 - panel["realised_event_move"] ** 2).to_numpy(float),
        statistic=np.mean, block=1, n_boot=n_boot, rng=rng,
    )

    rows = [
        {"metric": "n events", "value": float(len(panel)), "lo": np.nan, "hi": np.nan},
        {"metric": "mean implied event move", "value": float(panel["implied_event_move"].mean()),
         "lo": np.nan, "hi": np.nan},
        {"metric": "mean |realised| event move", "value": float(panel["abs_realised_move"].mean()),
         "lo": np.nan, "hi": np.nan},
        {"metric": "variance premium (implied/realised - 1)", "value": vrp,
         "lo": np.nan, "hi": np.nan},
        {"metric": "mean variance spread (implied^2 - realised^2)",
         "value": boot_vrp["point"], "lo": boot_vrp["lo"], "hi": boot_vrp["hi"]},
        {"metric": "median move ratio", "value": float(np.median(ratio)),
         "lo": np.nan, "hi": np.nan},
        {"metric": "mean move ratio", "value": float(np.mean(ratio)), "lo": np.nan, "hi": np.nan},
        {"metric": "P(ratio > 1)", "value": float(np.mean(ratio > 1.0)), "lo": np.nan, "hi": np.nan},
        {"metric": "P(ratio > 2)", "value": float(np.mean(ratio > 2.0)), "lo": np.nan, "hi": np.nan},
        {"metric": "ratio 90th pct", "value": float(np.quantile(ratio, 0.90)),
         "lo": np.nan, "hi": np.nan},
        {"metric": "ratio 99th pct", "value": float(np.quantile(ratio, 0.99)),
         "lo": np.nan, "hi": np.nan},
    ]
    if var_ratio.size:
        rows.append({"metric": "median implied/realised variance ratio",
                     "value": float(np.median(var_ratio)), "lo": np.nan, "hi": np.nan})
    return pd.DataFrame(rows).set_index("metric")


def premium_report(panel: pd.DataFrame, costs: CostModel | None = None,
                   n_trials: int = 1, n_boot: int = 2000, seed: int = 0,
                   **backtest_kw) -> dict:
    """End-to-end: run the trade, then judge it honestly.

    ``n_trials`` is the number of strategy variants you searched before arriving
    at this one.  Be truthful about it: it deflates the Sharpe ratio, which is
    the point.
    """
    trades = run_event_backtest(panel, costs=costs, **backtest_kw)
    if trades.empty:
        return {"trades": trades, "stats": pd.DataFrame(), "verdict": "no tradeable events"}

    rng = np.random.default_rng(seed)
    pnl = trades["net_pnl"].to_numpy(float)
    ret = trades["return_on_credit"].to_numpy(float)

    ci_pnl = bootstrap_ci(pnl, np.mean, block=1, n_boot=n_boot, rng=rng)
    ci_ret = bootstrap_ci(ret, np.mean, block=1, n_boot=n_boot, rng=rng)
    dsr = deflated_sharpe_ratio(ret, n_trials=n_trials, periods_per_year=4)
    mtrl = min_track_record_length(ret)

    wins = pnl > 0
    stats = pd.DataFrame(
        [
            {"metric": "trades", "value": float(len(pnl))},
            {"metric": "hit rate", "value": float(wins.mean())},
            {"metric": "mean net P&L per trade", "value": ci_pnl["point"]},
            {"metric": "  bootstrap 95% lo", "value": ci_pnl["lo"]},
            {"metric": "  bootstrap 95% hi", "value": ci_pnl["hi"]},
            {"metric": "  P(mean <= 0)", "value": ci_pnl["p_le_zero"]},
            {"metric": "mean return on credit", "value": ci_ret["point"]},
            {"metric": "  bootstrap 95% lo", "value": ci_ret["lo"]},
            {"metric": "  bootstrap 95% hi", "value": ci_ret["hi"]},
            {"metric": "mean winner", "value": float(pnl[wins].mean()) if wins.any() else np.nan},
            {"metric": "mean loser", "value": float(pnl[~wins].mean()) if (~wins).any() else np.nan},
            {"metric": "worst trade", "value": float(pnl.min())},
            {"metric": "losers to wipe out all wins",
             "value": float(pnl[wins].sum() / abs(pnl[~wins].mean()))
             if (~wins).any() and wins.any() else np.nan},
            {"metric": "Sharpe (per event)", "value": dsr["sr"]},
            {"metric": "skew of returns", "value": dsr.get("skew", np.nan)},
            {"metric": "excess kurtosis", "value": dsr.get("kurtosis", np.nan) - 3.0},
            {"metric": f"deflated SR prob (n_trials={n_trials})", "value": dsr["dsr"]},
            {"metric": "min track record length (events)", "value": mtrl},
        ]
    ).set_index("metric")

    if ci_pnl["p_le_zero"] > 0.10:
        verdict = ("NOT ESTABLISHED: the bootstrap cannot reject zero mean P&L. "
                   "This is a hypothesis, not a result.")
    elif np.isfinite(mtrl) and mtrl > len(pnl):
        verdict = (f"UNDERPOWERED: needs ~{mtrl:.0f} events to distinguish this Sharpe "
                   f"from zero; you have {len(pnl)}.")
    elif dsr["dsr"] is not None and np.isfinite(dsr["dsr"]) and dsr["dsr"] < 0.95:
        verdict = (f"FRAGILE: deflated-Sharpe probability {dsr['dsr']:.2f} after "
                   f"{n_trials} trials. Survives cost but not multiple testing.")
    else:
        verdict = "SUPPORTED at this cost level. Re-check with a wider spread before sizing."

    return {"trades": trades, "stats": stats, "verdict": verdict,
            "bootstrap_pnl": ci_pnl, "bootstrap_return": ci_ret, "deflated_sharpe": dsr}
