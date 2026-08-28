"""Regression tests for the findings of the external audit.

One test per finding, each named for what it would catch if the fix regressed.
The two that matter most are the straddle-conversion constant (a numerical
error in an exported helper) and the entry-date conditioning (a genuine
look-ahead that the synthetic generator could not expose, because it emits only
after-the-close reporters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from vrplab.backtest.straddle import CostModel, price_event_straddle, run_event_backtest
from vrplab.live.signal import RiskLimits, screen_event, size_position
from vrplab.research.study import (HOLD_DAYS_BY_EXIT, StudyConfig, fit_regime,
                                   run_study, _conditional_performance)
from vrplab.vol.implied import (SQRT_2_OVER_PI, SQRT_PI_OVER_2,
                                atm_straddle_to_expected_abs_move,
                                atm_straddle_to_implied_move, straddle_price)


# --------------------------------------------------------------------------- #
#  Finding 1: the straddle -> move conversion was out by a factor of 2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sig,T", [(0.60, 7 / 365), (0.25, 30 / 365), (1.20, 3 / 365)])
def test_straddle_to_one_sigma_move_matches_the_closed_form(sig, T):
    """ATM straddle = 2S[2N(x/2) - 1] with x = sigma*sqrt(T), so
    x ~= sqrt(pi/2) * straddle/S. The helper previously returned half of this."""
    S = 100.0
    st = float(straddle_price(S, S, T, 0.0, sig, 0.0))
    assert st == pytest.approx(2 * S * (2 * norm.cdf(sig * np.sqrt(T) / 2) - 1), abs=1e-9)
    recovered = float(atm_straddle_to_implied_move(st, S))
    assert recovered == pytest.approx(sig * np.sqrt(T), rel=2e-3)


@pytest.mark.parametrize("sig,T", [(0.60, 7 / 365), (0.25, 30 / 365)])
def test_straddle_over_spot_is_the_expected_absolute_move(sig, T):
    """The desk shorthand is the correct leading-order number, not a fudge:
    E|move| = sigma*sqrt(T)*sqrt(2/pi) = straddle/S."""
    S = 100.0
    st = float(straddle_price(S, S, T, 0.0, sig, 0.0))
    assert float(atm_straddle_to_expected_abs_move(st, S)) == pytest.approx(st / S)
    assert st / S == pytest.approx(sig * np.sqrt(T) * SQRT_2_OVER_PI, rel=2e-3)


def test_the_two_move_conversions_differ_by_the_right_constant():
    S, sig, T = 100.0, 0.5, 14 / 365
    st = float(straddle_price(S, S, T, 0.0, sig, 0.0))
    one_sigma = float(atm_straddle_to_implied_move(st, S))
    exp_abs = float(atm_straddle_to_expected_abs_move(st, S))
    assert one_sigma / exp_abs == pytest.approx(SQRT_PI_OVER_2, rel=1e-12)
    assert one_sigma > exp_abs, "a one-sigma move exceeds the mean absolute move"


# --------------------------------------------------------------------------- #
#  Finding 2: conditioning must align on entry_date, not the event date
# --------------------------------------------------------------------------- #
def _bmo_fixture():
    """A BMO event where the regime flips ON the announcement day.

    Entry is the previous close, so honest conditioning must report the OLD
    regime. Aligning on the event date reports the new one -- a state computed
    partly from the announcement move itself.
    """
    idx = pd.bdate_range("2024-01-01", periods=10)
    event_date, entry_date = idx[5], idx[4]
    trades = pd.DataFrame(
        {"symbol": ["X"], "entry_date": [entry_date], "net_pnl": [500.0]},
        index=pd.DatetimeIndex([event_date], name="event_date"),
    )
    state = pd.Series(0, index=idx, name="regime")
    state.loc[event_date:] = 1                     # regime flips on the print
    return trades, state, idx


def test_conditional_performance_uses_the_entry_session():
    trades, state, _ = _bmo_fixture()
    out = _conditional_performance(trades, state)
    assert list(out.index) == [0], (
        "conditioned on the event-date regime; for a BMO reporter that bar is "
        "post-announcement, so the announcement move leaks into the state"
    )


def test_conditional_performance_refuses_a_frame_without_entry_date():
    trades, state, _ = _bmo_fixture()
    with pytest.raises(ValueError, match="entry_date"):
        _conditional_performance(trades.drop(columns=["entry_date"]), state)


def test_backtest_carries_entry_date_through_from_the_panel():
    idx = pd.bdate_range("2024-01-01", periods=6)
    panel = pd.DataFrame(
        {"symbol": ["X"], "spot_entry": [100.0], "spot_exit": [101.0],
         "iv_entry": [0.5], "iv_exit": [0.3], "tenor_entry": [0.02],
         "entry_date": [idx[3]]},
        index=pd.DatetimeIndex([idx[4]], name="event_date"),
    )
    trades = run_event_backtest(panel)
    assert trades.iloc[0]["entry_date"] == idx[3]


def test_entry_date_defaults_to_the_event_date_when_absent():
    t = price_event_straddle(symbol="X", event_date=pd.Timestamp("2024-05-01"),
                             spot_entry=100.0, spot_exit=100.0, iv_entry=0.4,
                             iv_exit=0.3, tenor_entry=0.02)
    assert t.entry_date == pd.Timestamp("2024-05-01")


# --------------------------------------------------------------------------- #
#  Finding 3: the caveats have to actually print
# --------------------------------------------------------------------------- #
def test_conditional_caveats_appear_in_the_printed_report():
    """They lived in DataFrame.attrs, which to_string() drops -- so the number
    they exist to defuse printed bare."""
    from vrplab.data.synthetic import SyntheticConfig, SyntheticProvider
    p = SyntheticProvider(SyntheticConfig(n_days=1500, seed=5))
    res = run_study(p, p.events(), p.config.symbol,
                    StudyConfig(n_boot=200, regime_candidates=(1, 2), seed=1))
    text = res.report()
    if not res.conditional.empty and len(res.conditional) > 1:
        assert "do NOT overlap" in text
        assert "more tests" in text          # multiple-testing caveat
        assert "p_le_zero column is NOT corrected" in text


def test_report_declares_whether_regime_parameters_saw_the_future():
    from vrplab.data.synthetic import SyntheticConfig, SyntheticProvider
    p = SyntheticProvider(SyntheticConfig(n_days=1500, seed=5))
    res = run_study(p, p.events(), p.config.symbol,
                    StudyConfig(n_boot=200, regime_candidates=(1, 2), seed=1))
    text = res.report()
    if res.regime.get("fitted"):
        assert ("FULL sample" in text) or ("no parameter look-ahead" in text)
        assert "confounded" in text, "the filtered-label LR test must be caveated"


# --------------------------------------------------------------------------- #
#  Finding 4: walk-forward regime fitting must exist and must bind
# --------------------------------------------------------------------------- #
def test_fit_through_freezes_parameters_on_the_training_window():
    from vrplab.data.synthetic import SyntheticConfig, SyntheticProvider
    p = SyntheticProvider(SyntheticConfig(n_days=2000, seed=9))
    bars = p.full_bars()
    cut = bars.index[len(bars) // 2]

    full = fit_regime(bars, StudyConfig(regime_candidates=(1, 2), seed=1))
    walk = fit_regime(bars, StudyConfig(regime_candidates=(1, 2), seed=1), fit_through=cut)
    assert walk["fit_through"] == cut
    assert full["fit_through"] is None
    if full.get("fitted") and walk.get("fitted"):
        # Same length of filtered state (whole sample), different parameters.
        assert len(walk["state"]) == len(full["state"])
        assert walk["n_train"] < len(bars)
        assert not np.allclose(walk["model"].means_, full["model"].means_, atol=1e-12)


def test_fit_through_too_early_declines_rather_than_fitting_noise():
    from vrplab.data.synthetic import SyntheticConfig, SyntheticProvider
    p = SyntheticProvider(SyntheticConfig(n_days=800, seed=9))
    bars = p.full_bars()
    out = fit_regime(bars, StudyConfig(regime_candidates=(1, 2, 3), seed=1),
                     fit_through=bars.index[40])
    assert not out["fitted"]
    assert "too few" in out["reason"]


# --------------------------------------------------------------------------- #
#  Finding 5: theta charged must match the hold actually taken
# --------------------------------------------------------------------------- #
def test_overnight_hold_charges_less_than_a_full_day_of_theta():
    assert HOLD_DAYS_BY_EXIT["open"] == pytest.approx(17.5 / 24.0)
    assert HOLD_DAYS_BY_EXIT["close"] == 1.0
    kw = dict(symbol="X", event_date=pd.Timestamp("2024-01-10"), spot_entry=100.0,
              spot_exit=100.0, iv_entry=0.60, iv_exit=0.60, tenor_entry=7 / 365,
              costs=CostModel(half_spread_pct=0.0, commission_per_contract=0.0))
    overnight = price_event_straddle(**kw, hold_days=HOLD_DAYS_BY_EXIT["open"])
    full_day = price_event_straddle(**kw, hold_days=1.0)
    assert 0 < overnight.net_pnl < full_day.net_pnl


def test_study_derives_hold_days_from_exit_on():
    from vrplab.data.synthetic import SyntheticConfig, SyntheticProvider
    p = SyntheticProvider(SyntheticConfig(n_days=1200, seed=3))
    ev = p.events()
    a = run_study(p, ev, p.config.symbol,
                  StudyConfig(exit_on="open", n_boot=100, regime_candidates=(1,), seed=1))
    assert a.trades["tenor_exit"].iloc[0] == pytest.approx(
        a.trades["tenor_entry"].iloc[0] - HOLD_DAYS_BY_EXIT["open"] / 365.0)


# --------------------------------------------------------------------------- #
#  Finding 6: the two ratio columns must not share a name
# --------------------------------------------------------------------------- #
def test_panel_and_trade_ratio_columns_are_named_distinctly():
    from vrplab.data.synthetic import SyntheticConfig, SyntheticProvider
    from vrplab.research.eventstudy import build_event_panel
    p = SyntheticProvider(SyntheticConfig(n_days=1200, seed=3))
    ev = p.events()
    bars = p.full_bars()
    panel = build_event_panel(ev, bars, exit_on="open").frame
    trades = run_event_backtest(panel)
    assert "move_ratio" in panel.columns
    assert "move_ratio" not in trades.columns
    assert "move_vs_implied_sd" in trades.columns
    # And joining the two must not collide.
    joined = panel.join(trades[["move_vs_implied_sd", "net_pnl"]], how="inner")
    assert len(joined) == len(trades)
    # The panel ratio divides by J alone, so it is the larger of the two.
    assert (joined["move_ratio"] >= joined["move_vs_implied_sd"] - 1e-12).all()


# --------------------------------------------------------------------------- #
#  Finding 7: sizing must use the estimated diffusive vol, not a guess
# --------------------------------------------------------------------------- #
def test_size_position_uses_the_supplied_post_event_vol():
    """Assert on the stress loss, not the contract count.

    At the default 4-sigma stress the straddle is ~40% in the money, so it is
    almost all intrinsic and the post-event vol barely moves the answer -- a
    real property of the sizing rule, not a bug. The vol assumption bites at
    moderate stress levels, which is where this checks it.
    """
    lim = RiskLimits(account_equity=1_000_000.0, stress_move_sigmas=1.5)
    args = (100.0, 100.0, 7 / 365, 0.90, 0.09, lim)
    n_guess, loss_guess = size_position(*args)                    # 0.6*iv fallback
    n_meas, loss_meas = size_position(*args, post_iv=0.25)        # real diffusive vol
    assert loss_guess != loss_meas, "post_iv is being ignored"
    # A lower post-event vol means less time value left in the short strike,
    # so the stressed buy-back is cheaper and more contracts are permitted.
    assert n_meas >= n_guess


def test_screen_event_passes_the_diffusive_vol_into_sizing():
    t1, t2 = 7 / 365, 35 / 365
    sd, J = 0.22, 0.09
    iv1 = np.sqrt((sd**2 * t1 + J**2) / t1)
    iv2 = np.sqrt((sd**2 * t2 + J**2) / t2)
    sig = screen_event(symbol="X", asof=pd.Timestamp("2026-01-05"), spot=100.0,
                       front_iv=iv1, back_iv=iv2, front_tenor=t1, back_tenor=t2,
                       historical_moves=np.full(20, 0.03), spread_pct_of_mid=0.03,
                       limits=RiskLimits(account_equity=1_000_000.0))
    assert sig.diffusive_vol == pytest.approx(sd, abs=1e-6)
    expected, _ = size_position(100.0, 100.0, t1, iv1, J,
                                RiskLimits(account_equity=1_000_000.0), post_iv=sd)
    assert sig.max_contracts == expected


def test_screen_compares_expected_abs_move_not_one_sigma():
    """history is a mean of absolute moves, so the implied side must be
    converted by sqrt(2/pi) before the ratio is formed."""
    t1, t2 = 7 / 365, 35 / 365
    sd, J = 0.22, 0.08
    iv1 = np.sqrt((sd**2 * t1 + J**2) / t1)
    iv2 = np.sqrt((sd**2 * t2 + J**2) / t2)
    hist = np.full(30, 0.04)
    sig = screen_event(symbol="X", asof=pd.Timestamp("2026-01-05"), spot=100.0,
                       front_iv=iv1, back_iv=iv2, front_tenor=t1, back_tenor=t2,
                       historical_moves=hist, spread_pct_of_mid=0.02,
                       limits=RiskLimits(account_equity=1_000_000.0))
    assert sig.implied_vs_history == pytest.approx(SQRT_2_OVER_PI * J / 0.04, rel=1e-6)


# --------------------------------------------------------------------------- #
#  Finding 8: small ones
# --------------------------------------------------------------------------- #
def test_synthetic_config_has_no_undelivered_ground_truth_field():
    from vrplab.data.synthetic import SyntheticConfig
    assert not hasattr(SyntheticConfig(), "spread_pct"), (
        "documented as ground truth but never used; costs come from CostModel"
    )


def test_bootstrap_ci_docstring_names_the_key_it_returns():
    from vrplab.research.inference import bootstrap_ci
    doc = bootstrap_ci.__doc__ or ""
    assert "p_le_zero" in doc and "p_gt_zero" not in doc
    out = bootstrap_ci(np.random.default_rng(0).normal(1, 1, 200), np.mean, block=1,
                       n_boot=200)
    assert "p_le_zero" in out
