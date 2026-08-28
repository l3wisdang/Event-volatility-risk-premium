"""Ground-truth tests: every estimator must recover a planted answer.

These are not smoke tests.  Each one plants a known quantity in simulated data
and asserts the estimator finds it, which is the only way to know an estimator
measures what its name claims.  A test suite that only checks "it runs without
raising" is what lets a dashboard ship with an IV of 698%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vrplab.backtest.straddle import CostModel, price_event_straddle
from vrplab.data.base import VolUnits, to_annualised
from vrplab.data.synthetic import SyntheticConfig, SyntheticProvider
from vrplab.research.eventstudy import build_event_panel, premium_report, summarise_events
from vrplab.research.inference import (bootstrap_ci, effective_sample_size, ols_hac)
from vrplab.research.regime import (GaussianHMM, homogeneity_test, markov_order_test,
                                    select_n_states, transition_matrix_from_labels)
from vrplab.vol.implied import (bs_price, implied_vol, straddle_greeks, straddle_price)
from vrplab.vol.realised import estimator_efficiency, realised_vol
from vrplab.vol.termstructure import extract_event_variance


@pytest.fixture(scope="module")
def provider():
    return SyntheticProvider(SyntheticConfig(n_days=3000, seed=11))


# --------------------------------------------------------------------------- #
#  Pricing
# --------------------------------------------------------------------------- #
def test_put_call_parity():
    S, K, T, r, sig, q = 100.0, 105.0, 0.25, 0.04, 0.30, 0.01
    c = float(bs_price(S, K, T, r, sig, q, "call"))
    p = float(bs_price(S, K, T, r, sig, q, "put"))
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_implied_vol_round_trip():
    for sig in (0.08, 0.25, 0.62, 1.85):
        px = float(bs_price(100, 100, 0.1, 0.04, sig, 0.0, "call"))
        assert implied_vol(px, 100, 100, 0.1, 0.04, 0.0, "call") == pytest.approx(sig, abs=1e-6)


def test_implied_vol_rejects_arbitrage_violations():
    # A price below intrinsic has no implied vol; returning a boundary value
    # silently is how an 800% vol reaches a backtest.
    assert np.isnan(implied_vol(0.0, 100, 50, 0.5, 0.04, 0.0, "call"))
    assert np.isnan(implied_vol(200.0, 100, 100, 0.5, 0.04, 0.0, "call"))


def test_greeks_match_finite_differences():
    S, K, T, r, sig = 100.0, 100.0, 0.08, 0.04, 0.45
    g = straddle_greeks(S, K, T, r, sig)
    h = 1e-5
    fd_delta = (straddle_price(S + h, K, T, r, sig) - straddle_price(S - h, K, T, r, sig)) / (2 * h)
    fd_vega = (straddle_price(S, K, T, r, sig + h) - straddle_price(S, K, T, r, sig - h)) / (2 * h)
    assert float(g["delta"]) == pytest.approx(float(fd_delta), abs=1e-5)
    assert float(g["vega"]) == pytest.approx(float(fd_vega), rel=1e-5)


def test_degenerate_inputs_do_not_raise():
    assert float(bs_price(100, 90, 0.0, 0.04, 0.3, 0.0, "call")) == pytest.approx(10.0, abs=1e-9)
    assert float(bs_price(100, 110, 0.0, 0.04, 0.3, 0.0, "call")) == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(float(bs_price(100, 100, 0.5, 0.04, 0.0, 0.0, "call")))


# --------------------------------------------------------------------------- #
#  Volatility units -- the bug that produced "IV in the hundreds"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,units,expected",
    [
        (0.35, VolUnits.ANNUAL_DECIMAL, 0.35),
        (35.0, VolUnits.ANNUAL_PERCENT, 0.35),
        (0.35 / np.sqrt(252), VolUnits.DAILY_DECIMAL, 0.35),
        (100 * 0.35 / np.sqrt(252), VolUnits.DAILY_PERCENT, 0.35),
    ],
)
def test_vol_unit_conversion_is_idempotent_and_correct(value, units, expected):
    assert float(to_annualised(value, units)) == pytest.approx(expected, rel=1e-12)


def test_annual_decimal_is_never_rescaled():
    """The specific failure: an already-annualised 0.44 multiplied by sqrt(252)
    becomes 698%. Declaring units makes that unrepresentable."""
    assert float(to_annualised(0.44, VolUnits.ANNUAL_DECIMAL)) == pytest.approx(0.44)


# --------------------------------------------------------------------------- #
#  Term structure: recover the planted event move
# --------------------------------------------------------------------------- #
def test_event_variance_recovers_planted_move_exactly():
    sigma_d, J = 0.28, 0.065
    t1, t2 = 7 / 365, 35 / 365
    iv1 = np.sqrt((sigma_d**2 * t1 + J**2) / t1)
    iv2 = np.sqrt((sigma_d**2 * t2 + J**2) / t2)
    res = extract_event_variance([iv1, iv2], [t1, t2])
    assert res.feasible
    assert res.event_move == pytest.approx(J, rel=1e-10)
    assert res.diffusive_vol == pytest.approx(sigma_d, rel=1e-10)


def test_event_variance_infeasible_returns_nan_not_zero():
    # Inverted curve the model cannot represent: must not silently clip.
    res = extract_event_variance([0.20, 0.60], [7 / 365, 35 / 365])
    assert not res.feasible
    assert np.isnan(res.event_move)


def test_event_variance_overdetermined_fit_reports_residual():
    sigma_d, J = 0.25, 0.05
    tenors = np.array([7, 21, 35, 63]) / 365
    ivs = np.sqrt((sigma_d**2 * tenors + J**2) / tenors)
    res = extract_event_variance(ivs, tenors)
    assert res.n_expiries == 4
    assert res.residual_rms == pytest.approx(0.0, abs=1e-12)
    assert res.event_move == pytest.approx(J, rel=1e-8)


def test_atm_level_alone_does_not_identify_the_event(provider):
    """The point of the whole module: two names with identical ATM IV can carry
    very different event premia, so screening on IV level is not a signal."""
    t1, t2 = 7 / 365, 35 / 365
    quiet = extract_event_variance(
        [np.sqrt((0.60**2 * t1 + 0.02**2) / t1), np.sqrt((0.60**2 * t2 + 0.02**2) / t2)],
        [t1, t2])
    eventful = extract_event_variance(
        [np.sqrt((0.20**2 * t1 + 0.09**2) / t1), np.sqrt((0.20**2 * t2 + 0.09**2) / t2)],
        [t1, t2])
    assert eventful.event_move > 4 * quiet.event_move


# --------------------------------------------------------------------------- #
#  Realised volatility
# --------------------------------------------------------------------------- #
def test_realised_estimators_are_close_to_truth_in_a_single_regime():
    rng = np.random.default_rng(3)
    n, true_vol = 4000, 0.24
    dt = 1 / 252
    r = rng.normal(0, true_vol * np.sqrt(dt), n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[100.0], close[:-1]])
    noise = np.abs(rng.normal(0, true_vol * np.sqrt(dt) * 0.5, (n, 2)))
    df = pd.DataFrame({
        "open": open_, "close": close,
        "high": np.maximum(open_, close) * np.exp(noise[:, 0]),
        "low": np.minimum(open_, close) * np.exp(-noise[:, 1]),
        "volume": 1.0,
    }, index=pd.bdate_range("2015-01-01", periods=n))
    eff = estimator_efficiency(df, window=63, true_vol=true_vol)
    assert not eff.empty
    assert abs(eff.loc["close_to_close", "bias_pct"]) < 10.0
    # Range-based estimators must be at least as accurate as close-to-close.
    assert eff["rmse"].min() <= eff.loc["close_to_close", "rmse"] + 1e-12


def test_unknown_estimator_raises_rather_than_defaulting():
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                       "close": [1.0], "volume": [1.0]},
                      index=pd.bdate_range("2020-01-01", periods=1))
    with pytest.raises(ValueError, match="unknown estimator"):
        realised_vol(df, 5, method="parkinsonn")


# --------------------------------------------------------------------------- #
#  Inference: the fix for the overlapping-sample p-value
# --------------------------------------------------------------------------- #
def test_hac_standard_errors_exceed_ols_on_overlapping_data():
    """Reproduces the source dashboard's regression and shows the correction.

    Regress a 30-day forward average on the current level of a persistent
    series. OLS reports a t-stat that is inflated by roughly sqrt(30); the HAC
    standard error must be materially larger.
    """
    rng = np.random.default_rng(5)
    n = 1200
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.98 * x[t - 1] + rng.normal(0, 0.02)     # near-unit-root, like IV
    s = pd.Series(x)
    fwd = s.rolling(30).mean().shift(-30)
    ok = fwd.notna()
    res = ols_hac(fwd[ok].to_numpy(), s[ok].to_numpy(), names=["current"], horizon=30)
    inflation = res.se_hac[1] / res.se_ols[1]
    assert inflation > 2.0, f"HAC inflation only {inflation:.2f}; correction not biting"
    assert res.lags == 29
    assert res.ess < res.nobs / 5


def test_effective_sample_size_collapses_under_strong_persistence():
    rng = np.random.default_rng(1)
    n = 2000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.97 * x[t - 1] + rng.normal()
    assert effective_sample_size(x) < n / 5
    assert effective_sample_size(rng.normal(size=n)) > n / 2


def test_hac_recovers_a_known_slope():
    rng = np.random.default_rng(2)
    n = 3000
    x = rng.normal(size=n)
    y = 1.5 + 0.8 * x + rng.normal(0, 0.5, n)
    res = ols_hac(y, x, names=["x"])
    assert res.params[0] == pytest.approx(1.5, abs=0.05)
    assert res.params[1] == pytest.approx(0.8, abs=0.05)


def test_bootstrap_ci_covers_the_true_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(0.5, 1.0, 800)
    ci = bootstrap_ci(x, np.mean, block=1, n_boot=1000, rng=rng)
    assert ci["lo"] < 0.5 < ci["hi"]
    assert ci["p_le_zero"] < 0.01


# --------------------------------------------------------------------------- #
#  Regimes
# --------------------------------------------------------------------------- #
def test_hmm_recovers_planted_regime_parameters():
    rng = np.random.default_rng(9)
    P = np.array([[0.97, 0.03], [0.06, 0.94]])
    mus, sds = np.array([0.0, 0.0]), np.array([0.006, 0.020])
    n = 6000
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(2, p=P[states[t - 1]])
    x = rng.normal(mus[states], sds[states])

    m = GaussianHMM(n_states=2, n_init=6, random_state=1).fit(x)
    assert np.sqrt(m.vars_[0]) == pytest.approx(sds[0], rel=0.15)
    assert np.sqrt(m.vars_[1]) == pytest.approx(sds[1], rel=0.15)
    assert m.transmat_[0, 0] == pytest.approx(P[0, 0], abs=0.05)
    assert m.transmat_[1, 1] == pytest.approx(P[1, 1], abs=0.05)

    # And the states are actually recovered, up to the canonical ordering.
    path = m.viterbi(x)
    assert (path == states).mean() > 0.85


def test_bic_prefers_one_state_when_there_are_no_regimes():
    """The null the source build never entertains. Homoskedastic data must not
    produce a three-regime model."""
    rng = np.random.default_rng(12)
    x = rng.normal(0, 0.01, 4000)
    sel = select_n_states(x, candidates=(1, 2, 3), n_init=4, random_state=3)
    assert sel["best_bic"] == 1


def test_bic_finds_two_states_when_two_exist():
    rng = np.random.default_rng(13)
    P = np.array([[0.98, 0.02], [0.05, 0.95]])
    states = np.zeros(5000, dtype=int)
    for t in range(1, 5000):
        states[t] = rng.choice(2, p=P[states[t - 1]])
    x = rng.normal(0, np.array([0.005, 0.025])[states])
    sel = select_n_states(x, candidates=(1, 2, 3), n_init=4, random_state=3)
    assert sel["best_bic"] == 2


def test_filtered_probabilities_use_no_future_information():
    """The look-ahead test. Filtered state probabilities up to time t must not
    change when data after t is appended; smoothed ones must."""
    rng = np.random.default_rng(17)
    x = rng.normal(0, np.where(rng.random(2000) < 0.5, 0.005, 0.02))
    m = GaussianHMM(n_states=2, n_init=4, random_state=2).fit(x)
    cut = 1200
    f_short = m.filter(x[:cut])
    f_long = m.filter(x)[:cut]
    assert np.allclose(f_short, f_long, atol=1e-12)

    s_short = m.smooth(x[:cut])
    s_long = m.smooth(x)[:cut]
    assert not np.allclose(s_short, s_long, atol=1e-6)


def test_markov_order_test_rejects_dependence_when_there_is_none():
    rng = np.random.default_rng(21)
    labels = rng.integers(0, 3, 5000)
    res = markov_order_test(labels, n_states=3)
    assert res["pvalue"] > 0.05


def test_markov_order_test_detects_real_dependence():
    rng = np.random.default_rng(22)
    P = np.array([[0.9, 0.08, 0.02], [0.1, 0.8, 0.1], [0.02, 0.08, 0.9]])
    labels = np.zeros(5000, dtype=int)
    for t in range(1, 5000):
        labels[t] = rng.choice(3, p=P[labels[t - 1]])
    res = markov_order_test(labels, n_states=3)
    assert res["pvalue"] < 1e-10
    _, P_hat = transition_matrix_from_labels(labels, 3)
    assert np.allclose(P_hat, P, atol=0.03)


def test_homogeneity_test_detects_a_regime_break():
    rng = np.random.default_rng(23)
    a = rng.choice(2, 2000, p=[0.9, 0.1])
    b = rng.choice(2, 2000, p=[0.3, 0.7])
    res = homogeneity_test(np.concatenate([a, b]), n_splits=2, n_states=2)
    assert res["pvalue"] < 0.01


def test_tiny_sample_is_flagged_not_silently_estimated():
    """60 bars over 9 cells is the source build's situation."""
    rng = np.random.default_rng(24)
    labels = rng.integers(0, 3, 60)
    res = markov_order_test(labels, n_states=3)
    assert res["warning"], "small-cell warning must fire"


# --------------------------------------------------------------------------- #
#  Event study end to end
# --------------------------------------------------------------------------- #
def test_event_panel_recovers_the_planted_variance_premium():
    """Consistency: with a long sample and no fat tail, the measured premium
    must converge on the planted one."""
    cfg = SyntheticConfig(n_days=16000, seed=101, variance_risk_premium=0.25,
                          tail_prob=0.0)
    p = SyntheticProvider(cfg)
    events = p.events()
    bars = p.bars(__import__("vrplab").BarRequest(
        cfg.symbol, events.index.min() - pd.Timedelta(days=10),
        events.index.max() + pd.Timedelta(days=10)))
    panel = build_event_panel(events, bars, exit_on="open")
    assert panel.n_usable > 0.95 * panel.n_input, panel.coverage()

    df = panel.frame
    planted = events.loc[df.index, "true_implied_move"].to_numpy()
    assert np.corrcoef(df["implied_event_move"], planted)[0, 1] > 0.97
    assert np.mean(np.abs(df["implied_event_move"] - planted)) < 0.01

    measured = (float(np.mean(df["implied_event_move"] ** 2))
                / float(np.mean(df["realised_event_move"] ** 2))) - 1.0
    assert measured == pytest.approx(0.25, abs=0.10), f"measured {measured:.3f}"


def test_variance_premium_estimate_is_unreliable_at_realistic_sample_sizes():
    """The reason `min track record length` is printed next to every result.

    With a fat left tail and the ~40-80 earnings events a single name actually
    has, the measured premium swings across seeds by more than the premium
    itself. Any conclusion drawn from one name's history is noise.
    """
    measured = []
    for seed in range(12):
        cfg = SyntheticConfig(n_days=3000, seed=seed, variance_risk_premium=0.25)
        p = SyntheticProvider(cfg)
        ev = p.events()
        bars = p.bars(__import__("vrplab").BarRequest(
            cfg.symbol, ev.index.min() - pd.Timedelta(days=10),
            ev.index.max() + pd.Timedelta(days=10)))
        df = build_event_panel(ev, bars, exit_on="open").frame
        measured.append(float(np.mean(df["implied_event_move"] ** 2))
                        / float(np.mean(df["realised_event_move"] ** 2)) - 1.0)
    spread = float(np.std(measured, ddof=1))
    assert spread > 0.15, (
        f"expected the single-name estimate to be wildly noisy, got sd={spread:.3f}")


def test_costs_can_flip_the_conclusion(provider):
    """The whole reason the cost model is explicit: the source builds' P&L is a
    mid-to-mid number, and mid-to-mid is not a fill."""
    events = provider.events()
    bars = provider.bars(__import__("vrplab").BarRequest(
        provider.config.symbol, events.index.min() - pd.Timedelta(days=400),
        events.index.max() + pd.Timedelta(days=10)))
    panel = build_event_panel(events, bars).frame

    free = premium_report(panel, costs=CostModel(half_spread_pct=0.0,
                                                 commission_per_contract=0.0))
    real = premium_report(panel, costs=CostModel(half_spread_pct=0.05,
                                                 commission_per_contract=0.65))
    free_mean = free["stats"].loc["mean net P&L per trade", "value"]
    real_mean = real["stats"].loc["mean net P&L per trade", "value"]
    assert free_mean > real_mean
    assert real["bootstrap_pnl"]["p_le_zero"] >= free["bootstrap_pnl"]["p_le_zero"]


def test_summary_reports_the_tail_not_just_the_mean(provider):
    events = provider.events()
    bars = provider.bars(__import__("vrplab").BarRequest(
        provider.config.symbol, events.index.min() - pd.Timedelta(days=400),
        events.index.max() + pd.Timedelta(days=10)))
    s = summarise_events(build_event_panel(events, bars).frame, n_boot=300)
    for metric in ("P(ratio > 1)", "P(ratio > 2)", "ratio 99th pct"):
        assert metric in s.index
    assert 0.0 <= s.loc["P(ratio > 1)", "value"] <= 1.0


def test_bmo_and_amc_anchor_different_sessions():
    """Half the earnings universe reports before the open. Getting the session
    wrong inverts the trade; the source build has no session field."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    bars = pd.DataFrame({"open": np.linspace(100, 109, 10),
                         "high": np.linspace(101, 110, 10),
                         "low": np.linspace(99, 108, 10),
                         "close": np.linspace(100.5, 109.5, 10),
                         "volume": 1.0}, index=idx)
    t1, t2 = 7 / 365, 35 / 365
    iv1 = np.sqrt((0.25**2 * t1 + 0.05**2) / t1)
    iv2 = np.sqrt((0.25**2 * t2 + 0.05**2) / t2)
    base = {"symbol": "X", "front_iv": iv1, "back_iv": iv2,
            "front_tenor": t1, "back_tenor": t2}
    ev_date = idx[5]

    amc = build_event_panel(pd.DataFrame([{**base, "session": "AMC"}],
                                         index=[ev_date]), bars).frame
    bmo = build_event_panel(pd.DataFrame([{**base, "session": "BMO"}],
                                         index=[ev_date]), bars).frame
    assert amc.iloc[0]["entry_date"] == ev_date
    assert amc.iloc[0]["exit_date"] == idx[6]
    assert bmo.iloc[0]["entry_date"] == idx[4]
    assert bmo.iloc[0]["exit_date"] == ev_date


def test_theta_is_charged_across_the_hold():
    """Repricing at an unchanged T, as both source builds do, omits theta."""
    kw = dict(symbol="X", event_date=pd.Timestamp("2024-01-10"),
              spot_entry=100.0, spot_exit=100.0, iv_entry=0.60, iv_exit=0.60,
              tenor_entry=7 / 365, costs=CostModel(half_spread_pct=0.0,
                                                   commission_per_contract=0.0))
    held = price_event_straddle(**kw, hold_days=1.0)
    frozen = price_event_straddle(**kw, hold_days=0.0)
    assert held.net_pnl > frozen.net_pnl      # seller earns the day of decay
    assert frozen.net_pnl == pytest.approx(0.0, abs=1e-9)


def test_multiplier_is_applied_exactly_once():
    t = price_event_straddle(
        symbol="X", event_date=pd.Timestamp("2024-01-10"), spot_entry=100.0,
        spot_exit=100.0, iv_entry=0.50, iv_exit=0.30, tenor_entry=7 / 365,
        hold_days=1.0, costs=CostModel(half_spread_pct=0.0, commission_per_contract=0.0))
    assert t.net_pnl == pytest.approx(t.net_pnl_per_share * 100, rel=1e-12)
    assert t.net_pnl_per_share < 10.0        # a per-share number, sanity


def test_no_fabrication_on_missing_data():
    """The source build substitutes pre_iv = VIX/100*1.5 when IV is missing.
    Here a missing input drops the row and is counted."""
    from vrplab.backtest.straddle import run_event_backtest
    panel = pd.DataFrame({
        "symbol": ["A", "B"], "spot_entry": [100.0, 100.0], "spot_exit": [101.0, 101.0],
        "iv_entry": [0.5, np.nan], "iv_exit": [0.3, 0.3], "tenor_entry": [0.02, 0.02],
    }, index=pd.to_datetime(["2024-01-10", "2024-02-10"]))
    out = run_event_backtest(panel)
    assert len(out) == 1
    assert out.attrs["dropped_rows"] == 1


# --------------------------------------------------------------------------- #
#  Live screening logic (no IB connection required)
# --------------------------------------------------------------------------- #
def test_screen_refuses_without_enough_history():
    from vrplab.live.signal import RiskLimits, screen_event
    t1, t2 = 7 / 365, 35 / 365
    iv1 = np.sqrt((0.30**2 * t1 + 0.08**2) / t1)
    iv2 = np.sqrt((0.30**2 * t2 + 0.08**2) / t2)
    sig = screen_event(symbol="X", asof=pd.Timestamp("2026-01-05"), spot=100.0,
                       front_iv=iv1, back_iv=iv2, front_tenor=t1, back_tenor=t2,
                       historical_moves=np.array([0.05, 0.06]),
                       limits=RiskLimits(min_events_in_history=8))
    assert sig.action == "STAND DOWN"
    assert any("historical events" in r for r in sig.reasons)


def test_screen_flags_expensive_vol_against_own_history():
    from vrplab.live.signal import RiskLimits, screen_event
    t1, t2 = 7 / 365, 35 / 365
    iv1 = np.sqrt((0.30**2 * t1 + 0.10**2) / t1)
    iv2 = np.sqrt((0.30**2 * t2 + 0.10**2) / t2)
    hist = np.full(20, 0.03)                       # this name never moves much
    sig = screen_event(symbol="X", asof=pd.Timestamp("2026-01-05"), spot=100.0,
                       front_iv=iv1, back_iv=iv2, front_tenor=t1, back_tenor=t2,
                       historical_moves=hist, spread_pct_of_mid=0.03,
                       limits=RiskLimits(account_equity=2_000_000.0))
    assert sig.action.startswith("SELL VOL")
    assert sig.implied_event_move == pytest.approx(0.10, abs=1e-6)


def test_screen_refuses_a_wide_chain():
    from vrplab.live.signal import screen_event
    t1, t2 = 7 / 365, 35 / 365
    iv1 = np.sqrt((0.30**2 * t1 + 0.10**2) / t1)
    iv2 = np.sqrt((0.30**2 * t2 + 0.10**2) / t2)
    sig = screen_event(symbol="X", asof=pd.Timestamp("2026-01-05"), spot=100.0,
                       front_iv=iv1, back_iv=iv2, front_tenor=t1, back_tenor=t2,
                       historical_moves=np.full(20, 0.03), spread_pct_of_mid=0.25)
    assert sig.action == "STAND DOWN"
    assert any("spread" in r for r in sig.reasons)


def test_size_shrinks_as_the_tail_grows():
    from vrplab.live.signal import RiskLimits, size_position
    lim_tight = RiskLimits(stress_move_sigmas=6.0)
    lim_loose = RiskLimits(stress_move_sigmas=3.0)
    n_tight, _ = size_position(100, 100, 7 / 365, 0.9, 0.09, lim_tight)
    n_loose, _ = size_position(100, 100, 7 / 365, 0.9, 0.09, lim_loose)
    assert n_tight <= n_loose


def test_study_runs_end_to_end_and_reports():
    from vrplab.research.study import StudyConfig, run_study
    p = SyntheticProvider(SyntheticConfig(n_days=1500, seed=5))
    res = run_study(p, p.events(), p.config.symbol,
                    StudyConfig(n_boot=200, regime_candidates=(1, 2), seed=1))
    text = res.report()
    assert "VERDICT" in text
    assert "Coverage" in text
    assert len(res.trades) > 10
