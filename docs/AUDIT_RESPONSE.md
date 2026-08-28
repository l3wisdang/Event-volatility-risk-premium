# Response to external audit

Every finding accepted. Nothing was disputed. Below: what was verified, what
changed, and which test now guards it.

## Confirmed before fixing

The central numerical claim reproduces exactly. For `S=100, σ=60%, T=7/365`:

```
straddle (bs_price)          6.627794
straddle (closed form)       6.627794    2S[2N(x/2) − 1], match to 1e-9
true one-sigma move          8.309097%
straddle / S                 6.627794%
x · sqrt(2/π)                6.629700%   <- equals straddle/S
old function returned        4.153354%   <- ratio to truth: 2.0006
sqrt(π/2) · straddle/S       8.306708%   <- correct
```

The auditor's reading of the docstring was also right: `"√(π/2)/2 × 2 = 1.2533/2"`
is self-contradicting algebra, and the desk shorthand is the correct
leading-order expected absolute move, not a fudge.

## Findings and fixes

| # | Finding | Fix | Guarded by |
|---|---|---|---|
| 1 | `atm_straddle_to_implied_move` out by 2× | constant → `sqrt(π/2)`; added `atm_straddle_to_expected_abs_move` (= straddle/S) and named module constants `SQRT_2_OVER_PI` / `SQRT_PI_OVER_2`; derivation rewritten | `test_straddle_to_one_sigma_move_matches_the_closed_form` (3 vol/tenor pairs vs closed form), `test_straddle_over_spot_is_the_expected_absolute_move`, `test_the_two_move_conversions_differ_by_the_right_constant` |
| 2 | Conditional table aligned on event date → look-ahead for BMO reporters | `StraddleTrade` now carries `entry_date`, threaded through `run_event_backtest`; `_conditional_performance` aligns on it and **raises** if the column is absent | `test_conditional_performance_uses_the_entry_session` (BMO fixture where the regime flips on the print), `test_conditional_performance_refuses_a_frame_without_entry_date`, `test_backtest_carries_entry_date_through_from_the_panel` |
| 3 | Full-sample HMM parameter look-ahead, undocumented | added `fit_regime(..., fit_through=...)` which freezes parameters on a training window and filters forward; the report states which mode produced the table; added to README limitations | `test_fit_through_freezes_parameters_on_the_training_window`, `test_fit_through_too_early_declines_rather_than_fitting_noise`, `test_report_declares_whether_regime_parameters_saw_the_future` |
| 4 | Honesty note stranded in `DataFrame.attrs`, dropped by `to_string` | new `_conditional_block()` prints the table plus three numbered caveats inline, including the smallest cell size | `test_conditional_caveats_appear_in_the_printed_report` |
| 5 | Regime split is uncorrected multiple testing | caveat 2 states the k-way split is k more tests and gives the Bonferroni reading | same test |
| 6 | Markov-order test circular on filtered labels | report now **leads** with `BIC(k=1) − BIC(k*)` on the observations as primary evidence; the LR test is demoted to secondary and labelled `[confounded: the filter's own sticky prior induces label dependence]`; also wired in the previously-unused `homogeneity_test` | `test_report_declares_whether_regime_parameters_saw_the_future` asserts `"confounded"` appears |
| 7 | Full day of theta charged on a 17.5-hour hold | `HOLD_DAYS_BY_EXIT = {"open": 17.5/24, "close": 1.0}`; `StudyConfig.hold_days=None` derives it from `exit_on` | `test_overnight_hold_charges_less_than_a_full_day_of_theta`, `test_study_derives_hold_days_from_exit_on` |
| 8 | `move_ratio` meant two different things | trades column renamed `move_vs_implied_sd` with a docstring pointing at the panel's `move_ratio` | `test_panel_and_trade_ratio_columns_are_named_distinctly` (also checks the join no longer collides) |
| 9 | `size_position` used a `0.6·iv` guess with `diffusive_vol` one call up | added `post_iv` parameter; `screen_event` passes `term.diffusive_vol`; the heuristic is now a labelled fallback | `test_size_position_uses_the_supplied_post_event_vol`, `test_screen_event_passes_the_diffusive_vol_into_sizing` |
| 10 | `option_snapshot` never checked the pending error slot | polls the error slot each iteration and re-raises, so README fix #6 is true of this path too | — (needs a live chain; noted as unverified) |
| 11 | `SyntheticConfig.spread_pct` documented as ground truth, never used | removed | `test_synthetic_config_has_no_undelivered_ground_truth_field` |
| 12 | `build_event_panel` docstring listed `quote_date` as required | docstring corrected; states that entry/exit come from the bar index and session flag only | — |
| 13 | `bootstrap_ci` docstring said `p_gt_zero` | corrected to `p_le_zero` | `test_bootstrap_ci_docstring_names_the_key_it_returns` |
| 14 | `stationary_bootstrap` / `homogeneity_test` built but unused | `homogeneity_test` wired into `fit_regime` and printed; `stationary_bootstrap` is reachable via `bootstrap_ci(method="stationary")` and documented as such | — |

## One place the auditor's test design, not the finding, was wrong

The suggested check that `size_position` responds to `post_iv` fails at the
default 4σ stress — the straddle is ~40% in the money, almost all intrinsic, and
the post-event vol barely moves it. That is a real property of the sizing rule
rather than a bug, so the regression test asserts on the stress **loss** at a
moderate 1.5σ stress, where vega still matters, and additionally checks that a
lower post-event vol permits more contracts.

## Headline numbers after the theta fix

Charging 0.73 days instead of 1.0 on an overnight hold lowers the seller's P&L
slightly across the board. The conclusion strengthens rather than weakens:

| | before | after |
|---|---|---|
| mid-to-mid | +$270, p=0.03, needs 137 events | +$262, p=0.038, needs 145 |
| 2% half-spread | +$183, p=0.10 | +$175, p=0.105, now **NOT ESTABLISHED** |
| 5% half-spread | +$57, p=0.34 | +$48, p=0.359, deflated Sharpe 0.02 |

The measured variance premium is unchanged at +31.8% against a planted 25%,
since it does not depend on the cost or theta model.

## Test count

41 → 63 research tests (22 audit regressions added), plus 24 IB adapter tests.
**87 total, all passing.**
