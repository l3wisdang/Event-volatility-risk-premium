"""End-to-end demonstration on data whose answer we already know.

Run this first.  It plants a 25% event variance risk premium, then asks the
pipeline to find it, then charges a realistic spread and asks again.  The
difference between those two answers is the entire distance between a tutorial
dashboard and a research result.

    python examples/run_synthetic_study.py
"""

from __future__ import annotations

import numpy as np

from vrplab.backtest.straddle import CostModel
from vrplab.data.synthetic import SyntheticConfig, SyntheticProvider
from vrplab.data.base import BarRequest
from vrplab.research.study import StudyConfig, fit_regime, run_study


def main() -> None:
    cfg = SyntheticConfig(n_days=5200, seed=42, variance_risk_premium=0.25)
    provider = SyntheticProvider(cfg)
    events = provider.events()

    print(f"planted variance risk premium : {cfg.variance_risk_premium:.0%}")
    print(f"events generated              : {len(events)}")
    print(f"planted mean implied move     : {provider.truth['mean_implied_move']:.4f}")
    print(f"planted mean |realised| move  : {provider.truth['mean_abs_realised']:.4f}")

    # Fit the regime overlay ONCE. Refitting it per scenario would let the EM
    # restarts, rather than the cost assumption, move the conditional numbers.
    bars = provider.bars(BarRequest(cfg.symbol, events.index.min(), events.index.max()))
    regime = fit_regime(bars, StudyConfig(seed=3))

    for label, costs, trials in [
        ("A. mid-to-mid, no costs (how the tutorial builds measure it)",
         CostModel(half_spread_pct=0.0, commission_per_contract=0.0), 1),
        ("B. 2% half-spread, $0.65 commission (a good retail fill)",
         CostModel(half_spread_pct=0.02, commission_per_contract=0.65), 1),
        ("C. 5% half-spread (an ATM straddle into an actual print)",
         CostModel(half_spread_pct=0.05, commission_per_contract=0.65), 40),
    ]:
        print("\n\n" + "#" * 78)
        print("#", label)
        print("#" * 78)
        res = run_study(
            provider, events, cfg.symbol,
            StudyConfig(costs=costs, n_trials=trials, n_boot=1500, seed=3),
            regime=regime,
        )
        print(res.report())


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
