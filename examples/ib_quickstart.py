"""Interactive Brokers quickstart: connect, pull, screen.

Prerequisites
-------------
1. TWS or IB Gateway running and logged in (paper account is fine).
2. TWS > Settings > API > Settings: tick "Enable ActiveX and Socket Clients",
   note the Socket port (7497 paper TWS, 7496 live, 4002 Gateway paper).
3. TWS > Settings > Volatility and Analytics: note whether volatility is shown
   in DAILY or ANNUAL units, and set ``vol_units`` below to match. Guessing this
   is how a dashboard ends up reporting an implied vol of 698%.
4. ``pip install -e ".[ib]"``

Nothing here places an order. The last step produces a decision packet with an
explicit stand-down list, which is where a human belongs.

    python examples/ib_quickstart.py NVDA 2026-02-25
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from vrplab.data.base import BarRequest, VolUnits
from vrplab.data.ib import IBConfig, IBError, IBProvider
from vrplab.live.signal import RiskLimits, screen_from_ib
from vrplab.vol.realised import realised_vol


def main(symbol: str, earnings_date: str) -> None:
    cfg = IBConfig(
        host="127.0.0.1",
        port=7497,
        client_id=11,
        # >>> SET THIS TO MATCH YOUR TWS <<<
        vol_units=VolUnits.ANNUAL_DECIMAL,
        market_data_type=3,        # 3 = delayed; use 1 once you have subscriptions
        cache_dir=".vrplab_cache",
    )

    try:
        with IBProvider(cfg) as ib:
            # ---- 1. history -------------------------------------------- #
            end = pd.Timestamp.today().normalize()
            bars = ib.bars(BarRequest(symbol, end - pd.Timedelta(days=1500), end))
            print(f"{symbol}: {len(bars)} daily bars "
                  f"{bars.index.min():%Y-%m-%d} to {bars.index.max():%Y-%m-%d}")

            rv = realised_vol(bars, window=21, method="yang_zhang")
            print(f"  21-day Yang-Zhang realised vol: {rv.iloc[-1]:.1%}")

            # ---- 2. IB's own implied vol series, correctly scaled ------- #
            try:
                iv = ib.bars(BarRequest(symbol, end - pd.Timedelta(days=750), end,
                                        what="OPTION_IMPLIED_VOLATILITY"))
                print(f"  IB implied vol series (annualised): "
                      f"{iv['close'].iloc[-1]:.1%}  "
                      f"[units declared as {cfg.vol_units.value}]")
                print("  sanity check: if this is not in the 10%-150% range for a "
                      "large-cap, your vol_units setting is wrong.")
            except IBError as exc:
                print(f"  implied vol series unavailable: {exc}")

            # ---- 3. option expiries around the announcement ------------- #
            params = ib.option_params(symbol)
            print(f"  option parameter rows: {len(params)}")

            # ---- 4. screen the event ------------------------------------ #
            # Historical event moves would normally come from
            # build_event_panel over this name's earnings history. Until you
            # have that calendar, the screen correctly refuses to act.
            history = np.array([])
            sig = screen_from_ib(ib, symbol, earnings_date, history,
                                 limits=RiskLimits(account_equity=25_000))
            print("\n" + str(sig))

    except IBError as exc:
        print(f"\nIB error: {exc}")
        print("Checklist: TWS running? API enabled? Correct port? "
              "clientId not already in use? Market data subscription for "
              f"{symbol}?")
    except ImportError:
        print("ibapi is not installed. Run: pip install -e \".[ib]\"")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    date = sys.argv[2] if len(sys.argv) > 2 else str(pd.Timestamp.today().date())
    main(sym, date)
