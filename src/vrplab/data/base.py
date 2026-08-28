"""Data layer contracts.

Two ideas do all the work here.

**1. Volatility units are declared, never guessed.**  Interactive Brokers can
return the ``OPTION_IMPLIED_VOLATILITY`` series in daily or annualised units
depending on a *GUI checkbox* in TWS (Volatility and Analytics).  Both Quant
Guild builds multiply by sqrt(252) regardless, one of them behind a heuristic
(``if raw_iv.max() > 5: divide by 100``) that turns an already-annualised 0.44
into 698%.  Here a provider must state which convention it is handing over, and
:class:`VolUnits` performs the single conversion.

**2. Everything is a point-in-time snapshot.**  Bars carry the timestamp at
which they were retrieved, so a research result can be reproduced later against
the same data rather than against a silently revised vendor series.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = ["VolUnits", "BarRequest", "OptionQuote", "MarketDataProvider",
           "to_annualised", "REQUIRED_BAR_COLUMNS", "validate_bars"]

REQUIRED_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


class VolUnits(enum.Enum):
    """The convention a vendor volatility series arrives in."""

    ANNUAL_DECIMAL = "annual_decimal"    # 0.35 == 35% annualised
    ANNUAL_PERCENT = "annual_percent"    # 35.0 == 35% annualised
    DAILY_DECIMAL = "daily_decimal"      # 0.022 == 2.2% per day
    DAILY_PERCENT = "daily_percent"      # 2.2  == 2.2% per day


def to_annualised(values, units: VolUnits, periods_per_year: int = 252):
    """Convert a vendor volatility series to annualised decimals. Exactly one
    place in the codebase does this."""
    v = np.asarray(values, dtype=float)
    scale = np.sqrt(periods_per_year)
    if units is VolUnits.ANNUAL_DECIMAL:
        return v
    if units is VolUnits.ANNUAL_PERCENT:
        return v / 100.0
    if units is VolUnits.DAILY_DECIMAL:
        return v * scale
    if units is VolUnits.DAILY_PERCENT:
        return v / 100.0 * scale
    raise ValueError(f"unhandled units {units!r}")


@dataclass(frozen=True)
class BarRequest:
    """A fully specified historical bar request.

    ``what`` mirrors the IB vocabulary (``TRADES``, ``MIDPOINT``,
    ``OPTION_IMPLIED_VOLATILITY``, ``HISTORICAL_VOLATILITY``) so that a request
    is portable across providers and, importantly, so that a cache key is
    unambiguous.
    """

    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    bar_size: str = "1 day"
    what: str = "TRADES"
    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    use_rth: bool = True

    def cache_key(self) -> str:
        parts = [self.symbol, self.sec_type, self.exchange, self.currency,
                 self.bar_size.replace(" ", ""), self.what,
                 pd.Timestamp(self.start).strftime("%Y%m%d"),
                 pd.Timestamp(self.end).strftime("%Y%m%d"),
                 "rth" if self.use_rth else "all"]
        return "_".join(parts)


@dataclass(frozen=True)
class OptionQuote:
    """One option quote, with everything needed to reprice it and nothing
    inferred."""

    symbol: str
    expiry: pd.Timestamp
    strike: float
    right: str                     # "C" or "P"
    bid: float
    ask: float
    underlying: float
    asof: pd.Timestamp
    iv: float = np.nan             # annualised decimal
    multiplier: int = 100

    @property
    def mid(self) -> float:
        if not (np.isfinite(self.bid) and np.isfinite(self.ask)):
            return np.nan
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return float(self.ask - self.bid)

    @property
    def spread_pct_of_mid(self) -> float:
        m = self.mid
        return float(self.spread / m) if m and np.isfinite(m) and m > 0 else np.nan


@runtime_checkable
class MarketDataProvider(Protocol):
    """Anything that can serve bars to the research layer.

    Deliberately tiny.  The IB adapter, the free-data adapter and the synthetic
    generator all satisfy it, so a study written against this protocol runs
    unchanged on simulated data with known ground truth and on live IB data.
    """

    name: str

    def bars(self, request: BarRequest) -> pd.DataFrame: ...

    def option_chain(self, symbol: str, asof: pd.Timestamp) -> pd.DataFrame: ...


@dataclass
class ProviderMeta:
    """Provenance attached to every frame the layer returns."""

    provider: str
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    vol_units: VolUnits | None = None
    notes: str = ""


def validate_bars(df: pd.DataFrame, request: BarRequest) -> pd.DataFrame:
    """Fail loudly on a malformed bar frame.

    Checks the index is a sorted, unique DatetimeIndex, that OHLC is internally
    consistent (``low <= open,close <= high``), and that no price is
    non-positive.  A provider that silently returns 3 bars for a 2-year request
    is a bug you want to find in the data layer, not in a Sharpe ratio.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("bars must be indexed by a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("bar index is not sorted")
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()[:5]
        raise ValueError(f"duplicate bar timestamps, e.g. {list(dupes)}")
    missing = set(REQUIRED_BAR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)} for {request.symbol}")
    px = df[["open", "high", "low", "close"]]
    if (px <= 0).any().any():
        raise ValueError(f"non-positive prices in {request.symbol}")
    bad = (df["low"] > df[["open", "close"]].min(axis=1)) | (
        df["high"] < df[["open", "close"]].max(axis=1))
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} bars violate low <= open,close <= high "
            f"(first: {df.index[bad][0]})"
        )
    return df
