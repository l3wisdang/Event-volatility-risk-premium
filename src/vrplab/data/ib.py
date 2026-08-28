"""Interactive Brokers adapter, written the way the API actually wants.

Ten problems in the source builds this fixes, in the order they will bite you:

1. **The completion race.** They wait with ``while reqId not in historical_data``,
   which returns on the *first* bar callback while the reader thread is still
   appending. Here every request owns a ``threading.Event`` set by
   ``historicalDataEnd``, and the payload is handed over under a lock.
2. **Hard-coded ``reqId``.** They use 1, 2, 3 forever, so a late response from a
   cancelled request lands in the next request's bucket. Here IDs come from a
   monotonic counter and are never reused.
3. **Hard-coded ``clientId = 0``.** 0 is the master client ID with special TWS
   semantics, and a second app cannot connect. Configurable, default 11.
4. **Blocking the UI thread.** All waits here happen in the caller's worker
   thread; nothing in this file touches a GUI.
5. **The ``error()`` signature.** ``ibapi`` has changed it twice. Absorbing
   ``*args, **kwargs`` means a version bump does not raise ``TypeError`` inside
   the reader thread, where it would be invisible.
6. **Errors going to ``print``.** Every error is captured per-request and
   re-raised to the caller, so "no data" and "you lack a market data
   subscription" are distinguishable.
7. **Delayed market data.** They note error 10167 and then handle only tick
   types 1/2/4, so a delayed feed silently produces nothing. Here delayed tick
   types (66/67/68) are mapped explicitly.
8. **No pacing control.** IB throttles ``reqHistoricalData`` (60 requests per
   10 minutes, 6 identical per 2 seconds). A token bucket enforces it.
9. **Volatility units.** ``OPTION_IMPLIED_VOLATILITY`` arrives in whichever unit
   TWS is configured for. This adapter requires you to declare it and converts
   once, in one place.
10. **No persistence.** Every response is written to a parquet cache keyed on
    the full request, so a study is reproducible after the market moves on.

``ibapi`` is imported lazily so the rest of the package -- and its whole test
suite -- runs without TWS installed.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .base import BarRequest, VolUnits, to_annualised, validate_bars

log = logging.getLogger(__name__)

__all__ = ["IBConfig", "IBProvider", "IBError"]

# Status codes that are informational, not failures.
_BENIGN = {2104, 2106, 2107, 2108, 2158, 2119, 2176}
_DELAYED_NOTICE = 10167

# Live tick types -> canonical name; delayed equivalents map to the same names.
TICK_TYPES = {1: "bid", 2: "ask", 4: "last", 66: "bid", 67: "ask", 68: "last",
              6: "high", 7: "low", 9: "close", 14: "open"}


class IBError(RuntimeError):
    """An error IB returned for a specific request, surfaced to the caller."""

    def __init__(self, code: int, message: str, req_id: int | None = None):
        super().__init__(f"IB error {code} (reqId={req_id}): {message}")
        self.code, self.message, self.req_id = code, message, req_id


@dataclass
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 7497                    # 7497 paper TWS, 7496 live, 4002 gateway paper
    client_id: int = 11
    connect_timeout: float = 10.0
    request_timeout: float = 60.0
    # Declare what TWS is configured to send. Check
    # TWS > Settings > Volatility and Analytics. Getting this wrong is the
    # single most common silent error in retail IB code.
    vol_units: VolUnits = VolUnits.ANNUAL_DECIMAL
    market_data_type: int = 1           # 1 live, 2 frozen, 3 delayed, 4 delayed-frozen
    max_requests_per_10min: int = 55    # below IB's 60, deliberately
    cache_dir: str | None = ".vrplab_cache"


@dataclass
class _Pending:
    bars: list = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)
    error: IBError | None = None
    payload: Any = None


def _make_wrapper_class():
    """Build the EWrapper subclass at call time so ``ibapi`` stays optional."""
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper

    class _App(EWrapper, EClient):
        def __init__(self):
            EWrapper.__init__(self)
            EClient.__init__(self, self)
            self._lock = threading.RLock()
            self._pending: dict[int, _Pending] = {}
            self.connected_evt = threading.Event()
            self.next_order_id: int | None = None
            self.ticks: dict[int, dict] = {}

        # -- lifecycle ------------------------------------------------- #
        def connectAck(self):
            log.debug("connectAck")

        def nextValidId(self, orderId: int):
            self.next_order_id = orderId
            self.connected_evt.set()

        def connectionClosed(self):
            self.connected_evt.clear()
            with self._lock:
                for p in self._pending.values():
                    if not p.done.is_set():
                        p.error = IBError(-1, "connection closed before response")
                        p.done.set()
            log.warning("IB connection closed")

        # -- errors ---------------------------------------------------- #
        def error(self, *args, **kwargs):
            """Tolerant of every ibapi signature shipped to date.

            Old: (reqId, code, msg). Newer: (reqId, code, msg, advancedJson).
            Newest: (reqId, errorTime, code, msg, advancedJson).
            """
            nums = [a for a in args if isinstance(a, int)]
            strs = [a for a in args if isinstance(a, str)]
            req_id = nums[0] if nums else -1
            code = nums[-1] if len(nums) > 1 else -1
            # errorTime is a large epoch-ms int; the real code is the small one.
            candidates = [n for n in nums[1:] if abs(n) < 100000]
            if candidates:
                code = candidates[0]
            msg = strs[0] if strs else ""

            if code in _BENIGN:
                log.debug("IB status %s: %s", code, msg)
                return
            if code == _DELAYED_NOTICE:
                log.warning("delayed market data in use: %s", msg)
                return
            log.error("IB error %s (reqId=%s): %s", code, req_id, msg)
            with self._lock:
                p = self._pending.get(req_id)
                if p is not None and not p.done.is_set():
                    p.error = IBError(code, msg, req_id)
                    p.done.set()

        # -- historical bars ------------------------------------------- #
        def historicalData(self, reqId, bar):
            with self._lock:
                p = self._pending.get(reqId)
                if p is not None:
                    p.bars.append({
                        "date": bar.date, "open": bar.open, "high": bar.high,
                        "low": bar.low, "close": bar.close, "volume": float(bar.volume),
                    })

        def historicalDataEnd(self, reqId, start, end):
            with self._lock:
                p = self._pending.get(reqId)
            if p is not None:
                p.done.set()

        # -- contract details / option params -------------------------- #
        def contractDetails(self, reqId, contractDetails):
            with self._lock:
                p = self._pending.get(reqId)
                if p is not None:
                    p.bars.append(contractDetails)

        def contractDetailsEnd(self, reqId):
            with self._lock:
                p = self._pending.get(reqId)
            if p is not None:
                p.done.set()

        def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId,
                                              tradingClass, multiplier, expirations, strikes):
            with self._lock:
                p = self._pending.get(reqId)
                if p is not None:
                    p.bars.append({
                        "exchange": exchange, "trading_class": tradingClass,
                        "multiplier": multiplier,
                        "expirations": sorted(expirations), "strikes": sorted(strikes),
                    })

        def securityDefinitionOptionParameterEnd(self, reqId):
            with self._lock:
                p = self._pending.get(reqId)
            if p is not None:
                p.done.set()

        # -- streaming ticks ------------------------------------------- #
        def tickPrice(self, reqId, tickType, price, attrib):
            if price is None or price <= 0:
                return
            name = TICK_TYPES.get(tickType)
            if name is None:
                return
            with self._lock:
                self.ticks.setdefault(reqId, {})[name] = price
                self.ticks[reqId]["asof"] = pd.Timestamp.now("UTC")

        def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol,
                                  delta, optPrice, pvDividend, gamma, vega,
                                  theta, undPrice, *args):
            """The correct source of per-contract implied vol -- an actual
            option's IV at an actual strike and expiry, rather than IB's
            aggregate underlying series of unstated tenor."""
            with self._lock:
                d = self.ticks.setdefault(reqId, {})
                if impliedVol is not None and impliedVol > 0:
                    d["iv"] = impliedVol
                for key, val in (("delta", delta), ("gamma", gamma), ("vega", vega),
                                 ("theta", theta), ("opt_price", optPrice),
                                 ("und_price", undPrice)):
                    if val is not None:
                        d[key] = val
                d["asof"] = pd.Timestamp.now("UTC")

        # -- request plumbing ------------------------------------------ #
        def register(self, req_id: int) -> _Pending:
            p = _Pending()
            with self._lock:
                self._pending[req_id] = p
            return p

        def release(self, req_id: int):
            with self._lock:
                self._pending.pop(req_id, None)

    return _App


class _RateLimiter:
    """Token bucket for IB's historical-data pacing rules."""

    def __init__(self, max_calls: int, window_s: float = 600.0):
        self.max_calls, self.window_s = max_calls, window_s
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.window_s:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_for = self.window_s - (now - self._calls[0]) + 0.05
            log.info("IB pacing: sleeping %.1fs", sleep_for)
            time.sleep(min(sleep_for, 30.0))


class IBProvider:
    """Blocking, thread-safe IB market data provider.

    Use it from a worker thread or a script, never from a GUI callback.  The
    reader thread is owned internally and is a daemon.
    """

    name = "ib"

    def __init__(self, config: IBConfig | None = None):
        self.config = config or IBConfig()
        self._app = None
        self._thread: threading.Thread | None = None
        self._ids = itertools.count(1000)
        self._limiter = _RateLimiter(self.config.max_requests_per_10min)
        self._cache = None
        if self.config.cache_dir:
            from .cache import ParquetCache
            self._cache = ParquetCache(self.config.cache_dir)

    # ------------------------------------------------------------------ #
    def connect(self) -> "IBProvider":
        if self._app is not None and self._app.connected_evt.is_set():
            return self
        App = _make_wrapper_class()
        self._app = App()
        c = self.config
        self._app.connect(c.host, c.port, c.client_id)
        self._thread = threading.Thread(target=self._app.run, daemon=True,
                                        name="ib-reader")
        self._thread.start()
        if not self._app.connected_evt.wait(timeout=c.connect_timeout):
            self.disconnect()
            raise IBError(-1, f"no nextValidId within {c.connect_timeout}s; is TWS "
                              f"running with API enabled on port {c.port}?")
        self._app.reqMarketDataType(c.market_data_type)
        log.info("connected to IB at %s:%s as client %s", c.host, c.port, c.client_id)
        return self

    def disconnect(self):
        if self._app is not None:
            try:
                self._app.disconnect()
            except Exception:                       # noqa: BLE001 - teardown
                pass
            self._app.connected_evt.clear()
        self._app = None
        self._thread = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._app is not None and self._app.connected_evt.is_set()

    def _require(self):
        if not self.is_connected:
            raise IBError(-1, "not connected; call connect() first")

    # ------------------------------------------------------------------ #
    def _contract(self, req: BarRequest):
        from ibapi.contract import Contract
        c = Contract()
        c.symbol = req.symbol.upper()
        c.secType = req.sec_type
        c.exchange = req.exchange
        c.currency = req.currency
        return c

    @staticmethod
    def _duration(start: pd.Timestamp, end: pd.Timestamp) -> str:
        days = max(int((pd.Timestamp(end) - pd.Timestamp(start)).days) + 1, 1)
        if days <= 60:
            return f"{days} D"
        years = days / 365.0
        return f"{min(int(years) + 1, 30)} Y"

    def bars(self, request: BarRequest, use_cache: bool = True) -> pd.DataFrame:
        """Historical bars. Cached on disk, keyed by the full request."""
        if use_cache and self._cache is not None:
            hit = self._cache.get(request)
            if hit is not None:
                return hit

        self._require()
        self._limiter.acquire()
        req_id = next(self._ids)
        pending = self._app.register(req_id)
        contract = self._contract(request)
        # IB requires a timezone on endDateTime in recent builds; omitting it
        # raises 10314, which the source builds surface as "no data".
        end_str = pd.Timestamp(request.end).strftime("%Y%m%d %H:%M:%S") + " US/Eastern"
        try:
            self._app.reqHistoricalData(
                req_id, contract, end_str,
                self._duration(request.start, request.end),
                request.bar_size, request.what,
                1 if request.use_rth else 0, 1, False, [],
            )
            if not pending.done.wait(timeout=self.config.request_timeout):
                self._app.cancelHistoricalData(req_id)
                raise IBError(-1, f"historical data timed out after "
                                  f"{self.config.request_timeout}s", req_id)
            if pending.error is not None:
                raise pending.error
            rows = list(pending.bars)
        finally:
            self._app.release(req_id)

        if not rows:
            raise IBError(-1, f"no bars returned for {request.symbol} "
                              f"({request.what}, {request.bar_size})", req_id)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"].astype(str).str.split(" ").str[0],
                                    format="mixed")
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if request.what == "OPTION_IMPLIED_VOLATILITY":
            # The one and only unit conversion in the codebase.
            for col in ("open", "high", "low", "close"):
                df[col] = to_annualised(df[col].to_numpy(), self.config.vol_units)
            df.attrs["vol_units"] = "annual_decimal"
        else:
            df = validate_bars(df, request)

        df = df.loc[pd.Timestamp(request.start):pd.Timestamp(request.end)]
        df.attrs["provider"] = self.name
        # An ISO string, not a Timestamp: DataFrame.attrs is serialised to JSON
        # when the frame is written to parquet, and a Timestamp there silently
        # drops the whole attrs dict on the way to the cache.
        df.attrs["retrieved_at"] = pd.Timestamp.now("UTC").isoformat()
        if self._cache is not None:
            self._cache.put(request, df)
        return df

    # ------------------------------------------------------------------ #
    def option_params(self, symbol: str, sec_type: str = "STK",
                      exchange: str = "SMART", currency: str = "USD") -> pd.DataFrame:
        """Expiries and strikes via ``reqSecDefOptParams``.

        This is how you find the two expiries that bracket an earnings date --
        the input the term-structure decomposition needs, and the thing neither
        source build ever queries.
        """
        self._require()
        con_id = self._con_id(symbol, sec_type, exchange, currency)
        req_id = next(self._ids)
        pending = self._app.register(req_id)
        try:
            self._app.reqSecDefOptParams(req_id, symbol.upper(), "", sec_type, con_id)
            if not pending.done.wait(timeout=self.config.request_timeout):
                raise IBError(-1, "reqSecDefOptParams timed out", req_id)
            if pending.error is not None:
                raise pending.error
            return pd.DataFrame(list(pending.bars))
        finally:
            self._app.release(req_id)

    def _con_id(self, symbol: str, sec_type: str, exchange: str, currency: str) -> int:
        self._require()
        req = BarRequest(symbol, pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02"),
                         sec_type=sec_type, exchange=exchange, currency=currency)
        contract = self._contract(req)
        req_id = next(self._ids)
        pending = self._app.register(req_id)
        try:
            self._app.reqContractDetails(req_id, contract)
            if not pending.done.wait(timeout=self.config.request_timeout):
                raise IBError(-1, "reqContractDetails timed out", req_id)
            if pending.error is not None:
                raise pending.error
            if not pending.bars:
                raise IBError(-1, f"no contract found for {symbol}", req_id)
            return int(pending.bars[0].contract.conId)
        finally:
            self._app.release(req_id)

    def option_chain(self, symbol: str, asof: pd.Timestamp) -> pd.DataFrame:
        """Thin wrapper for protocol compatibility; see ``option_params``."""
        return self.option_params(symbol)

    def option_snapshot(self, symbol: str, expiry: str, strike: float, right: str,
                        exchange: str = "SMART", currency: str = "USD",
                        wait: float = 5.0) -> dict:
        """Live quote and model IV for one option contract.

        Generic tick list ``106`` requests option implied volatility, delivered
        through ``tickOptionComputation`` -- an IV that belongs to a specific
        strike and expiry, which is what a straddle price actually needs.
        """
        from ibapi.contract import Contract
        self._require()
        c = Contract()
        c.symbol, c.secType, c.exchange, c.currency = symbol.upper(), "OPT", exchange, currency
        c.lastTradeDateOrContractMonth = expiry
        c.strike = float(strike)
        c.right = right.upper()[0]
        c.multiplier = "100"

        req_id = next(self._ids)
        pending = self._app.register(req_id)
        try:
            self._app.reqMktData(req_id, c, "106", False, False, [])
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                # An error for this reqId -- a bad contract, a missing
                # entitlement -- sets the pending event. Without this check the
                # loop just spins out the full timeout and reports nothing,
                # which would make the "errors are raised, not printed"
                # guarantee true of bars() but not of this path.
                if pending.done.is_set() and pending.error is not None:
                    self._app.cancelMktData(req_id)
                    raise pending.error
                with self._app._lock:
                    snap = dict(self._app.ticks.get(req_id, {}))
                if {"bid", "ask"} <= snap.keys():
                    break
                time.sleep(0.05)
            self._app.cancelMktData(req_id)
            if pending.error is not None:
                raise pending.error
            with self._app._lock:
                snap = dict(self._app.ticks.pop(req_id, {}))
            snap.update({"symbol": symbol.upper(), "expiry": expiry,
                         "strike": float(strike), "right": right.upper()[0]})
            return snap
        finally:
            self._app.release(req_id)
