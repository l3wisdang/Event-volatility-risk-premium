"""Integration tests for the Interactive Brokers adapter.

These run against the **real** ``ibapi`` package, with only the TCP socket
replaced.  Everything above the socket is exercised for real: the EWrapper
subclass is built from the actual ``EWrapper``/``EClient`` base classes, the
callbacks IB would invoke are invoked with the arguments IB would pass, from a
separate thread, at realistic timing.

That boundary is chosen deliberately.  The socket and wire protocol are IB's
code and are not where the bugs live.  The bugs live in the completion
handshake, the request-ID bookkeeping, the error signature, the threading and
the unit conversion -- all of which are covered here.

Skipped automatically if ``ibapi`` is not installed, so the main suite still
runs on a machine without it.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ibapi", reason="ibapi not installed; pip install -e '.[ib]'")

from vrplab.data.base import BarRequest, VolUnits          # noqa: E402
from vrplab.data.ib import IBConfig, IBError, IBProvider   # noqa: E402


class _Bar:
    """Duck-typed stand-in for ``ibapi.common.BarData``."""

    def __init__(self, date, o, h, l, c, v):
        self.date, self.open, self.high, self.low, self.close, self.volume = date, o, h, l, c, v


class FakeTWS:
    """Drives the adapter's callbacks the way a real TWS would.

    ``delay_between_bars`` exists to make the completion race observable: with a
    naive ``while reqId not in historical_data`` wait, the caller returns after
    the first bar and sees a truncated series.  The adapter must wait for
    ``historicalDataEnd``.
    """

    def __init__(self, app, n_bars=40, delay_between_bars=0.0, iv_mode=False,
                 error_code=None, error_msg="", start_price=100.0):
        self.app = app
        self.n_bars = n_bars
        self.delay = delay_between_bars
        self.iv_mode = iv_mode
        self.error_code = error_code
        self.error_msg = error_msg
        self.start_price = start_price
        self.seen_req_ids: list[int] = []

    def serve_historical(self, req_id: int):
        self.seen_req_ids.append(req_id)
        if self.error_code is not None:
            self.app.error(req_id, self.error_code, self.error_msg)
            return
        base = pd.Timestamp("2024-01-02")
        for i in range(self.n_bars):
            d = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
            if self.iv_mode:
                # A DAILY implied vol, as TWS sends when its Volatility and
                # Analytics unit is set to daily. 0.35 annualised / sqrt(252).
                v = 0.35 / np.sqrt(252)
                self.app.historicalData(req_id, _Bar(d, v, v, v, v, 0))
            else:
                p = self.start_price + i * 0.5
                self.app.historicalData(req_id, _Bar(d, p, p + 1.0, p - 1.0, p + 0.25, 1_000))
            if self.delay:
                time.sleep(self.delay)
        self.app.historicalDataEnd(req_id, "", "")


def make_provider(monkeypatch, **fake_kw):
    """Build a connected IBProvider whose socket is replaced by FakeTWS."""
    from ibapi.client import EClient

    holder: dict = {}

    def fake_connect(self, host, port, clientId):
        holder["app"] = self
        holder["tws"] = FakeTWS(self, **fake_kw)
        # TWS answers a successful connection with nextValidId.
        threading.Timer(0.01, lambda: self.nextValidId(1)).start()

    def fake_run(self):
        while True:
            time.sleep(0.05)

    def fake_req_hist(self, reqId, contract, endDateTime, durationStr, barSizeSetting,
                      whatToShow, useRTH, formatDate, keepUpToDate, chartOptions):
        holder["last_request"] = dict(
            reqId=reqId, endDateTime=endDateTime, durationStr=durationStr,
            barSize=barSizeSetting, whatToShow=whatToShow, useRTH=useRTH,
            symbol=contract.symbol, secType=contract.secType,
        )
        tws = holder["tws"]
        tws.iv_mode = whatToShow == "OPTION_IMPLIED_VOLATILITY"
        threading.Thread(target=tws.serve_historical, args=(reqId,), daemon=True).start()

    monkeypatch.setattr(EClient, "connect", fake_connect, raising=True)
    monkeypatch.setattr(EClient, "run", fake_run, raising=True)
    monkeypatch.setattr(EClient, "reqHistoricalData", fake_req_hist, raising=True)
    monkeypatch.setattr(EClient, "cancelHistoricalData", lambda self, reqId: None, raising=True)
    monkeypatch.setattr(EClient, "reqMarketDataType", lambda self, t: None, raising=True)
    monkeypatch.setattr(EClient, "disconnect", lambda self: None, raising=True)
    return holder


# --------------------------------------------------------------------------- #
def test_wrapper_class_builds_against_the_real_ibapi():
    """The subclass must actually inherit from IB's classes and override real
    callback names -- a typo in a callback name is silent, and is exactly the
    bug that left `historicalDataEnd` unimplemented in the source build."""
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper

    from vrplab.data.ib import _make_wrapper_class

    App = _make_wrapper_class()
    assert issubclass(App, EWrapper) and issubclass(App, EClient)
    for name in ("error", "nextValidId", "historicalData", "historicalDataEnd",
                 "connectionClosed", "tickPrice", "tickOptionComputation",
                 "contractDetails", "contractDetailsEnd",
                 "securityDefinitionOptionParameter",
                 "securityDefinitionOptionParameterEnd"):
        assert hasattr(EWrapper, name), f"{name} is not a real EWrapper callback"
        assert getattr(App, name) is not getattr(EWrapper, name), f"{name} not overridden"


def test_connect_waits_for_next_valid_id(monkeypatch):
    make_provider(monkeypatch)
    ib = IBProvider(IBConfig(cache_dir=None, connect_timeout=2.0))
    assert not ib.is_connected
    ib.connect()
    assert ib.is_connected
    ib.disconnect()
    assert not ib.is_connected


def test_connect_times_out_cleanly_when_tws_never_answers(monkeypatch):
    from ibapi.client import EClient
    monkeypatch.setattr(EClient, "connect", lambda self, h, p, c: None, raising=True)
    monkeypatch.setattr(EClient, "run", lambda self: time.sleep(5), raising=True)
    monkeypatch.setattr(EClient, "disconnect", lambda self: None, raising=True)
    ib = IBProvider(IBConfig(cache_dir=None, connect_timeout=0.3))
    with pytest.raises(IBError, match="nextValidId"):
        ib.connect()
    assert not ib.is_connected


def test_bars_waits_for_historical_data_end_not_the_first_bar(monkeypatch):
    """The completion race, reproduced.

    Bars arrive 5ms apart. A wait keyed on 'has any data arrived' would return
    after roughly one bar; the adapter must return all 40.
    """
    make_provider(monkeypatch, n_bars=40, delay_between_bars=0.005)
    with IBProvider(IBConfig(cache_dir=None)) as ib:
        df = ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"),
                                pd.Timestamp("2024-03-01")))
    assert len(df) == 40, f"got {len(df)} bars; completion handshake is racing"
    assert df.index.is_monotonic_increasing
    assert list(df.columns[:5]) == ["open", "high", "low", "close", "volume"]


def test_request_ids_are_never_reused(monkeypatch):
    holder = make_provider(monkeypatch, n_bars=5)
    with IBProvider(IBConfig(cache_dir=None)) as ib:
        for _ in range(4):
            ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"),
                               pd.Timestamp("2024-03-01")))
    seen = holder["tws"].seen_req_ids
    assert len(seen) == 4
    assert len(set(seen)) == 4, f"reqIds reused: {seen}"


def test_request_carries_a_timezone_on_end_datetime(monkeypatch):
    """Recent TWS builds reject a naive endDateTime with error 10314, which the
    source builds surface indistinguishably as 'no data'."""
    holder = make_provider(monkeypatch, n_bars=5)
    with IBProvider(IBConfig(cache_dir=None)) as ib:
        ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")))
    assert holder["last_request"]["endDateTime"].endswith("US/Eastern")


def test_ib_error_is_raised_to_the_caller_not_printed(monkeypatch):
    """354 is 'requested market data is not subscribed'. The caller must be able
    to tell that apart from an empty result."""
    make_provider(monkeypatch, error_code=354, error_msg="Requested market data is not subscribed.")
    with IBProvider(IBConfig(cache_dir=None, request_timeout=3.0)) as ib:
        with pytest.raises(IBError) as exc:
            ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")))
    assert exc.value.code == 354
    assert "not subscribed" in exc.value.message


def test_benign_status_codes_do_not_abort_a_request(monkeypatch):
    """2104 'market data farm connection is OK' is a status message, not an
    error, and must not kill an in-flight request."""
    holder = make_provider(monkeypatch, n_bars=10, delay_between_bars=0.002)
    with IBProvider(IBConfig(cache_dir=None)) as ib:
        app = holder["app"]
        threading.Timer(0.005, lambda: app.error(-1, 2104, "Market data farm OK")).start()
        df = ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")))
    assert len(df) == 10


@pytest.mark.parametrize("args", [
    (99, 354, "not subscribed"),                                  # ibapi 9.x
    (99, 354, "not subscribed", ""),                              # ibapi 10.1x
    (99, 1717229400000, 354, "not subscribed", ""),               # ibapi 10.30+
])
def test_error_signature_is_tolerant_across_ibapi_versions(monkeypatch, args):
    """ibapi has changed error() twice. A TypeError here happens inside the
    reader thread, where it is invisible."""
    from vrplab.data.ib import _make_wrapper_class
    app = _make_wrapper_class()()
    pending = app.register(99)
    app.error(*args)
    assert pending.done.is_set(), f"error not routed for signature {args}"
    assert pending.error is not None and pending.error.code == 354


def test_connection_closed_releases_every_waiter(monkeypatch):
    """A dropped socket must fail in-flight requests, not hang until timeout."""
    from vrplab.data.ib import _make_wrapper_class
    app = _make_wrapper_class()()
    p1, p2 = app.register(1), app.register(2)
    app.connectionClosed()
    assert p1.done.is_set() and p2.done.is_set()
    assert "connection closed" in str(p1.error)


def test_daily_implied_vol_is_annualised_exactly_once(monkeypatch):
    """The units bug. TWS is sending DAILY decimals; declaring that must
    produce 35% annualised, not 2.2% and not 698%."""
    make_provider(monkeypatch, n_bars=20)
    cfg = IBConfig(cache_dir=None, vol_units=VolUnits.DAILY_DECIMAL)
    with IBProvider(cfg) as ib:
        df = ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"),
                                pd.Timestamp("2024-03-01"),
                                what="OPTION_IMPLIED_VOLATILITY"))
    assert df["close"].iloc[0] == pytest.approx(0.35, rel=1e-9)
    assert df.attrs["vol_units"] == "annual_decimal"


def test_declaring_annual_units_leaves_the_series_untouched(monkeypatch):
    """Same wire data, declared as already annualised: the adapter must not
    scale it. This is the branch that produced implied vols in the hundreds."""
    make_provider(monkeypatch, n_bars=20)
    cfg = IBConfig(cache_dir=None, vol_units=VolUnits.ANNUAL_DECIMAL)
    with IBProvider(cfg) as ib:
        df = ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"),
                                pd.Timestamp("2024-03-01"),
                                what="OPTION_IMPLIED_VOLATILITY"))
    assert df["close"].iloc[0] == pytest.approx(0.35 / np.sqrt(252), rel=1e-9)


def test_implied_vol_bars_skip_ohlc_validation(monkeypatch):
    """An IV series is not a price series; validate_bars would reject it for
    having high == low. It must not be run on this path."""
    make_provider(monkeypatch, n_bars=12)
    with IBProvider(IBConfig(cache_dir=None)) as ib:
        df = ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"),
                                pd.Timestamp("2024-03-01"),
                                what="OPTION_IMPLIED_VOLATILITY"))
    assert len(df) == 12


def test_malformed_price_bars_are_rejected_loudly(monkeypatch):
    """A bar with high < close is corrupt. Better to raise in the data layer
    than to find it in a Sharpe ratio."""
    from ibapi.client import EClient
    holder = make_provider(monkeypatch, n_bars=3)

    def bad_req(self, reqId, contract, *a, **k):
        def serve():
            self.historicalData(reqId, _Bar("20240102", 100, 100.5, 99, 105, 10))
            self.historicalDataEnd(reqId, "", "")
        threading.Thread(target=serve, daemon=True).start()

    monkeypatch.setattr(EClient, "reqHistoricalData", bad_req, raising=True)
    with IBProvider(IBConfig(cache_dir=None)) as ib:
        with pytest.raises(ValueError, match="low <= open,close <= high"):
            ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")))


def test_empty_response_raises_rather_than_returning_an_empty_frame(monkeypatch):
    from ibapi.client import EClient
    make_provider(monkeypatch)

    def empty(self, reqId, contract, *a, **k):
        threading.Timer(0.01, lambda: self.historicalDataEnd(reqId, "", "")).start()

    monkeypatch.setattr(EClient, "reqHistoricalData", empty, raising=True)
    with IBProvider(IBConfig(cache_dir=None)) as ib:
        with pytest.raises(IBError, match="no bars returned"):
            ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")))


def test_timeout_is_reported_and_the_request_cancelled(monkeypatch):
    from ibapi.client import EClient
    make_provider(monkeypatch)
    cancelled = []
    monkeypatch.setattr(EClient, "reqHistoricalData",
                        lambda self, *a, **k: None, raising=True)
    monkeypatch.setattr(EClient, "cancelHistoricalData",
                        lambda self, reqId: cancelled.append(reqId), raising=True)
    with IBProvider(IBConfig(cache_dir=None, request_timeout=0.3)) as ib:
        with pytest.raises(IBError, match="timed out"):
            ib.bars(BarRequest("TEST", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")))
    assert cancelled, "a timed-out historical request must be cancelled"


def test_delayed_tick_types_produce_the_same_fields_as_live(monkeypatch):
    """IB sends delayed bid/ask/last as 66/67/68. Handling only 1/2/4 means a
    delayed feed yields nothing at all, silently."""
    from vrplab.data.ib import _make_wrapper_class
    app = _make_wrapper_class()()
    for tick_type, price in ((66, 10.0), (67, 10.4), (68, 10.2)):
        app.tickPrice(7, tick_type, price, None)
    snap = app.ticks[7]
    assert snap["bid"] == 10.0 and snap["ask"] == 10.4 and snap["last"] == 10.2


def test_non_positive_ticks_are_ignored(monkeypatch):
    from vrplab.data.ib import _make_wrapper_class
    app = _make_wrapper_class()()
    app.tickPrice(7, 4, -1.0, None)
    assert 7 not in app.ticks


def test_option_computation_captures_per_contract_iv(monkeypatch):
    """The right source of implied vol: an actual strike and expiry, rather
    than IB's aggregate underlying series of unstated tenor."""
    from vrplab.data.ib import _make_wrapper_class
    app = _make_wrapper_class()()
    app.tickOptionComputation(5, 13, 0, 0.62, 0.51, 8.4, 0.0, 0.03, 0.11, -0.25, 181.5)
    d = app.ticks[5]
    assert d["iv"] == pytest.approx(0.62)
    assert d["und_price"] == pytest.approx(181.5)
    assert d["vega"] == pytest.approx(0.11)


def test_cache_round_trips_and_avoids_a_second_request(tmp_path, monkeypatch):
    holder = make_provider(monkeypatch, n_bars=15)
    cfg = IBConfig(cache_dir=str(tmp_path))
    req = BarRequest("TEST", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01"))
    with IBProvider(cfg) as ib:
        first = ib.bars(req)
        second = ib.bars(req)
    assert len(holder["tws"].seen_req_ids) == 1, "second call should hit the cache"
    pd.testing.assert_frame_equal(first[["open", "close"]], second[["open", "close"]])


def test_pacing_limiter_blocks_past_the_quota():
    from vrplab.data.ib import _RateLimiter
    lim = _RateLimiter(max_calls=3, window_s=60.0)
    for _ in range(3):
        lim.acquire()
    done = threading.Event()
    threading.Thread(target=lambda: (lim.acquire(), done.set()), daemon=True).start()
    assert not done.wait(timeout=0.4), "4th call inside the window should block"


def test_provider_satisfies_the_data_protocol():
    from vrplab.data.base import MarketDataProvider
    assert isinstance(IBProvider(IBConfig(cache_dir=None)), MarketDataProvider)
