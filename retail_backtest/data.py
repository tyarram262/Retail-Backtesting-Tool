"""Daily OHLCV market data via yfinance, with local caching and explicit,
self-computed price adjustment.

KNOWN DATA LIMITATIONS — read before trusting any backtest built on this:

- SURVIVORSHIP BIAS: a hand-written ticker universe contains only companies
  that survived to today. Delisted/acquired tickers simply return no data
  from Yahoo, so historical results are biased upward. Until we have
  point-in-time universe data, treat absolute performance numbers with
  suspicion; relative comparisons (backtest vs live over the same period,
  which is this tool's core job) are much less affected.

- ADJUSTED-CLOSE QUIRKS: Yahoo's 'Adj Close' is back-adjusted from *today*
  and mixes splits and dividends into one opaque series that silently
  changes every ex-dividend date. We do NOT use it. Worse, yfinance's
  auto_adjust=False prices are NOT as-traded either: they are already
  split-adjusted to the fetch date (verified: NVDA June 2024 pre-split days
  show ~$120 when it traded ~$1208), and so are the dividend amounts and
  volume. The fetch layer therefore UN-adjusts everything back to true
  as-traded prices using the full split history — including splits that
  occurred AFTER the requested window, which still scale what Yahoo
  returns. From those true raw prices we compute two adjustments ourselves:
    * sim_* prices: SPLIT-ONLY adjusted (continuous across splits, no
      dividend distortion). All simulation math runs on these, and fills
      convert back to raw price / actual share count so the backtest trade
      log is directly comparable to a live broker log.
    * adj_close: split+dividend adjusted (total-return style), used ONLY
      for signal computation. RS ratios are scale-invariant, so
      back-adjustment leaks no future information into signals.

- DIVIDENDS ARE EXCLUDED from simulated returns (sim prices are split-only).
  A live account receives them; the reconciler treats that as a known,
  explainable divergence rather than hiding it inside adjusted prices.

- Yahoo daily bars can be revised/corrected after the fact and occasionally
  disagree with broker fills even at the open. Small price-matching noise in
  reconciliation is expected; that's what the tolerance thresholds are for.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

RAW_COLUMNS = ["open", "high", "low", "close", "volume", "dividends", "splits"]
_PRICE_FIELDS = ["open", "high", "low", "close"]


class DataError(RuntimeError):
    pass


def adjust_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Add adjustment columns to a raw per-ticker frame.

    Input columns: open/high/low/close/volume/dividends/splits, where
    splits[t] is the split ratio taking effect on t (prices on/after t are
    already post-split; 0 = no split that day).

    Adds:
      split_cum  C_t: cumulative product of split ratios up to and incl. t
      sim_*      split-only adjusted prices, in END-OF-RANGE share basis:
                 sim_p_t = p_t * C_t / C_end
      adj_close  sim_close additionally back-adjusted for dividends using
                 the CRSP-style factor f_d = 1 - D_d / close_{d-1}
    """
    out = raw.copy()
    ratios = out["splits"].where(out["splits"] > 0, 1.0)
    c = ratios.cumprod()
    c_end = c.iloc[-1]
    out["split_cum"] = c
    for fld in _PRICE_FIELDS:
        out[f"sim_{fld}"] = out[fld] * c / c_end

    # Dividend back-adjustment on the split-adjusted series. The factor for
    # ex-date d applies to all prices BEFORE d. Raw D_d / raw close_{d-1} is
    # scale-consistent because both sides carry the same split basis.
    f = 1.0 - out["dividends"] / out["close"].shift(1)
    f = f.fillna(1.0).clip(lower=0.0)
    # product of factors at dates strictly AFTER t
    future_factor = f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    out["adj_close"] = out["sim_close"] * future_factor
    return out


@dataclass
class MarketData:
    """Wide per-field frames: index = master trading calendar, columns = tickers."""

    frames: dict[str, pd.DataFrame] = field(repr=False)  # adjusted per-ticker frames
    calendar: pd.DatetimeIndex = field(repr=False)
    open: pd.DataFrame = field(init=False, repr=False)
    high: pd.DataFrame = field(init=False, repr=False)
    low: pd.DataFrame = field(init=False, repr=False)
    close: pd.DataFrame = field(init=False, repr=False)
    sim_open: pd.DataFrame = field(init=False, repr=False)
    sim_high: pd.DataFrame = field(init=False, repr=False)
    sim_low: pd.DataFrame = field(init=False, repr=False)
    sim_close: pd.DataFrame = field(init=False, repr=False)
    adj_close: pd.DataFrame = field(init=False, repr=False)
    split_cum: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for fld in [
            "open", "high", "low", "close",
            "sim_open", "sim_high", "sim_low", "sim_close",
            "adj_close", "split_cum",
        ]:
            wide = pd.DataFrame(
                {t: f[fld] for t, f in self.frames.items()}, index=self.calendar
            )
            setattr(self, fld, wide)
        # valuation prices: last known sim_close, for marking equity on days a
        # ticker doesn't trade (never used for fills)
        self.valuation_close = self.sim_close.ffill()

    @property
    def tickers(self) -> list[str]:
        return list(self.frames)

    def basis_factor(self, ticker: str, when: pd.Timestamp) -> float:
        """C_t / C_end: converts sim-basis quantities to actual (as-traded)
        quantities at `when`, and actual prices = sim price / factor."""
        c = self.split_cum[ticker]
        c_end = c.dropna().iloc[-1]
        val = c.asof(when)
        if pd.isna(val):
            val = c.dropna().iloc[0] if len(c.dropna()) else 1.0
        return float(val / c_end)


def market_from_raw(raw_frames: dict[str, pd.DataFrame]) -> MarketData:
    """Build MarketData from raw per-ticker frames (used by tests and the
    fetch path alike, so synthetic and real data run identical code)."""
    adjusted = {}
    calendar: pd.DatetimeIndex | None = None
    for ticker, raw in raw_frames.items():
        if raw.empty:
            raise DataError(f"no data for ticker {ticker!r}")
        missing = [c for c in RAW_COLUMNS if c not in raw.columns]
        if missing:
            raise DataError(f"{ticker}: raw frame missing columns {missing}")
        frame = adjust_frame(raw.sort_index())
        adjusted[ticker] = frame
        calendar = frame.index if calendar is None else calendar.union(frame.index)
    assert calendar is not None
    return MarketData(frames=adjusted, calendar=calendar)


# ---------------------------------------------------------------------------
# yfinance fetch + cache
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, ticker: str, start: date, end: date) -> Path:
    # v2: cache now stores true as-traded prices (yahoo split-adjustment undone)
    key = hashlib.sha1(f"{ticker}|{start}|{end}|v2".encode()).hexdigest()[:16]
    return cache_dir / f"{ticker}_{key}.pkl"


def _unadjust_yahoo(frame: pd.DataFrame, all_splits: pd.Series) -> pd.DataFrame:
    """Convert Yahoo's split-adjusted bars back to true as-traded values.

    all_splits must be the ticker's FULL split history (Ticker.splits), not
    just in-window events: a split after the window still scales every bar
    Yahoo returns for the window. For each bar, multiply prices and dividends
    (divide volume) by the product of all split ratios with ex-date > bar date.
    """
    r = pd.Series(1.0, index=frame.index)
    for ex_date, ratio in all_splits.items():
        if ratio and ratio > 0:
            r[frame.index < ex_date] *= float(ratio)
    out = frame.copy()
    for col in ("open", "high", "low", "close", "dividends"):
        out[col] = out[col] * r
    out["volume"] = out["volume"] / r
    return out


def fetch_raw_ticker(
    ticker: str,
    start: date,
    end: date,
    cache_dir: str | Path = "data_cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Raw daily bars + dividend/split events for one ticker, disk-cached.

    Cache is keyed by (ticker, start, end); a changed range refetches. Yahoo
    can revise recent bars, so pass refresh=True when a run's end date is
    near today and exactness matters.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, ticker, start, end)
    if path.exists() and not refresh:
        with path.open("rb") as fh:
            return pickle.load(fh)

    import yfinance as yf

    yft = yf.Ticker(ticker)
    hist = yft.history(start=str(start), end=str(end), auto_adjust=False, actions=True)
    if hist.empty:
        raise DataError(
            f"yfinance returned no data for {ticker!r} in {start}..{end} "
            "(delisted? typo? see survivorship-bias note in retail_backtest/data.py)"
        )
    hist.index = pd.DatetimeIndex(hist.index.tz_localize(None)).normalize()
    renamed = hist.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "splits",
        }
    )
    for col in ("dividends", "splits"):
        if col not in renamed.columns:
            renamed[col] = 0.0

    all_splits = yft.splits
    if all_splits is None or all_splits.empty:
        all_splits = pd.Series(dtype=float)
    else:
        all_splits.index = pd.DatetimeIndex(all_splits.index.tz_localize(None)).normalize()

    raw = _unadjust_yahoo(renamed[RAW_COLUMNS], all_splits)
    with path.open("wb") as fh:
        pickle.dump(raw, fh)
    return raw


def load_market_data(
    tickers: list[str],
    start: date,
    end: date,
    cache_dir: str | Path = "data_cache",
    refresh: bool = False,
) -> MarketData:
    raw_frames = {
        t: fetch_raw_ticker(t, start, end, cache_dir=cache_dir, refresh=refresh)
        for t in dict.fromkeys(tickers)  # dedupe, keep order
    }
    return market_from_raw(raw_frames)
