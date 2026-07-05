import pandas as pd
import pytest

from retail_backtest.data import DataError, _unadjust_yahoo, market_from_raw
from synth import raw_frame

DATES = pd.bdate_range("2024-01-01", periods=10)


def test_unadjust_yahoo_restores_as_traded_prices():
    # Yahoo returns split-adjusted bars: constant 100 across a 4:1 split at
    # index 5, plus a 2:1 split AFTER the window that still scales everything
    fetched = raw_frame([100.0] * 10, DATES, splits={5: 4.0}, dividends={2: 0.25})
    full_splits = pd.Series(
        [4.0, 2.0], index=[DATES[5], DATES[-1] + pd.Timedelta(days=30)]
    )
    raw = _unadjust_yahoo(fetched, full_splits)
    # before the in-window split: undo both (4 * 2); after: undo only the later 2
    assert raw["close"].iloc[:5].tolist() == pytest.approx([800.0] * 5)
    assert raw["close"].iloc[5:].tolist() == pytest.approx([200.0] * 5)
    assert raw["dividends"].iloc[2] == pytest.approx(0.25 * 8)  # as-paid amount
    assert raw["volume"].iloc[0] == pytest.approx(1_000_000 / 8)
    # sim prices built from the un-adjusted bars are continuous again
    market = market_from_raw({"STK": raw})
    assert market.sim_close["STK"].tolist() == pytest.approx([200.0] * 10)


def test_split_adjustment_is_continuous():
    # 2:1 split at index 5: raw halves, sim series must be continuous
    raw = raw_frame([100.0] * 5 + [50.0] * 5, DATES, splits={5: 2.0})
    market = market_from_raw({"STK": raw})
    assert market.sim_close["STK"].tolist() == pytest.approx([50.0] * 10)
    # basis factor converts sim quantities to as-traded quantities
    assert market.basis_factor("STK", DATES[2]) == pytest.approx(0.5)
    assert market.basis_factor("STK", DATES[7]) == pytest.approx(1.0)


def test_dividend_adjustment_applies_to_prior_dates_only():
    raw = raw_frame([100.0] * 6, pd.bdate_range("2024-01-01", periods=6), dividends={3: 1.0})
    market = market_from_raw({"STK": raw})
    adj = market.adj_close["STK"]
    # factor 1 - 1/100 = 0.99 hits dates BEFORE the ex-date, not after
    assert adj.iloc[:3].tolist() == pytest.approx([99.0, 99.0, 99.0])
    assert adj.iloc[3:].tolist() == pytest.approx([100.0, 100.0, 100.0])
    # sim prices (used for fills) are untouched by dividends
    assert market.sim_close["STK"].tolist() == pytest.approx([100.0] * 6)


def test_missing_columns_rejected():
    bad = raw_frame([100.0] * 10, DATES).drop(columns=["dividends"])
    with pytest.raises(DataError, match="missing columns"):
        market_from_raw({"STK": bad})


def test_empty_frame_rejected():
    with pytest.raises(DataError, match="no data"):
        market_from_raw({"STK": pd.DataFrame()})
