from datetime import date

import pandas as pd
import pytest

from retail_backtest.engine.simulator import rebalance_days
from retail_backtest.engine.stats import max_drawdown_pct, sharpe_ratio, window_bounds


def test_window_bounds_tile_the_range():
    bounds = window_bounds(date(2023, 1, 1), date(2023, 12, 31), test_months=3, step_months=3)
    assert len(bounds) == 4
    assert bounds[0] == (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-03-31"))
    assert bounds[1][0] == pd.Timestamp("2023-04-01")
    assert bounds[-1][1] == pd.Timestamp("2023-12-31")


def test_window_bounds_final_window_clipped():
    bounds = window_bounds(date(2023, 1, 1), date(2023, 5, 15), test_months=3, step_months=3)
    assert bounds[-1] == (pd.Timestamp("2023-04-01"), pd.Timestamp("2023-05-15"))


def test_sharpe_undefined_for_zero_variance():
    assert sharpe_ratio(pd.Series([0.01] * 10)) is None


def test_sharpe_zero_for_zero_mean():
    assert sharpe_ratio(pd.Series([0.01, -0.01] * 5)) == pytest.approx(0.0)


def test_max_drawdown():
    eq = pd.Series([100.0, 120.0, 90.0, 110.0])
    assert max_drawdown_pct(eq) == pytest.approx(-25.0)


def test_weekly_rebalance_days_pick_first_eligible_day():
    dates = pd.bdate_range("2024-01-01", periods=20)  # Mon Jan 1 .. Fri Jan 26
    mondays = rebalance_days(dates, "weekly", "monday")
    assert mondays == {pd.Timestamp(f"2024-01-{d:02d}") for d in (1, 8, 15, 22)}
    wednesdays = rebalance_days(dates, "weekly", "wednesday")
    assert wednesdays == {pd.Timestamp(f"2024-01-{d:02d}") for d in (3, 10, 17, 24)}


def test_monthly_rebalance_days_first_trading_day():
    dates = pd.bdate_range("2024-01-01", periods=45)  # into early March
    firsts = rebalance_days(dates, "monthly", None)
    assert firsts == {
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    }
