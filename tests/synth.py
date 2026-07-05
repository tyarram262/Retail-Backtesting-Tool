"""Synthetic market data + config builders shared by the engine tests.

Synthetic frames go through the SAME adjustment path (market_from_raw →
adjust_frame) as real yfinance data, so tests exercise production code."""

from __future__ import annotations

import pandas as pd

from retail_backtest.config import parse_run_config, parse_strategy_config
from retail_backtest.data import MarketData, market_from_raw


def raw_frame(
    close,
    dates: pd.DatetimeIndex,
    open_=None,
    low=None,
    high=None,
    splits: dict[int, float] | None = None,
    dividends: dict[int, float] | None = None,
) -> pd.DataFrame:
    close_s = pd.Series(list(close), index=dates, dtype=float)
    open_s = (
        close_s.shift(1).fillna(close_s.iloc[0])  # default: open = prior close
        if open_ is None
        else pd.Series(list(open_), index=dates, dtype=float)
    )
    low_s = (
        pd.concat([open_s, close_s], axis=1).min(axis=1)
        if low is None
        else pd.Series(list(low), index=dates, dtype=float)
    )
    high_s = (
        pd.concat([open_s, close_s], axis=1).max(axis=1)
        if high is None
        else pd.Series(list(high), index=dates, dtype=float)
    )
    splits_s = pd.Series(0.0, index=dates)
    for idx, ratio in (splits or {}).items():
        splits_s.iloc[idx] = ratio
    div_s = pd.Series(0.0, index=dates)
    for idx, amount in (dividends or {}).items():
        div_s.iloc[idx] = amount
    return pd.DataFrame(
        {
            "open": open_s,
            "high": high_s,
            "low": low_s,
            "close": close_s,
            "volume": 1_000_000.0,
            "dividends": div_s,
            "splits": splits_s,
        }
    )


def make_market(frames: dict[str, pd.DataFrame]) -> MarketData:
    return market_from_raw(frames)


def make_strategy(
    universe: list[str],
    *,
    exit_any: list[dict] | None = None,
    entry_all: list[dict] | None = None,
    costs: dict | None = None,
    rebalance: dict | None = None,
    lookback: int = 5,
    benchmark: str = "BENCH",
    max_positions: int = 4,
):
    return parse_strategy_config(
        {
            "schema_version": 1,
            "meta": {"name": "synthetic_test"},
            "universe": {"tickers": universe},
            "signals": {
                "rs": {
                    "indicator": "relative_strength",
                    "lookback_days": lookback,
                    "benchmark": benchmark,
                }
            },
            "entry": {"all": entry_all or [{"signal": "rs", "op": "gt", "value": 1.0}]},
            "exit": {"any": exit_any or [{"signal": "rs", "op": "lt", "value": 0.95}]},
            "ranking": {"by": "rs", "order": "desc"},
            "sizing": {
                "method": "equal_weight",
                "max_positions": max_positions,
                "max_weight_pct": 20,
                "min_weight_pct": 10,
            },
            "rebalance": rebalance or {"frequency": "daily"},
            "costs": costs
            or {"slippage_bps": 0, "commission_per_share": 0, "commission_per_trade": 0},
        }
    )


def make_run(start, end, capital: float = 100_000, train: int = 1, test: int = 1, step: int = 1):
    return parse_run_config(
        {
            "schema_version": 1,
            "run": {
                "start": start,
                "end": end,
                "initial_capital": capital,
                "walk_forward": {
                    "train_months": train,
                    "test_months": test,
                    "step_months": step,
                },
            },
        }
    )
