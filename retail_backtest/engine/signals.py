"""Signal computation. One wide DataFrame per declared signal
(index = calendar, columns = universe tickers).

relative_strength is price-ratio RS on dividend+split adjusted closes:
    RS_t = (P_t / P_{t-n}) / (B_t / B_{t-n})
> 1.0 means the stock outperformed the benchmark over the lookback window.
NaN (insufficient history, halted ticker) never satisfies any condition:
a candidate with NaN signals can't enter, and NaN can't trigger an exit.
"""

from __future__ import annotations

import operator

import pandas as pd

from ..config.models import SignalCondition, StrategyConfig
from ..data import MarketData

_OPS = {"gt": operator.gt, "ge": operator.ge, "lt": operator.lt, "le": operator.le}


def compute_signals(cfg: StrategyConfig, market: MarketData) -> dict[str, pd.DataFrame]:
    universe = cfg.universe.tickers
    signals: dict[str, pd.DataFrame] = {}
    for name, sdef in cfg.signals.items():
        n = sdef.lookback_days
        stock = market.adj_close[universe]
        stock_ratio = stock / stock.shift(n)
        bench = market.adj_close[sdef.benchmark]
        bench_ratio = bench / bench.shift(n)
        signals[name] = stock_ratio.div(bench_ratio, axis=0)
    return signals


def condition_holds(
    cond: SignalCondition,
    signals: dict[str, pd.DataFrame],
    ticker: str,
    when: pd.Timestamp,
) -> bool:
    value = signals[cond.signal].at[when, ticker]
    if pd.isna(value):
        return False
    return bool(_OPS[cond.op](float(value), cond.value))


def signal_snapshot(
    signals: dict[str, pd.DataFrame], ticker: str, when: pd.Timestamp
) -> dict[str, float | None]:
    """All declared signal values for a ticker at a decision point; stored on
    every backtest trade so divergences can be explained later."""
    out: dict[str, float | None] = {}
    for name, frame in signals.items():
        value = frame.at[when, ticker]
        out[name] = None if pd.isna(value) else round(float(value), 6)
    return out
