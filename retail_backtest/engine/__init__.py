"""Backtest orchestration: data → signals → simulation → walk-forward stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config.models import RunConfig, StrategyConfig
from ..data import MarketData, load_market_data
from .signals import compute_signals
from .simulator import Fill, SimResult, Simulator
from .stats import WindowStats, compute_window_stats, window_bounds

__all__ = ["run_backtest", "BacktestResult", "Fill", "WindowStats"]


@dataclass
class BacktestResult:
    strategy: StrategyConfig
    run: RunConfig
    fills: list[Fill]
    equity: pd.Series
    invested: pd.Series
    round_trips: pd.DataFrame
    windows: list[WindowStats]
    aggregate: WindowStats
    notes: list[str] = field(default_factory=list)

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.row() for f in self.fills])

    def summary(self) -> dict:
        """Everything needed to interpret the run later, assumptions included
        (costs and execution timing are echoed on purpose — they are part of
        the result, not incidental settings)."""
        return {
            "strategy": self.strategy.meta.name,
            "universe": self.strategy.universe.tickers,
            "run": {
                "start": str(self.run.run.start),
                "end": str(self.run.run.end),
                "initial_capital": self.run.run.initial_capital,
                "walk_forward": self.run.run.walk_forward.model_dump(),
            },
            "costs": self.strategy.costs.model_dump(),
            "execution": self.strategy.execution.model_dump(),
            "aggregate": self.aggregate.row(),
            "windows": [w.row() for w in self.windows],
            "n_fills": len(self.fills),
            "notes": self.notes,
        }


def warmup_start(cfg: StrategyConfig, run: RunConfig) -> pd.Timestamp:
    """Fetch start: run start minus warm-up months minus a calendar buffer
    for the longest signal lookback (trading days → ~1.6x calendar days)."""
    lookback_buffer_days = int(cfg.max_lookback_days() * 1.6) + 10
    return (
        pd.Timestamp(run.run.start)
        - pd.DateOffset(months=run.run.walk_forward.train_months)
        - pd.Timedelta(days=lookback_buffer_days)
    )


def run_backtest(
    cfg: StrategyConfig,
    run: RunConfig,
    market: MarketData | None = None,
    cache_dir: str | Path = "data_cache",
    refresh: bool = False,
) -> BacktestResult:
    if market is None:
        tickers = list(dict.fromkeys(cfg.universe.tickers + cfg.benchmarks()))
        market = load_market_data(
            tickers,
            start=warmup_start(cfg, run).date(),
            # yfinance's end is exclusive; +1 day includes run.end itself
            end=(pd.Timestamp(run.run.end) + pd.Timedelta(days=1)).date(),
            cache_dir=cache_dir,
            refresh=refresh,
        )

    signals = compute_signals(cfg, market)
    sim: SimResult = Simulator(cfg, run, market, signals).run_sim()

    fills_dates = pd.Series([f.trade_date for f in sim.fills])
    bounds = window_bounds(
        run.run.start,
        run.run.end,
        run.run.walk_forward.test_months,
        run.run.walk_forward.step_months,
    )
    windows = [
        compute_window_stats(
            f"W{i + 1}", sim.equity, sim.invested, sim.round_trips, fills_dates, ws, we
        )
        for i, (ws, we) in enumerate(bounds)
    ]
    aggregate = compute_window_stats(
        "FULL", sim.equity, sim.invested, sim.round_trips, fills_dates,
        pd.Timestamp(run.run.start), pd.Timestamp(run.run.end),
    )

    return BacktestResult(
        strategy=cfg,
        run=run,
        fills=sim.fills,
        equity=sim.equity,
        invested=sim.invested,
        round_trips=sim.round_trips,
        windows=windows,
        aggregate=aggregate,
        notes=sim.notes,
    )
