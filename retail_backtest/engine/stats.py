"""Performance stats and walk-forward window slicing.

Walk-forward here is rolling out-of-sample evaluation with fixed parameters:
ONE continuous simulation over [start, end] (positions carry across window
boundaries, as they would in a live account), then stats are computed per
test-window slice. This matches how the strategy is actually run live, which
is what reconciliation needs. Independent per-window restarts (flat at each
window start) would test start-date robustness instead — not implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_returns: pd.Series, risk_free: float = 0.0) -> float | None:
    """Annualized Sharpe on daily returns, rf=0 by default. None if undefined
    (fewer than 2 observations or zero variance)."""
    r = daily_returns.dropna()
    if len(r) < 2:
        return None
    std = r.std(ddof=1)
    # float noise makes a constant series' std ~1e-19, not 0 — treat any
    # variance that tiny as undefined rather than reporting a 1e16 Sharpe
    if math.isnan(std) or std < 1e-12:
        return None
    return float((r.mean() - risk_free / TRADING_DAYS_PER_YEAR) / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown_pct(equity: pd.Series) -> float:
    """Max peak-to-trough drawdown, returned as a negative percentage."""
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    dd = eq / eq.cummax() - 1.0
    return float(dd.min() * 100)


def window_bounds(
    start: date, end: date, test_months: int, step_months: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Tile [start, end] with test windows: [start + k*step, +test_months),
    clipped at end. The final window may be shorter than test_months."""
    bounds = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cursor <= end_ts:
        w_end = min(cursor + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end_ts)
        bounds.append((cursor, w_end))
        cursor = cursor + pd.DateOffset(months=step_months)
    return bounds


@dataclass
class WindowStats:
    label: str
    start: date
    end: date
    total_return_pct: float | None
    sharpe: float | None
    max_drawdown_pct: float
    win_rate_pct: float | None  # of round trips CLOSED in the window
    n_round_trips_closed: int
    n_fills: int
    avg_exposure_pct: float | None

    def row(self) -> dict:
        return {
            "window": self.label,
            "start": self.start,
            "end": self.end,
            "return_pct": _rnd(self.total_return_pct),
            "sharpe": _rnd(self.sharpe),
            "max_dd_pct": _rnd(self.max_drawdown_pct),
            "win_rate_pct": _rnd(self.win_rate_pct),
            "closed_trades": self.n_round_trips_closed,
            "fills": self.n_fills,
            "exposure_pct": _rnd(self.avg_exposure_pct),
        }


def _rnd(x: float | None, digits: int = 2) -> float | None:
    return None if x is None else round(x, digits)


def compute_window_stats(
    label: str,
    equity: pd.Series,
    invested: pd.Series,
    round_trips: pd.DataFrame,
    fills_dates: pd.Series,
    w_start: pd.Timestamp,
    w_end: pd.Timestamp,
) -> WindowStats:
    eq = equity.loc[w_start:w_end]
    inv = invested.loc[w_start:w_end]

    total_return = None
    if len(eq) >= 2:
        total_return = float((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100)

    closed = round_trips
    if not round_trips.empty:
        closed = round_trips[
            round_trips["exit_date"].notna()
            & (round_trips["exit_date"] >= w_start)
            & (round_trips["exit_date"] <= w_end)
        ]
    win_rate = None
    if len(closed):
        win_rate = float((closed["pnl_pct"] > 0).mean() * 100)

    n_fills = int(((fills_dates >= w_start) & (fills_dates <= w_end)).sum()) if len(fills_dates) else 0

    return WindowStats(
        label=label,
        start=w_start.date(),
        end=w_end.date(),
        total_return_pct=total_return,
        sharpe=sharpe_ratio(eq.pct_change()),
        max_drawdown_pct=max_drawdown_pct(eq),
        win_rate_pct=win_rate,
        n_round_trips_closed=int(len(closed)) if not round_trips.empty else 0,
        n_fills=n_fills,
        avg_exposure_pct=float((inv / eq).mean() * 100) if len(eq) else None,
    )
