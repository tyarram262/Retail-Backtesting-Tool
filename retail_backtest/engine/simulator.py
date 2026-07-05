"""Daily portfolio simulator.

Execution semantics (these define what "the strategy should have done"
means for reconciliation — change with care):

- Signals are evaluated at the CLOSE of day T; resulting orders fill at the
  next trading day's OPEN, with slippage applied (buys up, sells down).
- ENTRIES happen only on rebalance days. EXIT signal conditions are
  evaluated every day.
- STOPS (stop_loss, trailing_stop) are modeled as standing intraday orders,
  matching a live broker stop: if the day's open gaps through the level the
  fill is the open; if the intraday low touches it the fill is the level
  itself. Stop-loss anchors to the entry fill price; trailing anchors to the
  highest close since entry (seeded with the entry fill). When both are
  configured the higher (first-hit) level fires and gets the reason code.
- SIZING (equal-weight capped): new entries target
  min(1/positions_after_entries, max_weight_pct) of total equity, computed
  at signal time. Existing positions are NOT resized at rebalance (no
  trimming churn). If cash at fill can't fund at least min_weight_pct, the
  entry is skipped and noted.
- Same-open ordering: sells process before buys, so an exit can free the
  slot and the cash that a same-morning entry uses.
- All simulation math runs on split-adjusted sim prices; every fill also
  records the as-traded price and share count (raw basis at the fill date)
  so the trade log lines up with a live broker log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from ..config.models import RunConfig, SignalCondition, StopRule, StrategyConfig
from ..data import MarketData
from .signals import condition_holds, signal_snapshot

_WEEKDAY_IDX = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}
_MAX_FILL_ATTEMPTS = 5


@dataclass
class Fill:
    trade_id: int
    position_id: int
    source: str
    ticker: str
    side: str  # buy | sell
    trade_date: pd.Timestamp
    quantity: float  # actual (as-traded) shares at trade_date basis
    price: float  # actual (as-traded) price, slippage included
    commission: float
    reason: str
    signal_date: pd.Timestamp
    signal_snapshot: dict
    sim_quantity: float  # split-adjusted internals, for debugging/audit
    sim_price: float

    def row(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "position_id": self.position_id,
            "source": self.source,
            "ticker": self.ticker,
            "side": self.side,
            "trade_date": self.trade_date.date(),
            "quantity": round(self.quantity, 6),
            "price": round(self.price, 4),
            "commission": round(self.commission, 4),
            "reason": self.reason,
            "signal_date": self.signal_date.date(),
            "signal_snapshot": json.dumps(self.signal_snapshot),
            "sim_quantity": round(self.sim_quantity, 6),
            "sim_price": round(self.sim_price, 4),
        }


@dataclass
class _Position:
    pid: int
    ticker: str
    sim_shares: float
    entry_sim_price: float
    entry_date: pd.Timestamp
    entry_commission: float
    trail_ref: float
    exiting: bool = False


@dataclass
class _Order:
    ticker: str
    side: str
    reason: str
    signal_date: pd.Timestamp
    snapshot: dict
    dollars: float = 0.0  # buys: target notional
    equity_basis: float = 0.0  # buys: equity used for sizing, for the min-weight floor
    position: _Position | None = None  # sells
    attempts: int = 0


@dataclass
class SimResult:
    fills: list[Fill]
    equity: pd.Series
    invested: pd.Series
    round_trips: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def rebalance_days(
    dates: pd.DatetimeIndex, frequency: str, day_of_week: str | None
) -> set[pd.Timestamp]:
    """Rebalance dates within the simulated calendar. Weekly: first trading
    day of the ISO week with weekday >= the configured day (a Monday holiday
    rolls to Tuesday). Monthly: first trading day of the month."""
    if frequency == "daily":
        return set(dates)
    picked: set[pd.Timestamp] = set()
    if frequency == "weekly":
        target = _WEEKDAY_IDX[day_of_week]  # type: ignore[index]
        iso = dates.isocalendar()
        for (_, _), group in pd.Series(dates, index=dates).groupby(
            [iso.year.values, iso.week.values]
        ):
            eligible = [d for d in group if d.dayofweek >= target]
            if eligible:
                picked.add(min(eligible))
    else:  # monthly
        for (_, _), group in pd.Series(dates, index=dates).groupby(
            [dates.year, dates.month]
        ):
            picked.add(min(group))
    return picked


class Simulator:
    def __init__(
        self,
        cfg: StrategyConfig,
        run: RunConfig,
        market: MarketData,
        signals: dict[str, pd.DataFrame],
    ):
        self.cfg = cfg
        self.run = run
        self.market = market
        self.signals = signals
        self.cash = float(run.run.initial_capital)
        self.positions: dict[str, _Position] = {}
        self.pending: list[_Order] = []
        self.fills: list[Fill] = []
        self.round_trips: list[dict] = []
        self.notes: list[str] = []
        self._next_pid = 1
        self._next_trade_id = 1

        stops = {c.rule: c for c in cfg.exit.any if isinstance(c, StopRule)}
        self.stop_loss_pct = stops["stop_loss"].pct if "stop_loss" in stops else None
        self.trailing_pct = stops["trailing_stop"].pct if "trailing_stop" in stops else None
        self.exit_signal_conds = [c for c in cfg.exit.any if isinstance(c, SignalCondition)]
        self.slip = cfg.costs.slippage_bps / 10_000.0

    # -- public ------------------------------------------------------------

    def run_sim(self) -> SimResult:
        cal = self.market.calendar
        start = pd.Timestamp(self.run.run.start)
        end = pd.Timestamp(self.run.run.end)
        sim_dates = cal[(cal >= start) & (cal <= end)]
        rebal = rebalance_days(
            sim_dates, self.cfg.rebalance.frequency, self.cfg.rebalance.day_of_week
        )

        equity_curve: dict[pd.Timestamp, float] = {}
        invested_curve: dict[pd.Timestamp, float] = {}

        for t in sim_dates:
            self._process_pending(t)
            self._check_stops(t)
            self._update_trail_refs(t)
            self._queue_signal_exits(t)
            if t in rebal:
                self._queue_entries(t)
            invested = self._invested_value(t)
            invested_curve[t] = invested
            equity_curve[t] = self.cash + invested

        for order in self.pending:
            self.notes.append(
                f"unfilled at range end: {order.side} {order.ticker} "
                f"(signaled {order.signal_date.date()}, reason {order.reason}) — "
                "a live account would fill this after the run window"
            )
        for pos in self.positions.values():
            self.round_trips.append(self._round_trip_row(pos, None, None, None, None))

        return SimResult(
            fills=self.fills,
            equity=pd.Series(equity_curve, name="equity"),
            invested=pd.Series(invested_curve, name="invested"),
            round_trips=pd.DataFrame(
                self.round_trips,
                columns=[
                    "position_id", "ticker", "entry_date", "entry_price", "quantity",
                    "exit_date", "exit_price", "exit_reason", "pnl_pct",
                ],
            ),
            notes=self.notes,
        )

    # -- open: pending order fills ------------------------------------------

    def _process_pending(self, t: pd.Timestamp) -> None:
        orders = sorted(self.pending, key=lambda o: o.side == "buy")  # sells first
        self.pending = []
        for order in orders:
            open_sim = self.market.sim_open.at[t, order.ticker]
            if pd.isna(open_sim):
                order.attempts += 1
                if order.attempts < _MAX_FILL_ATTEMPTS:
                    self.pending.append(order)
                else:
                    self.notes.append(
                        f"dropped {order.side} {order.ticker}: no tradable bar within "
                        f"{_MAX_FILL_ATTEMPTS} days of signal {order.signal_date.date()}"
                    )
                continue
            if order.side == "sell":
                pos = order.position
                assert pos is not None
                if self.positions.get(pos.ticker) is not pos:
                    continue  # already stopped out intraday before this open
                self._exit_position(
                    pos, t, float(open_sim) * (1 - self.slip), order.reason,
                    order.signal_date, order.snapshot,
                )
            else:
                self._fill_entry(order, t, float(open_sim))

    def _fill_entry(self, order: _Order, t: pd.Timestamp, open_sim: float) -> None:
        if order.ticker in self.positions:
            return
        if len(self.positions) >= self.cfg.sizing.max_positions:
            self.notes.append(f"skipped entry {order.ticker} on {t.date()}: no free slot at fill")
            return
        sim_price = open_sim * (1 + self.slip)
        sim_shares = order.dollars / sim_price
        factor = self.market.basis_factor(order.ticker, t)
        commission = self._commission(sim_shares * factor)
        cost = sim_shares * sim_price + commission
        if cost > self.cash:
            scale = self.cash / cost if cost > 0 else 0.0
            sim_shares *= scale
            commission = self._commission(sim_shares * factor)
            cost = sim_shares * sim_price + commission
        achieved = sim_shares * sim_price
        if order.equity_basis > 0 and (
            achieved / order.equity_basis * 100 < self.cfg.sizing.min_weight_pct - 1e-9
        ):
            self.notes.append(
                f"skipped entry {order.ticker} on {t.date()}: fundable size "
                f"{achieved / order.equity_basis * 100:.1f}% is below the "
                f"{self.cfg.sizing.min_weight_pct}% floor"
            )
            return
        self.cash -= cost
        pos = _Position(
            pid=self._next_pid,
            ticker=order.ticker,
            sim_shares=sim_shares,
            entry_sim_price=sim_price,
            entry_date=t,
            entry_commission=commission,
            trail_ref=sim_price,
        )
        self._next_pid += 1
        self.positions[order.ticker] = pos
        self._record_fill(
            pos, t, "buy", sim_shares, sim_price, commission,
            order.reason, order.signal_date, order.snapshot,
        )

    # -- intraday: stops ------------------------------------------------------

    def _check_stops(self, t: pd.Timestamp) -> None:
        if self.stop_loss_pct is None and self.trailing_pct is None:
            return
        for pos in list(self.positions.values()):
            open_sim = self.market.sim_open.at[t, pos.ticker]
            low_sim = self.market.sim_low.at[t, pos.ticker]
            if pd.isna(open_sim) or pd.isna(low_sim):
                continue
            stop_lvl = (
                pos.entry_sim_price * (1 - self.stop_loss_pct / 100)
                if self.stop_loss_pct is not None else None
            )
            trail_lvl = (
                pos.trail_ref * (1 - self.trailing_pct / 100)
                if self.trailing_pct is not None else None
            )
            candidates = [(lvl, name) for lvl, name in
                          [(stop_lvl, "stop_loss"), (trail_lvl, "trailing_stop")]
                          if lvl is not None]
            level, reason = max(candidates)  # higher level is hit first
            if float(open_sim) <= level:
                fill_sim = float(open_sim)  # gapped through: fill at open
            elif float(low_sim) <= level:
                fill_sim = level  # touched intraday: fill at the stop level
            else:
                continue
            snapshot = signal_snapshot(self.signals, pos.ticker, t)
            self._exit_position(
                pos, t, fill_sim * (1 - self.slip), reason, signal_date=t, snapshot=snapshot
            )

    # -- close: trail refs, signal exits, entries ------------------------------

    def _update_trail_refs(self, t: pd.Timestamp) -> None:
        for pos in self.positions.values():
            close_sim = self.market.sim_close.at[t, pos.ticker]
            if not pd.isna(close_sim):
                pos.trail_ref = max(pos.trail_ref, float(close_sim))

    def _queue_signal_exits(self, t: pd.Timestamp) -> None:
        for pos in self.positions.values():
            if pos.exiting:
                continue
            for cond in self.exit_signal_conds:
                if condition_holds(cond, self.signals, pos.ticker, t):
                    pos.exiting = True
                    self.pending.append(_Order(
                        ticker=pos.ticker, side="sell", reason="exit_signal",
                        signal_date=t,
                        snapshot=signal_snapshot(self.signals, pos.ticker, t),
                        position=pos,
                    ))
                    break

    def _queue_entries(self, t: pd.Timestamp) -> None:
        sizing = self.cfg.sizing
        held_staying = sum(1 for p in self.positions.values() if not p.exiting)
        pending_buys = sum(1 for o in self.pending if o.side == "buy")
        slots = sizing.max_positions - held_staying - pending_buys
        if slots <= 0:
            return
        busy = set(self.positions) | {o.ticker for o in self.pending}
        candidates = [
            tk for tk in self.cfg.universe.tickers
            if tk not in busy
            and all(condition_holds(c, self.signals, tk, t) for c in self.cfg.entry.all)
        ]
        if not candidates:
            return
        ranks = self.signals[self.cfg.ranking.by].loc[t, candidates].dropna()
        ranks = ranks.sort_values(ascending=self.cfg.ranking.order == "asc")
        selected = list(ranks.index[:slots])
        if not selected:
            return
        n_after = held_staying + pending_buys + len(selected)
        target_pct = min(100.0 / n_after, sizing.max_weight_pct)
        equity_now = self.cash + self._invested_value(t)
        dollars = target_pct / 100.0 * equity_now
        for tk in selected:
            self.pending.append(_Order(
                ticker=tk, side="buy", reason="entry_signal", signal_date=t,
                snapshot=signal_snapshot(self.signals, tk, t),
                dollars=dollars, equity_basis=equity_now,
            ))

    # -- bookkeeping -----------------------------------------------------------

    def _exit_position(
        self,
        pos: _Position,
        t: pd.Timestamp,
        sim_price: float,
        reason: str,
        signal_date: pd.Timestamp,
        snapshot: dict,
    ) -> None:
        factor = self.market.basis_factor(pos.ticker, t)
        commission = self._commission(pos.sim_shares * factor)
        self.cash += pos.sim_shares * sim_price - commission
        del self.positions[pos.ticker]
        self._record_fill(
            pos, t, "sell", pos.sim_shares, sim_price, commission, reason, signal_date, snapshot
        )
        gross_in = pos.sim_shares * pos.entry_sim_price
        pnl = pos.sim_shares * (sim_price - pos.entry_sim_price) - commission - pos.entry_commission
        self.round_trips.append(
            self._round_trip_row(pos, t, sim_price, reason, pnl / gross_in * 100)
        )

    def _round_trip_row(
        self,
        pos: _Position,
        exit_date: pd.Timestamp | None,
        exit_sim_price: float | None,
        exit_reason: str | None,
        pnl_pct: float | None,
    ) -> dict:
        entry_factor = self.market.basis_factor(pos.ticker, pos.entry_date)
        row = {
            "position_id": pos.pid,
            "ticker": pos.ticker,
            "entry_date": pos.entry_date,
            "entry_price": pos.entry_sim_price / entry_factor,
            "quantity": pos.sim_shares * entry_factor,
            "exit_date": exit_date,
            "exit_price": None,
            "exit_reason": exit_reason,
            "pnl_pct": pnl_pct,
        }
        if exit_date is not None and exit_sim_price is not None:
            exit_factor = self.market.basis_factor(pos.ticker, exit_date)
            row["exit_price"] = exit_sim_price / exit_factor
        return row

    def _record_fill(
        self,
        pos: _Position,
        t: pd.Timestamp,
        side: str,
        sim_shares: float,
        sim_price: float,
        commission: float,
        reason: str,
        signal_date: pd.Timestamp,
        snapshot: dict,
    ) -> None:
        factor = self.market.basis_factor(pos.ticker, t)
        self.fills.append(Fill(
            trade_id=self._next_trade_id,
            position_id=pos.pid,
            source="backtest",
            ticker=pos.ticker,
            side=side,
            trade_date=t,
            quantity=sim_shares * factor,
            price=sim_price / factor,
            commission=commission,
            reason=reason,
            signal_date=signal_date,
            signal_snapshot=snapshot,
            sim_quantity=sim_shares,
            sim_price=sim_price,
        ))
        self._next_trade_id += 1

    def _commission(self, actual_shares: float) -> float:
        c = self.cfg.costs
        return c.commission_per_share * actual_shares + c.commission_per_trade

    def _invested_value(self, t: pd.Timestamp) -> float:
        total = 0.0
        for pos in self.positions.values():
            px = self.market.valuation_close.at[t, pos.ticker]
            if not pd.isna(px):
                total += pos.sim_shares * float(px)
        return total
