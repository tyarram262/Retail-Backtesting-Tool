"""Engine semantics tests on synthetic data. Each test constructs prices
where the correct trade (date, price, quantity, reason) is computable by
hand, then asserts the simulator produced exactly that."""

import pandas as pd
import pytest

from retail_backtest.engine import run_backtest
from synth import make_market, make_run, make_strategy, raw_frame

D60 = pd.bdate_range("2024-01-02", periods=60)
D40 = pd.bdate_range("2024-01-02", periods=40)
D35 = pd.bdate_range("2024-01-02", periods=35)

BENCH_FLAT_60 = raw_frame([100.0] * 60, D60)


def bench(dates):
    return raw_frame([100.0] * len(dates), dates)


# ---------------------------------------------------------------------------
# Entry: signal on close T, fill next open, equal-weight capped sizing
# ---------------------------------------------------------------------------


def test_entry_fills_next_open_at_capped_weight():
    win = raw_frame([100 * 1.01**i for i in range(60)], D60)
    cfg = make_strategy(["WIN"], exit_any=[{"signal": "rs", "op": "lt", "value": 0.5}])
    run = make_run(D60[10].date(), D60[-1].date())
    result = run_backtest(cfg, run, market=make_market({"WIN": win, "BENCH": BENCH_FLAT_60}))

    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.side == "buy"
    assert fill.reason == "entry_signal"
    assert fill.signal_date == D60[10]  # signal at close of first sim day...
    assert fill.trade_date == D60[11]  # ...fills at next day's open
    expected_price = 100 * 1.01**10  # open = prior close, zero slippage
    assert fill.price == pytest.approx(expected_price, rel=1e-9)
    # 1 position -> target = min(100/1, 20) = 20% of 100k, NOT all-in
    assert fill.quantity * fill.price == pytest.approx(20_000, rel=1e-9)
    assert fill.signal_snapshot["rs"] == pytest.approx(1.01**5, rel=1e-4)
    # open round trip, no exit
    assert result.round_trips["exit_date"].isna().all()


def test_exit_signal_fires_on_rs_breach():
    closes = [100 * 1.01**i for i in range(30)]
    closes += [closes[29] * 0.97 ** (i - 29) for i in range(30, 60)]
    win = raw_frame(closes, D60)
    cfg = make_strategy(["WIN"])  # exit: rs < 0.95
    run = make_run(D60[10].date(), D60[-1].date())
    result = run_backtest(cfg, run, market=make_market({"WIN": win, "BENCH": BENCH_FLAT_60}))

    assert len(result.fills) == 2
    sell = result.fills[1]
    assert sell.side == "sell"
    assert sell.reason == "exit_signal"
    # first day RS_5 < 0.95 is index 32 (1.01^2 * 0.97^3 = 0.931)
    assert sell.signal_date == D60[32]
    assert sell.trade_date == D60[33]
    assert sell.price == pytest.approx(closes[32], rel=1e-9)

    closed = result.round_trips.dropna(subset=["exit_date"])
    assert len(closed) == 1
    assert closed.iloc[0]["pnl_pct"] > 0
    assert result.aggregate.win_rate_pct == 100.0
    assert len(result.windows) == 3  # ~Jan16..Mar25 in 1-month windows


# ---------------------------------------------------------------------------
# Stops: intraday standing orders
# ---------------------------------------------------------------------------


def test_stop_loss_touched_intraday_fills_at_level_same_day():
    entry_px = 100 * 1.02**8
    closes = [100 * 1.02**i for i in range(9)] + [entry_px] * 6 + [0.9 * entry_px] * 20
    stk = raw_frame(closes, D35)
    cfg = make_strategy(["STK"], exit_any=[{"rule": "stop_loss", "pct": 8}])
    run = make_run(D35[8].date(), D35[-1].date())
    result = run_backtest(cfg, run, market=make_market({"STK": stk, "BENCH": bench(D35)}))

    assert len(result.fills) == 2
    buy, sell = result.fills
    assert buy.price == pytest.approx(entry_px, rel=1e-9)
    assert sell.reason == "stop_loss"
    # low (0.90 * entry) pierced the level intraday; open didn't gap through,
    # so the fill is the stop level itself, SAME day (not next open)
    assert sell.trade_date == D35[15]
    assert sell.price == pytest.approx(entry_px * 0.92, rel=1e-9)
    assert sell.quantity == pytest.approx(buy.quantity, rel=1e-9)
    closed = result.round_trips.dropna(subset=["exit_date"])
    assert closed.iloc[0]["pnl_pct"] == pytest.approx(-8.0, rel=1e-6)


def test_stop_loss_gap_down_fills_at_open():
    entry_px = 100 * 1.02**8
    closes = [100 * 1.02**i for i in range(9)] + [entry_px] * 6 + [0.82 * entry_px] * 20
    opens = [closes[0]] + closes[:-1]
    opens[15] = 0.80 * entry_px  # gap straight through the 0.92 level
    stk = raw_frame(closes, D35, open_=opens)
    cfg = make_strategy(["STK"], exit_any=[{"rule": "stop_loss", "pct": 8}])
    run = make_run(D35[8].date(), D35[-1].date())
    result = run_backtest(cfg, run, market=make_market({"STK": stk, "BENCH": bench(D35)}))

    sell = result.fills[1]
    assert sell.reason == "stop_loss"
    assert sell.trade_date == D35[15]
    assert sell.price == pytest.approx(0.80 * entry_px, rel=1e-9)  # open, not level


def test_trailing_stop_anchors_to_highest_close():
    peak = 100 * 1.02**20
    closes = [100 * 1.02**i for i in range(21)]  # rise through index 20
    closes.append(0.86 * peak)  # index 21: drop below 12% trail
    while len(closes) < 40:
        closes.append(closes[-1] * 0.97)
    stk = raw_frame(closes, D40)
    cfg = make_strategy(["STK"], exit_any=[{"rule": "trailing_stop", "pct": 12}])
    run = make_run(D40[8].date(), D40[-1].date())
    result = run_backtest(cfg, run, market=make_market({"STK": stk, "BENCH": bench(D40)}))

    assert len(result.fills) == 2
    sell = result.fills[1]
    assert sell.reason == "trailing_stop"
    assert sell.trade_date == D40[21]
    assert sell.price == pytest.approx(0.88 * peak, rel=1e-9)
    closed = result.round_trips.dropna(subset=["exit_date"])
    assert closed.iloc[0]["pnl_pct"] > 0  # rode the trend up before the stop


# ---------------------------------------------------------------------------
# Ranking, slots, costs
# ---------------------------------------------------------------------------


def test_ranking_fills_best_signals_up_to_max_positions():
    frames = {
        f"T{k}": raw_frame([100 * (1 + 0.002 * k) ** i for i in range(60)], D60)
        for k in range(1, 7)
    }
    frames["BENCH"] = BENCH_FLAT_60
    cfg = make_strategy(
        [f"T{k}" for k in range(1, 7)],
        exit_any=[{"signal": "rs", "op": "lt", "value": 0.5}],
    )
    run = make_run(D60[10].date(), D60[-1].date())
    result = run_backtest(cfg, run, market=make_market(frames))

    buys = [f for f in result.fills if f.side == "buy"]
    assert len(result.fills) == 4  # slots full afterwards, exits never fire
    assert {f.ticker for f in buys} == {"T3", "T4", "T5", "T6"}  # steepest RS wins
    for f in buys:
        # 4 positions -> min(100/4, 20) = 20% each; 20% stays in cash
        assert f.quantity * f.price == pytest.approx(20_000, rel=1e-9)
    assert 60 < result.aggregate.avg_exposure_pct < 95


def test_slippage_and_commissions_applied():
    win = raw_frame([100 * 1.01**i for i in range(60)], D60)
    cfg = make_strategy(
        ["WIN"],
        exit_any=[{"signal": "rs", "op": "lt", "value": 0.5}],
        costs={"slippage_bps": 10, "commission_per_share": 0.01, "commission_per_trade": 1.5},
    )
    run = make_run(D60[10].date(), D60[-1].date())
    result = run_backtest(cfg, run, market=make_market({"WIN": win, "BENCH": BENCH_FLAT_60}))

    fill = result.fills[0]
    raw_open = 100 * 1.01**10
    assert fill.price == pytest.approx(raw_open * 1.0010, rel=1e-9)  # buy slips UP
    assert fill.commission == pytest.approx(fill.quantity * 0.01 + 1.5, rel=1e-9)


# ---------------------------------------------------------------------------
# Splits: sim continuity + as-traded trade log
# ---------------------------------------------------------------------------


def test_split_mid_hold_keeps_sim_continuous_and_trade_log_as_traded():
    pre = [100 * 1.02**i for i in range(15)]
    post = [100 * 1.02**i / 4 for i in range(15, 20)]  # 4:1 split at index 15
    tail = [post[-1] * 0.95 ** (i - 19) for i in range(20, 35)]
    stk = raw_frame(pre + post + tail, D35, splits={15: 4.0})
    cfg = make_strategy(["STK"])  # exit: rs < 0.95
    run = make_run(D35[8].date(), D35[-1].date())
    result = run_backtest(cfg, run, market=make_market({"STK": stk, "BENCH": bench(D35)}))

    assert len(result.fills) == 2
    buy, sell = result.fills
    # entry before the split: as-traded (pre-split) price and share count
    assert buy.price == pytest.approx(100 * 1.02**8, rel=1e-9)
    assert buy.quantity == pytest.approx(20_000 / (100 * 1.02**8), rel=1e-9)
    # split did NOT fake an RS crash: exit comes from the real 5%/day decline
    assert sell.reason == "exit_signal"
    assert sell.trade_date == D35[23]
    # exit after the split: 4x the shares at the post-split price
    assert sell.quantity == pytest.approx(4 * buy.quantity, rel=1e-9)
    assert sell.price == pytest.approx((100 * 1.02**19 / 4) * 0.95**3, rel=1e-9)
    # no phantom equity jump on split day (position ~21% weight, +2% move)
    ratio = result.equity[D35[15]] / result.equity[D35[14]]
    assert 1.003 < ratio < 1.006


# ---------------------------------------------------------------------------
# Rebalance timing
# ---------------------------------------------------------------------------


def test_weekly_rebalance_enters_only_on_rebalance_day():
    win = raw_frame([100 * 1.01**i for i in range(60)], D60)
    cfg = make_strategy(
        ["WIN"],
        exit_any=[{"signal": "rs", "op": "lt", "value": 0.5}],
        rebalance={"frequency": "weekly", "day_of_week": "monday"},
    )
    run = make_run(D60[9].date(), D60[-1].date())  # starts Mon 2024-01-15
    result = run_backtest(cfg, run, market=make_market({"WIN": win, "BENCH": BENCH_FLAT_60}))

    assert len(result.fills) == 1
    assert result.fills[0].signal_date == D60[9]  # Monday close
    assert result.fills[0].trade_date == D60[10]  # Tuesday open
