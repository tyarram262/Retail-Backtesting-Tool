# Retail Backtesting Tool

Backtest-vs-live reconciliation for systematic retail strategies: backtest a
declarative YAML strategy with walk-forward validation, ingest the actual
trade log from running it live/paper, and attribute exactly where and why the
two diverged. Analysis only — no order placement.

## Layout

- `retail_backtest/config/` — strategy/run config models + strict loader
- `strategies/` — strategy configs (what to trade)
- `runs/` — run configs (how to evaluate: date range, capital, walk-forward)
- `tests/` — pytest suite

## Usage

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m retail_backtest validate strategies/momentum_rs_v1.yaml runs/run_2023_2026.yaml
.venv/bin/pytest
```

## Build stages

1. **Config parser + validator** — done
2. **Backtest engine** (walk-forward, costs always applied) — done
   (`python -m retail_backtest backtest strategies/momentum_rs_v1.yaml runs/run_2023_2026.yaml`)
3. Live trade CSV ingestion — next
4. Reconciliation engine (divergence attribution)
5. Report view

Known data caveats (yfinance): survivorship bias in any ticker list, and
yfinance's "unadjusted" prices are actually split-adjusted to the fetch date —
the data layer un-adjusts them back to as-traded prices (see
`retail_backtest/data.py` module docstring). Signals compute on
dividend+split adjusted prices; simulated fills are recorded at as-traded
prices/quantities so they are directly comparable to live broker fills.
