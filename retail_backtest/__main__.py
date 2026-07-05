"""CLI entry point.

Usage:
    python -m retail_backtest validate <config.yaml> [<config.yaml> ...]
    python -m retail_backtest backtest <strategy.yaml> <run.yaml> [--out DIR] [--refresh]

validate auto-detects config kind: a top-level `run` key means run config,
otherwise strategy config. Exits non-zero if any file fails validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, parse_run_config, parse_strategy_config
from .config.loader import _load_yaml_mapping


def _validate_file(path: Path) -> str:
    data = _load_yaml_mapping(path)
    if "run" in data:
        cfg = parse_run_config(data, source=str(path))
        wf = cfg.run.walk_forward
        return (
            f"run config: {cfg.run.start}..{cfg.run.end}, "
            f"${cfg.run.initial_capital:,.0f}, walk-forward "
            f"{wf.train_months}m warm-up / {wf.test_months}m test / {wf.step_months}m step"
        )
    cfg = parse_strategy_config(data, source=str(path))
    return (
        f"strategy '{cfg.meta.name}': {len(cfg.universe.tickers)} tickers, "
        f"{len(cfg.signals)} signals, benchmarks {cfg.benchmarks()}, "
        f"max {cfg.sizing.max_positions} positions"
    )


def _cmd_backtest(args: argparse.Namespace) -> int:
    import pandas as pd

    from .engine import run_backtest

    cfg = parse_strategy_config(_load_yaml_mapping(args.strategy), source=str(args.strategy))
    run_cfg = parse_run_config(_load_yaml_mapping(args.run), source=str(args.run))

    result = run_backtest(cfg, run_cfg, cache_dir=args.cache, refresh=args.refresh)

    out_dir = args.out / f"{cfg.meta.name}_{run_cfg.run.start}_{run_cfg.run.end}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.fills_frame().to_csv(out_dir / "trades.csv", index=False)
    curves = pd.DataFrame({"equity": result.equity, "invested": result.invested})
    curves.rename_axis("date").to_csv(out_dir / "equity.csv")
    rt = result.round_trips.copy()
    for col in ("entry_date", "exit_date"):
        rt[col] = pd.to_datetime(rt[col]).dt.date
    rt.to_csv(out_dir / "round_trips.csv", index=False)
    windows_frame = pd.DataFrame([w.row() for w in result.windows])
    windows_frame.to_csv(out_dir / "windows.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result.summary(), indent=2, default=str))

    print(f"strategy '{cfg.meta.name}'  {run_cfg.run.start}..{run_cfg.run.end}  "
          f"initial ${run_cfg.run.initial_capital:,.0f}")
    print()
    print("Per walk-forward window (continuous sim, sliced):")
    print(windows_frame.to_string(index=False))
    print()
    agg = result.aggregate.row()
    print(
        f"FULL PERIOD: return {agg['return_pct']}%  sharpe {agg['sharpe']}  "
        f"max_dd {agg['max_dd_pct']}%  win_rate {agg['win_rate_pct']}%  "
        f"closed_trades {agg['closed_trades']}  fills {agg['fills']}"
    )
    if result.notes:
        print("\nnotes:")
        for note in result.notes:
            print(f"  - {note}")
    print(f"\nartifacts: {out_dir}/  (trades.csv, round_trips.csv, equity.csv, "
          "windows.csv, summary.json)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retail_backtest")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate strategy/run config files")
    validate.add_argument("paths", nargs="+", type=Path)

    backtest = sub.add_parser("backtest", help="run a walk-forward backtest")
    backtest.add_argument("strategy", type=Path, help="strategy config YAML")
    backtest.add_argument("run", type=Path, help="run config YAML")
    backtest.add_argument("--out", type=Path, default=Path("output"))
    backtest.add_argument("--cache", type=Path, default=Path("data_cache"))
    backtest.add_argument("--refresh", action="store_true",
                          help="refetch price data even if cached")

    args = parser.parse_args(argv)

    if args.command == "backtest":
        try:
            return _cmd_backtest(args)
        except ConfigError as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 1

    failed = False
    for path in args.paths:
        try:
            summary = _validate_file(path)
        except ConfigError as exc:
            failed = True
            print(f"FAIL  {exc}", file=sys.stderr)
        else:
            print(f"OK    {path}  ({summary})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
