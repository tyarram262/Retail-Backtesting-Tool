import copy
from datetime import date
from pathlib import Path

import pytest
import yaml

from retail_backtest.config import (
    ConfigError,
    load_run_config,
    load_strategy_config,
    parse_run_config,
    parse_strategy_config,
)

REPO_ROOT = Path(__file__).parents[1]
STRATEGY_FIXTURE = REPO_ROOT / "strategies" / "momentum_rs_v1.yaml"
RUN_FIXTURE = REPO_ROOT / "runs" / "run_2023_2026.yaml"


@pytest.fixture
def strategy_dict() -> dict:
    """The shipped momentum config as a mutable dict for negative tests."""
    return copy.deepcopy(yaml.safe_load(STRATEGY_FIXTURE.read_text()))


@pytest.fixture
def run_dict() -> dict:
    return copy.deepcopy(yaml.safe_load(RUN_FIXTURE.read_text()))


def expect_rejection(data: dict, fragment: str, parse=parse_strategy_config):
    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert fragment in str(excinfo.value), (
        f"expected {fragment!r} in error:\n{excinfo.value}"
    )


# ---------------------------------------------------------------------------
# The shipped fixtures are valid and parse to the confirmed semantics
# ---------------------------------------------------------------------------


def test_momentum_fixture_parses():
    cfg = load_strategy_config(STRATEGY_FIXTURE)
    assert cfg.meta.name == "momentum_rs_v1"
    assert cfg.sizing.max_positions == 4
    assert cfg.sizing.max_weight_pct == 20
    assert cfg.benchmarks() == ["QQQ", "SPY"]
    assert cfg.max_lookback_days() == 60
    assert cfg.execution.fill_timing == "next_open"
    assert cfg.costs.slippage_bps == 5.0
    assert len(cfg.entry.all) == 4
    exit_stops = {c.rule for c in cfg.exit.any if hasattr(c, "rule")}
    assert exit_stops == {"stop_loss", "trailing_stop"}


def test_run_fixture_parses():
    cfg = load_run_config(RUN_FIXTURE)
    assert cfg.run.start == date(2023, 1, 1)
    assert cfg.run.walk_forward.test_months == 3


def test_tickers_normalized(strategy_dict):
    strategy_dict["universe"]["tickers"] = [" aapl ", "brk.b"]
    cfg = parse_strategy_config(strategy_dict)
    assert cfg.universe.tickers == ["AAPL", "BRK.B"]


def test_costs_default_when_omitted(strategy_dict):
    del strategy_dict["costs"]
    cfg = parse_strategy_config(strategy_dict)
    assert cfg.costs.slippage_bps == 5.0
    assert cfg.costs.commission_per_share == 0.0


# ---------------------------------------------------------------------------
# Ambiguous / malformed strategy configs are rejected, not guessed at
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected(strategy_dict):
    strategy_dict["stop_loss"] = 8
    expect_rejection(strategy_dict, "stop_loss")


def test_unknown_indicator_rejected(strategy_dict):
    strategy_dict["signals"]["rs_20_spy"]["indicator"] = "rsi"
    expect_rejection(strategy_dict, "signals.rs_20_spy")


def test_undefined_signal_in_entry_rejected(strategy_dict):
    strategy_dict["entry"]["all"][0]["signal"] = "rs_90_spy"
    expect_rejection(strategy_dict, "undefined signal 'rs_90_spy'")


def test_undefined_ranking_signal_rejected(strategy_dict):
    strategy_dict["ranking"]["by"] = "momentum_score"
    expect_rejection(strategy_dict, "ranking.by references undefined signal")


def test_bad_op_rejected(strategy_dict):
    strategy_dict["entry"]["all"][0]["op"] = ">="
    expect_rejection(strategy_dict, "entry.all.0.op")


def test_duplicate_tickers_rejected(strategy_dict):
    strategy_dict["universe"]["tickers"] = ["AAPL", "aapl", "MSFT"]
    expect_rejection(strategy_dict, "duplicates")


def test_infeasible_sizing_rejected(strategy_dict):
    # 12 positions => 8.3% each, below the 10% floor: impossible to satisfy
    strategy_dict["sizing"]["max_positions"] = 12
    expect_rejection(strategy_dict, "infeasible sizing")


def test_min_weight_above_max_rejected(strategy_dict):
    strategy_dict["sizing"]["min_weight_pct"] = 25
    expect_rejection(strategy_dict, "min_weight_pct")


def test_weekly_requires_day_of_week(strategy_dict):
    del strategy_dict["rebalance"]["day_of_week"]
    expect_rejection(strategy_dict, "day_of_week is required")


def test_daily_forbids_day_of_week(strategy_dict):
    strategy_dict["rebalance"]["frequency"] = "daily"
    expect_rejection(strategy_dict, "only valid for weekly")


def test_duplicate_stop_rules_rejected(strategy_dict):
    strategy_dict["exit"]["any"].append({"rule": "stop_loss", "pct": 15})
    expect_rejection(strategy_dict, "duplicate stop rules")


def test_unsupported_fill_timing_rejected(strategy_dict):
    strategy_dict["execution"]["fill_timing"] = "same_close"
    expect_rejection(strategy_dict, "execution.fill_timing")


def test_stop_pct_must_be_positive(strategy_dict):
    strategy_dict["exit"]["any"][1]["pct"] = 0
    expect_rejection(strategy_dict, "exit.any.1")


# ---------------------------------------------------------------------------
# YAML-level ambiguity
# ---------------------------------------------------------------------------


def test_duplicate_yaml_key_rejected(tmp_path, strategy_dict):
    # A second sizing: block would silently win with vanilla yaml.safe_load
    text = STRATEGY_FIXTURE.read_text() + (
        "\nsizing:\n"
        "  method: equal_weight\n"
        "  max_positions: 10\n"
        "  max_weight_pct: 20\n"
        "  min_weight_pct: 10\n"
    )
    path = tmp_path / "dupe.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError) as excinfo:
        load_strategy_config(path)
    assert "duplicate key" in str(excinfo.value)


def test_non_mapping_top_level_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError) as excinfo:
        load_strategy_config(path)
    assert "top level must be a mapping" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Run configs
# ---------------------------------------------------------------------------


def test_run_end_before_start_rejected(run_dict):
    run_dict["run"]["end"] = date(2022, 1, 1)
    expect_rejection(run_dict, "must be after", parse=parse_run_config)


def test_run_range_shorter_than_test_window_rejected(run_dict):
    run_dict["run"]["end"] = date(2023, 2, 1)
    expect_rejection(run_dict, "shorter than one test window", parse=parse_run_config)


def test_step_larger_than_test_rejected(run_dict):
    run_dict["run"]["walk_forward"]["step_months"] = 6
    expect_rejection(run_dict, "uncovered gaps", parse=parse_run_config)
