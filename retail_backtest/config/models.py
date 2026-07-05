"""Pydantic models for strategy configs and run configs.

Validation philosophy: reject ambiguous configs instead of guessing intent.
Unknown keys are errors everywhere (extra="forbid"), rule conditions may only
reference declared signals, and internally inconsistent combinations (sizing
bounds that can't be satisfied, weekly rebalance without a weekday) fail at
load time rather than producing a silently wrong backtest.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_SIGNAL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Average days per month; only used for coarse "is the range long enough"
# checks, never for window boundary math (that uses the trading calendar).
_DAYS_PER_MONTH = 30.44


def _normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise ValueError(
            f"invalid ticker {raw!r} (expected 1-10 chars: letters, digits, '.', '-')"
        )
    return ticker


Ticker = Annotated[str, AfterValidator(_normalize_ticker)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Strategy config
# --------------------------------------------------------------------------


class Meta(StrictModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = ""

    @model_validator(mode="after")
    def _name_is_slug(self) -> "Meta":
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"meta.name {self.name!r} must be alphanumeric with '_' or '-' "
                "(it is used in filenames and DB keys)"
            )
        return self


class Universe(StrictModel):
    # MVP: explicit ticker list only. Dynamic filters like "top N by market
    # cap" need point-in-time constituent data that yfinance can't provide
    # without survivorship bias, so they are deliberately unsupported here.
    tickers: list[Ticker] = Field(min_length=1)

    @model_validator(mode="after")
    def _no_duplicates(self) -> "Universe":
        seen: set[str] = set()
        dupes = sorted({t for t in self.tickers if t in seen or seen.add(t)})
        if dupes:
            raise ValueError(f"universe.tickers contains duplicates: {dupes}")
        return self


class SignalDef(StrictModel):
    """A named indicator, referenced by entry/exit conditions and ranking.

    relative_strength is the price-ratio form:
        RS = (P_t / P_{t-n}) / (B_t / B_{t-n})
    where n = lookback_days and B is the benchmark. RS > 1.0 means the stock
    outperformed the benchmark over the window.
    """

    indicator: Literal["relative_strength"]
    lookback_days: int = Field(ge=2, le=252)
    benchmark: Ticker


class SignalCondition(StrictModel):
    signal: str
    op: Literal["gt", "ge", "lt", "le"]
    value: float


class StopRule(StrictModel):
    rule: Literal["stop_loss", "trailing_stop"]
    pct: float = Field(gt=0, lt=100)


ExitCondition = Union[SignalCondition, StopRule]


class Entry(StrictModel):
    # AND semantics: every condition must hold for a candidate to enter.
    all: list[SignalCondition] = Field(min_length=1)


class Exit(StrictModel):
    # OR semantics: the first condition to trigger exits the position, and
    # which one fired is recorded as the trade's reason code.
    any: list[ExitCondition] = Field(min_length=1)

    @model_validator(mode="after")
    def _at_most_one_of_each_stop(self) -> "Exit":
        kinds = [c.rule for c in self.any if isinstance(c, StopRule)]
        dupes = sorted({k for k in kinds if kinds.count(k) > 1})
        if dupes:
            raise ValueError(f"exit.any declares duplicate stop rules: {dupes}")
        return self


class Ranking(StrictModel):
    """Tie-breaker when entry signals exceed open position slots."""

    by: str
    order: Literal["asc", "desc"]


class Sizing(StrictModel):
    """Equal-weight, capped: target weight = min(1/positions_held,
    max_weight_pct), floored at min_weight_pct. With fewer positions than
    max_positions the cap binds and the remainder stays in cash.
    """

    method: Literal["equal_weight"]
    max_positions: int = Field(ge=1, le=50)
    max_weight_pct: float = Field(gt=0, le=100)
    min_weight_pct: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _bounds_feasible(self) -> "Sizing":
        if self.min_weight_pct > self.max_weight_pct:
            raise ValueError(
                f"sizing.min_weight_pct ({self.min_weight_pct}) exceeds "
                f"max_weight_pct ({self.max_weight_pct})"
            )
        full_target = min(100.0 / self.max_positions, self.max_weight_pct)
        if full_target < self.min_weight_pct - 1e-9:
            raise ValueError(
                f"infeasible sizing: at max_positions={self.max_positions} the "
                f"equal-weight target is {full_target:.1f}%, below "
                f"min_weight_pct={self.min_weight_pct}%"
            )
        return self


class Rebalance(StrictModel):
    # monthly = first trading day of each month (no day_of_month knob in MVP).
    frequency: Literal["daily", "weekly", "monthly"]
    day_of_week: Literal["monday", "tuesday", "wednesday", "thursday", "friday"] | None = None

    @model_validator(mode="after")
    def _day_matches_frequency(self) -> "Rebalance":
        if self.frequency == "weekly" and self.day_of_week is None:
            raise ValueError("rebalance.day_of_week is required when frequency is 'weekly'")
        if self.frequency != "weekly" and self.day_of_week is not None:
            raise ValueError(
                f"rebalance.day_of_week is only valid for weekly frequency, "
                f"not {self.frequency!r}"
            )
        return self


class Execution(StrictModel):
    """Fixed in MVP: signals computed on day T's close, fills at day T+1's
    open plus slippage. Declared as Literals so any other value is rejected
    loudly rather than silently approximated.
    """

    signal_timing: Literal["close"] = "close"
    fill_timing: Literal["next_open"] = "next_open"


class Costs(StrictModel):
    # Costs are ALWAYS applied to simulated fills; these defaults are used
    # when the block is omitted and are echoed into every run record so the
    # assumption is visible later.
    slippage_bps: float = Field(default=5.0, ge=0)
    commission_per_share: float = Field(default=0.0, ge=0)
    commission_per_trade: float = Field(default=0.0, ge=0)


class StrategyConfig(StrictModel):
    schema_version: Literal[1]
    meta: Meta
    universe: Universe
    signals: dict[str, SignalDef] = Field(min_length=1)
    entry: Entry
    exit: Exit
    ranking: Ranking
    sizing: Sizing
    rebalance: Rebalance
    execution: Execution = Execution()
    costs: Costs = Costs()

    @model_validator(mode="after")
    def _validate_references(self) -> "StrategyConfig":
        problems: list[str] = []

        for name in self.signals:
            if not _SIGNAL_NAME_RE.match(name):
                problems.append(
                    f"signal name {name!r} must be lower_snake_case starting with a letter"
                )

        declared = set(self.signals)

        def check(path: str, signal: str) -> None:
            if signal not in declared:
                problems.append(
                    f"{path} references undefined signal {signal!r} "
                    f"(declared: {sorted(declared)})"
                )

        for i, cond in enumerate(self.entry.all):
            check(f"entry.all[{i}]", cond.signal)
        for i, cond in enumerate(self.exit.any):
            if isinstance(cond, SignalCondition):
                check(f"exit.any[{i}]", cond.signal)
        check("ranking.by", self.ranking.by)

        if problems:
            raise ValueError("; ".join(problems))
        return self

    def benchmarks(self) -> list[str]:
        """All benchmark tickers referenced by signals (needs price data too)."""
        return sorted({s.benchmark for s in self.signals.values()})

    def max_lookback_days(self) -> int:
        return max(s.lookback_days for s in self.signals.values())


# --------------------------------------------------------------------------
# Run config (how to evaluate a strategy — kept separate from what to trade)
# --------------------------------------------------------------------------


class WalkForward(StrictModel):
    """Rolling out-of-sample evaluation with FIXED parameters (no
    optimization). train_months is indicator warm-up only: price data is
    fetched from `start - train_months` (plus lookback buffer) so that test
    windows tile [start, end] completely. Stats are reported per test window.
    """

    train_months: int = Field(ge=1, le=60)
    test_months: int = Field(ge=1, le=24)
    step_months: int = Field(ge=1, le=24)

    @model_validator(mode="after")
    def _windows_cover_range(self) -> "WalkForward":
        if self.step_months > self.test_months:
            raise ValueError(
                f"walk_forward.step_months ({self.step_months}) > test_months "
                f"({self.test_months}) would leave uncovered gaps between windows"
            )
        return self


class RunSpec(StrictModel):
    start: date
    end: date
    initial_capital: float = Field(gt=0)
    walk_forward: WalkForward

    @model_validator(mode="after")
    def _range_is_usable(self) -> "RunSpec":
        if self.end <= self.start:
            raise ValueError(f"run.end ({self.end}) must be after run.start ({self.start})")
        approx_months = (self.end - self.start).days / _DAYS_PER_MONTH
        if approx_months < self.walk_forward.test_months:
            raise ValueError(
                f"run range {self.start}..{self.end} (~{approx_months:.1f} months) is "
                f"shorter than one test window ({self.walk_forward.test_months} months)"
            )
        return self


class RunConfig(StrictModel):
    schema_version: Literal[1]
    run: RunSpec
