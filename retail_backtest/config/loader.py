"""Load and validate YAML configs into typed models.

All failure modes surface as ConfigError with a list of human-readable
problems (one per issue, with the YAML path), so callers — CLI now, API
later — can show them without unpacking pydantic internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from .models import RunConfig, StrategyConfig


class ConfigError(ValueError):
    def __init__(self, source: str, problems: list[str]):
        self.source = source
        self.problems = problems
        bullet_list = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"invalid config {source}:\n{bullet_list}")


class _StrictYamlLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    Vanilla yaml.safe_load silently keeps the last duplicate, which would let
    e.g. a second `sizing:` block override the first without any warning —
    exactly the kind of ambiguity this tool refuses to guess through.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep)


def _load_yaml_mapping(path: Path) -> dict:
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(str(path), [f"cannot read file: {exc}"]) from exc
    try:
        data = yaml.load(text, Loader=_StrictYamlLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(str(path), [f"YAML parse error: {exc}"]) from exc
    if not isinstance(data, dict):
        raise ConfigError(
            str(path), [f"top level must be a mapping, got {type(data).__name__}"]
        )
    return data


def _format_validation_error(exc: ValidationError) -> list[str]:
    problems = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        problems.append(f"{loc}: {err['msg']}")
    return problems


def parse_strategy_config(data: Mapping[str, Any], source: str = "<dict>") -> StrategyConfig:
    try:
        return StrategyConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(source, _format_validation_error(exc)) from exc


def parse_run_config(data: Mapping[str, Any], source: str = "<dict>") -> RunConfig:
    try:
        return RunConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(source, _format_validation_error(exc)) from exc


def load_strategy_config(path: str | Path) -> StrategyConfig:
    path = Path(path)
    return parse_strategy_config(_load_yaml_mapping(path), source=str(path))


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    return parse_run_config(_load_yaml_mapping(path), source=str(path))
