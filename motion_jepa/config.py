from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SCENARIOS = ("collide", "contain", "dominoes", "drape", "drop", "link", "roll", "support")
SCENARIO_TO_ID = {name: index for index, name in enumerate(SCENARIOS)}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    config = deepcopy(config)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("paths", "data", "model", "loss", "train", "cache"):
        if section not in config:
            raise ValueError(f"missing config section: {section}")
    data = config["data"]
    model = config["model"]
    future = list(data["future_indices"])
    if len(future) != int(model["horizons"]):
        raise ValueError("future_indices and model.horizons disagree")
    if any(index <= 0 or index >= int(data["input_frames"]) for index in future):
        raise ValueError("future_indices must be inside the sampled clip and exclude frame zero")
    if int(data["height"]) != int(model["grid_height"]) * 16:
        raise ValueError("the refinement head currently requires height == grid_height * 16")
    if int(data["width"]) != int(model["grid_width"]) * 16:
        raise ValueError("the refinement head currently requires width == grid_width * 16")


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `section.key=value` CLI overrides using YAML scalar parsing."""
    result = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value: {override}")
        dotted, raw = override.split("=", 1)
        keys = dotted.split(".")
        target: dict[str, Any] = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"unknown override path: {dotted}")
            target = target[key]
        if keys[-1] not in target:
            raise ValueError(f"unknown override key: {dotted}")
        target[keys[-1]] = yaml.safe_load(raw)
    validate_config(result)
    return result
