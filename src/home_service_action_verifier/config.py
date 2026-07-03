from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PoseConfig:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    max_gap_frames: int = 15


@dataclass(frozen=True)
class WindowsConfig:
    ctx_seconds: list[float] = field(default_factory=lambda: [0, 1, 3, 5, 10])
    contact_source: str = "gt"


@dataclass(frozen=True)
class FeaturesConfig:
    hold_dist_threshold: float = 0.08
    crop_scale: float = 1.5
    crop_max_px: int = 128


@dataclass(frozen=True)
class EvalConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    bootstrap_n: int = 1000
    target_fpr: float = 0.20


@dataclass(frozen=True)
class AppConfig:
    pose: PoseConfig = field(default_factory=PoseConfig)
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    data = load_yaml(path)
    return AppConfig(
        pose=PoseConfig(**data.get("pose", {})),
        windows=WindowsConfig(**data.get("windows", {})),
        features=FeaturesConfig(**data.get("features", {})),
        eval=EvalConfig(**data.get("eval", {})),
    )


def load_conditions(path: str | Path = "configs/conditions.yaml") -> dict[str, list[str]]:
    data = load_yaml(path)
    conditions: dict[str, list[str]] = {}
    for name, spec in data.items():
        if not isinstance(spec, dict) or "groups" not in spec:
            raise ValueError(f"Condition {name} must define groups")
        groups = spec["groups"]
        if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
            raise ValueError(f"Condition {name} groups must be a list of strings")
        conditions[str(name)] = groups
    return conditions


def snapshot_config(config: AppConfig, conditions: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "pose": config.pose.__dict__,
        "windows": config.windows.__dict__,
        "features": config.features.__dict__,
        "eval": config.eval.__dict__,
        "conditions": conditions,
    }
