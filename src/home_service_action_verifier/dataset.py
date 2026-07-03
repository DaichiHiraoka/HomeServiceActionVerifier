from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_conditions
from .schemas import FEATURE_GROUPS, LABEL_COLUMN, META_COLUMNS, is_feature_column


def feature_columns_for_groups(features: pd.DataFrame, groups: list[str]) -> list[str]:
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    prefixes = tuple(f"{group}__" for group in groups)
    return [c for c in features.columns if is_feature_column(c) and c.startswith(prefixes)]


def build_dataset(
    features: pd.DataFrame,
    condition: str,
    ctx_sec: float,
    conditions_path: str | Path = "configs/conditions.yaml",
    t_end_rel: float | None = float("inf"),
) -> pd.DataFrame:
    conditions = load_conditions(conditions_path)
    if condition not in conditions:
        raise ValueError(f"unknown condition {condition}; available: {sorted(conditions)}")
    selected = features[features["ctx_sec"].astype(float) == float(ctx_sec)].copy()
    if t_end_rel is not None and "t_end_rel" in selected.columns:
        target = float(t_end_rel)
        if np.isinf(target):
            selected = selected[np.isinf(selected["t_end_rel"].astype(float))]
        else:
            selected = selected[selected["t_end_rel"].astype(float) == target]
    cols = [c for c in META_COLUMNS if c in selected.columns]
    feature_cols = feature_columns_for_groups(selected, conditions[condition])
    if not feature_cols:
        raise ValueError(f"condition {condition} selected no feature columns")
    return selected[cols + feature_cols]


def split_xy(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = [c for c in dataset.columns if is_feature_column(c)]
    X = dataset[feature_cols]
    y = dataset[LABEL_COLUMN]
    groups = dataset["actor"]
    return X, y, groups


def write_dataset(dataset: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out, index=False)
