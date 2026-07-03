from __future__ import annotations

import numpy as np
import pandas as pd

from home_service_action_verifier.windows import WindowSpec


def window_slice(df: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    if "frame" not in df.columns:
        return df.iloc[0:0]
    return df[(df["frame"] >= window.start_frame) & (df["frame"] <= window.end_frame)]


def object_slice(objects: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    df = window_slice(objects, window)
    if "object_id" in df.columns:
        df = df[df["object_id"] == window.object_id]
    return df


def shoulder_width(skeleton: pd.DataFrame) -> float:
    cols = {"lm11_x", "lm11_y", "lm12_x", "lm12_y"}
    if not cols.issubset(skeleton.columns):
        return 1.0
    dist = np.sqrt(
        (skeleton["lm11_x"] - skeleton["lm12_x"]) ** 2
        + (skeleton["lm11_y"] - skeleton["lm12_y"]) ** 2
    )
    value = float(np.nanmedian(dist))
    if not np.isfinite(value) or value <= 1e-6:
        return 1.0
    return value


def point_distance(df: pd.DataFrame, ax: str, ay: str, bx: str, by: str) -> pd.Series:
    return np.sqrt((df[ax] - df[bx]) ** 2 + (df[ay] - df[by]) ** 2)


def aggregate_series(prefix: str, values: pd.Series | np.ndarray) -> dict[str, float]:
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_last": np.nan,
        }
    return {
        f"{prefix}_min": float(s.min()),
        f"{prefix}_max": float(s.max()),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std(ddof=0)),
        f"{prefix}_last": float(s.iloc[-1]),
    }


def centroid(points: list[list[float]]) -> tuple[float, float]:
    arr = np.asarray(points, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def trunk_center(skeleton: pd.DataFrame) -> pd.DataFrame:
    required = ["lm11_x", "lm11_y", "lm12_x", "lm12_y", "lm23_x", "lm23_y", "lm24_x", "lm24_y"]
    if not set(required).issubset(skeleton.columns):
        return pd.DataFrame({"frame": skeleton.get("frame", pd.Series(dtype=int)), "x": np.nan, "y": np.nan})
    return pd.DataFrame(
        {
            "frame": skeleton["frame"],
            "x": skeleton[["lm11_x", "lm12_x", "lm23_x", "lm24_x"]].mean(axis=1),
            "y": skeleton[["lm11_y", "lm12_y", "lm23_y", "lm24_y"]].mean(axis=1),
        }
    )
