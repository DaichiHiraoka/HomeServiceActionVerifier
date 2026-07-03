from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .video_io import get_video_fps, iter_video_frames

POSE_LANDMARK_COUNT = 33


def skeleton_columns() -> list[str]:
    cols = ["frame", "time_s"]
    for i in range(POSE_LANDMARK_COUNT):
        cols.extend([f"lm{i:02d}_x", f"lm{i:02d}_y", f"lm{i:02d}_z", f"lm{i:02d}_vis"])
    cols.append("pose_ok")
    return cols


def extract_pose(
    video_path: str | Path,
    out_path: str | Path | None = None,
    max_gap_frames: int = 15,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
) -> pd.DataFrame:
    import cv2
    import mediapipe as mp

    rows: list[dict[str, object]] = []
    pose = mp.solutions.pose.Pose(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
    )
    try:
        for frame, time_s, image in iter_video_frames(video_path):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            row: dict[str, object] = {"frame": frame, "time_s": time_s}
            if result.pose_landmarks is None:
                for i in range(POSE_LANDMARK_COUNT):
                    row.update(
                        {
                            f"lm{i:02d}_x": np.nan,
                            f"lm{i:02d}_y": np.nan,
                            f"lm{i:02d}_z": np.nan,
                            f"lm{i:02d}_vis": np.nan,
                        }
                    )
                row["pose_ok"] = False
            else:
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    row.update(
                        {
                            f"lm{i:02d}_x": float(lm.x),
                            f"lm{i:02d}_y": float(lm.y),
                            f"lm{i:02d}_z": float(lm.z),
                            f"lm{i:02d}_vis": float(lm.visibility),
                        }
                    )
                row["pose_ok"] = True
            rows.append(row)
    finally:
        pose.close()

    df = interpolate_skeleton_gaps(pd.DataFrame(rows, columns=skeleton_columns()), max_gap_frames)
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    return df


def build_empty_skeleton_for_video(video_path: str | Path) -> pd.DataFrame:
    fps = get_video_fps(video_path)
    rows = []
    for frame, _, _ in iter_video_frames(video_path):
        row = {"frame": frame, "time_s": frame / fps, "pose_ok": False}
        for i in range(POSE_LANDMARK_COUNT):
            row.update(
                {
                    f"lm{i:02d}_x": np.nan,
                    f"lm{i:02d}_y": np.nan,
                    f"lm{i:02d}_z": np.nan,
                    f"lm{i:02d}_vis": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=skeleton_columns())


def interpolate_skeleton_gaps(df: pd.DataFrame, max_gap_frames: int = 15) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    coord_cols = [c for c in out.columns if c.startswith("lm")]
    for col in coord_cols:
        original = out[col].copy()
        interpolated = original.interpolate(method="linear", limit_area="inside")
        for start, end in _nan_runs(original):
            if end - start > max_gap_frames:
                interpolated.iloc[start:end] = np.nan
        out[col] = interpolated
    return out


def pose_missing_rate(df: pd.DataFrame) -> float:
    if df.empty or "pose_ok" not in df:
        return 1.0
    return 1.0 - float(df["pose_ok"].mean())


def _nan_runs(series: pd.Series) -> list[tuple[int, int]]:
    is_nan = series.isna().to_numpy()
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(is_nan):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(series)))
    return runs
