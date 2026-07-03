from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from home_service_action_verifier.config import FeaturesConfig
from home_service_action_verifier.schemas import AnnotationJson, feature_column
from home_service_action_verifier.windows import WindowSpec

from .common import object_slice

HOG_COMPONENTS = 36
HOG_TARGET_PX = 32
MAX_CROP_FRAMES = 8


def compute_crop_features(
    skeleton: pd.DataFrame,
    objects: pd.DataFrame,
    annotation: AnnotationJson,
    window: WindowSpec,
    config: FeaturesConfig,
) -> dict[str, float]:
    obj = object_slice(objects, window)
    visible = obj[obj["visible"].astype(bool)] if not obj.empty else obj
    features = _empty_hog_features()
    if visible.empty:
        features.update(
            {
                feature_column("CROP", "available_rate"): 0.0,
                feature_column("CROP", "crop_area_mean"): float("nan"),
                feature_column("CROP", "video_readable"): 0.0,
            }
        )
        return features
    crop_area = visible["w"] * visible["h"] * config.crop_scale * config.crop_scale
    features.update(
        {
            feature_column("CROP", "available_rate"): float(len(visible) / len(obj)),
            feature_column("CROP", "crop_area_mean"): float(crop_area.mean()),
        }
    )
    hog_vectors = _read_crop_hog_vectors(annotation, visible, config)
    features[feature_column("CROP", "video_readable")] = 1.0 if hog_vectors else 0.0
    if not hog_vectors:
        return features
    arr = np.vstack(hog_vectors)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    for i, value in enumerate(means):
        features[feature_column("CROP", f"hog_{i:03d}_mean")] = float(value)
    for i, value in enumerate(stds):
        features[feature_column("CROP", f"hog_{i:03d}_std")] = float(value)
    return features


def _read_crop_hog_vectors(
    annotation: AnnotationJson, visible: pd.DataFrame, config: FeaturesConfig
) -> list[np.ndarray]:
    import cv2
    from skimage.feature import hog

    video_path = Path(annotation["video"])
    if not video_path.exists():
        return []
    frame_rows = _sample_visible_rows(visible)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    vectors: list[np.ndarray] = []
    target_px = min(HOG_TARGET_PX, int(config.crop_max_px))
    try:
        for _, row in frame_rows.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
            ok, image = cap.read()
            if not ok:
                continue
            crop = _crop_bbox(image, row, config.crop_scale)
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (target_px, target_px), interpolation=cv2.INTER_AREA)
            vector = hog(
                resized,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(1, 1),
                feature_vector=True,
            )
            if len(vector) == HOG_COMPONENTS:
                vectors.append(vector.astype(float))
    finally:
        cap.release()
    return vectors


def _sample_visible_rows(visible: pd.DataFrame) -> pd.DataFrame:
    if len(visible) <= MAX_CROP_FRAMES:
        return visible
    positions = np.linspace(0, len(visible) - 1, MAX_CROP_FRAMES).round().astype(int)
    return visible.iloc[positions]


def _crop_bbox(image: np.ndarray, row: pd.Series, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    box_w = float(row["w"]) * scale
    box_h = float(row["h"]) * scale
    cx = float(row["cx"])
    cy = float(row["cy"])
    x0 = max(0, int(round((cx - box_w / 2) * width)))
    x1 = min(width, int(round((cx + box_w / 2) * width)))
    y0 = max(0, int(round((cy - box_h / 2) * height)))
    y1 = min(height, int(round((cy + box_h / 2) * height)))
    return image[y0:y1, x0:x1]


def _empty_hog_features() -> dict[str, float]:
    features: dict[str, float] = {}
    for i in range(HOG_COMPONENTS):
        features[feature_column("CROP", f"hog_{i:03d}_mean")] = float("nan")
        features[feature_column("CROP", f"hog_{i:03d}_std")] = float("nan")
    return features
