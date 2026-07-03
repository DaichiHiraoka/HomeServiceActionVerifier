from __future__ import annotations

import numpy as np
import pandas as pd

from home_service_action_verifier.config import FeaturesConfig
from home_service_action_verifier.schemas import AnnotationJson, feature_column
from home_service_action_verifier.windows import WindowSpec

from .common import aggregate_series, object_slice, shoulder_width, trunk_center, window_slice


def compute_relation_features(
    skeleton: pd.DataFrame,
    objects: pd.DataFrame,
    annotation: AnnotationJson,
    window: WindowSpec,
    config: FeaturesConfig,
) -> dict[str, float]:
    skel = window_slice(skeleton, window)
    obj = object_slice(objects, window)
    features: dict[str, float] = {}
    if skel.empty or obj.empty:
        for name in ("wrist_obj_dist", "wrist_obj_dist_delta", "trunk_obj_dist"):
            features.update({feature_column("REL", k): v for k, v in aggregate_series(name, []).items()})
        features[feature_column("REL", "hold_duration_s")] = np.nan
        return features

    merged = skel.merge(obj[obj["visible"].astype(bool)], on="frame", how="inner")
    scale = shoulder_width(skeleton)
    if merged.empty:
        for name in ("wrist_obj_dist", "wrist_obj_dist_delta", "trunk_obj_dist"):
            features.update({feature_column("REL", k): v for k, v in aggregate_series(name, []).items()})
        features[feature_column("REL", "hold_duration_s")] = 0.0
        return features
    left = np.sqrt((merged["lm15_x"] - merged["cx"]) ** 2 + (merged["lm15_y"] - merged["cy"]) ** 2)
    right = np.sqrt((merged["lm16_x"] - merged["cx"]) ** 2 + (merged["lm16_y"] - merged["cy"]) ** 2)
    wrist_dist = pd.Series(np.minimum(left, right) / scale)
    trunk = trunk_center(skel)
    trunk_merged = trunk.merge(obj[obj["visible"].astype(bool)], on="frame", how="inner")
    trunk_dist = np.sqrt((trunk_merged["x"] - trunk_merged["cx"]) ** 2 + (trunk_merged["y"] - trunk_merged["cy"]) ** 2)
    for name, values in {
        "wrist_obj_dist": wrist_dist,
        "wrist_obj_dist_delta": wrist_dist.diff(),
        "trunk_obj_dist": trunk_dist / scale,
    }.items():
        features.update({feature_column("REL", k): v for k, v in aggregate_series(name, values).items()})
    features[feature_column("REL", "hold_duration_s")] = float(
        (wrist_dist < config.hold_dist_threshold).sum() / float(annotation["fps"])
    )
    return features
