from __future__ import annotations

import numpy as np
import pandas as pd

from home_service_action_verifier.config import FeaturesConfig
from home_service_action_verifier.schemas import AnnotationJson, feature_column
from home_service_action_verifier.windows import WindowSpec

from .common import aggregate_series, shoulder_width, window_slice


def compute_skeleton_point_features(
    skeleton: pd.DataFrame,
    objects: pd.DataFrame,
    annotation: AnnotationJson,
    window: WindowSpec,
    config: FeaturesConfig,
) -> dict[str, float]:
    center_frame = window.contact_start_frame
    fps = float(annotation["fps"])
    radius = max(1, int(round(0.5 * fps)))
    df = skeleton[(skeleton["frame"] >= center_frame - radius) & (skeleton["frame"] <= center_frame + radius)]
    scale = shoulder_width(skeleton)
    features: dict[str, float] = {}
    if df.empty:
        for name in ("wrist_height", "hip_ankle_gap", "trunk_lean"):
            features.update({feature_column("SKEL_POINT", k): v for k, v in aggregate_series(name, []).items()})
        return features
    wrist_y = df[[c for c in ("lm15_y", "lm16_y") if c in df]].mean(axis=1)
    ankle_y = df[[c for c in ("lm27_y", "lm28_y") if c in df]].mean(axis=1)
    hip_y = df[[c for c in ("lm23_y", "lm24_y") if c in df]].mean(axis=1)
    shoulder_x = df[[c for c in ("lm11_x", "lm12_x") if c in df]].mean(axis=1)
    hip_x = df[[c for c in ("lm23_x", "lm24_x") if c in df]].mean(axis=1)
    derived = {
        "wrist_height": 1.0 - wrist_y,
        "hip_ankle_gap": (ankle_y - hip_y) / scale,
        "trunk_lean": (shoulder_x - hip_x).abs() / scale,
    }
    for name, values in derived.items():
        features.update({feature_column("SKEL_POINT", k): v for k, v in aggregate_series(name, values).items()})
    return features


def compute_skeleton_sequence_features(
    skeleton: pd.DataFrame,
    objects: pd.DataFrame,
    annotation: AnnotationJson,
    window: WindowSpec,
    config: FeaturesConfig,
) -> dict[str, float]:
    df = window_slice(skeleton, window)
    scale = shoulder_width(skeleton)
    features: dict[str, float] = {}
    if df.empty:
        for name in ("trunk_speed", "trunk_step", "crouch_level"):
            features.update({feature_column("SKEL_SEQ", k): v for k, v in aggregate_series(name, []).items()})
        features[feature_column("SKEL_SEQ", "stop_rate")] = np.nan
        features[feature_column("SKEL_SEQ", "crouch_transition_count")] = np.nan
        return features

    trunk_x = df[[c for c in ("lm11_x", "lm12_x", "lm23_x", "lm24_x") if c in df]].mean(axis=1)
    trunk_y = df[[c for c in ("lm11_y", "lm12_y", "lm23_y", "lm24_y") if c in df]].mean(axis=1)
    step = np.sqrt(trunk_x.diff() ** 2 + trunk_y.diff() ** 2) / scale
    speed = step * float(annotation["fps"])
    hip_y = df[[c for c in ("lm23_y", "lm24_y") if c in df]].mean(axis=1)
    ankle_y = df[[c for c in ("lm27_y", "lm28_y") if c in df]].mean(axis=1)
    crouch = (ankle_y - hip_y) / scale
    crouch_state = crouch > float(crouch.median(skipna=True))
    transitions = int(crouch_state.astype("float64").diff().abs().fillna(0).sum())
    for name, values in {
        "trunk_speed": speed,
        "trunk_step": step,
        "crouch_level": crouch,
    }.items():
        features.update({feature_column("SKEL_SEQ", k): v for k, v in aggregate_series(name, values).items()})
    features[feature_column("SKEL_SEQ", "stop_rate")] = float((speed.fillna(0) < 0.02).mean())
    features[feature_column("SKEL_SEQ", "crouch_transition_count")] = float(transitions)
    return features
