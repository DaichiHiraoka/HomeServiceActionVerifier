from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from home_service_action_verifier.config import FeaturesConfig
from home_service_action_verifier.schemas import AnnotationJson, feature_column
from home_service_action_verifier.windows import WindowSpec

from .common import aggregate_series, centroid, object_slice, trunk_center, window_slice


def compute_context_features(
    skeleton: pd.DataFrame,
    objects: pd.DataFrame,
    annotation: AnnotationJson,
    window: WindowSpec,
    config: FeaturesConfig,
) -> dict[str, float]:
    obj = object_slice(objects, window)
    post_obj = obj[(obj["frame"] >= window.contact_end_frame) & (obj["visible"].astype(bool))]
    visible = obj[obj["visible"].astype(bool)]
    work_x, work_y = centroid(annotation["work_area"])
    exit_x, exit_y = annotation["exit_point"]
    features: dict[str, float] = {}

    if visible.empty:
        features[feature_column("CTX", "obj_return_dist")] = np.nan
        features[feature_column("CTX", "obj_exit_velocity_mean")] = np.nan
    else:
        first = visible.iloc[0]
        last = visible.iloc[-1]
        features[feature_column("CTX", "obj_return_dist")] = float(
            np.sqrt((last["cx"] - first["cx"]) ** 2 + (last["cy"] - first["cy"]) ** 2)
        )
        velocity_source = post_obj if len(post_obj) >= 2 else visible
        if len(velocity_source) >= 2:
            exit_vec = np.array([exit_x - work_x, exit_y - work_y], dtype=float)
            norm = np.linalg.norm(exit_vec) or 1.0
            exit_unit = exit_vec / norm
            delta = velocity_source[["cx", "cy"]].diff().to_numpy(dtype=float)
            proj = delta @ exit_unit * float(annotation["fps"])
            features[feature_column("CTX", "obj_exit_velocity_mean")] = float(np.nanmean(proj))
        else:
            features[feature_column("CTX", "obj_exit_velocity_mean")] = np.nan

    skel = window_slice(skeleton, window)
    trunk = trunk_center(skel)
    post_trunk = trunk[trunk["frame"] >= window.contact_end_frame]
    if post_trunk.empty:
        features[feature_column("CTX", "post_trunk_work_dist_mean")] = np.nan
        features[feature_column("CTX", "post_work_area_return_rate")] = np.nan
        features[feature_column("CTX", "trunk_exit_velocity_mean")] = np.nan
    else:
        dist = np.sqrt((post_trunk["x"] - work_x) ** 2 + (post_trunk["y"] - work_y) ** 2)
        features.update({feature_column("CTX", k): v for k, v in aggregate_series("post_trunk_work_dist", dist).items()})
        points = post_trunk[["x", "y"]].to_numpy(dtype=float)
        inside = MplPath(np.asarray(annotation["work_area"], dtype=float)).contains_points(points)
        features[feature_column("CTX", "post_work_area_return_rate")] = float(inside.mean())
        if len(post_trunk) >= 2:
            exit_vec = np.array([exit_x - work_x, exit_y - work_y], dtype=float)
            norm = np.linalg.norm(exit_vec) or 1.0
            delta = post_trunk[["x", "y"]].diff().to_numpy(dtype=float)
            features[feature_column("CTX", "trunk_exit_velocity_mean")] = float(
                np.nanmean((delta @ (exit_vec / norm)) * float(annotation["fps"]))
            )
        else:
            features[feature_column("CTX", "trunk_exit_velocity_mean")] = np.nan
    if post_obj.empty:
        features[feature_column("CTX", "post_obj_visible_rate")] = 0.0
    else:
        features[feature_column("CTX", "post_obj_visible_rate")] = float(post_obj["visible"].mean())
    return features
