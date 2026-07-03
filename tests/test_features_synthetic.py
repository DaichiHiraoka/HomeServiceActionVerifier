from __future__ import annotations

import pandas as pd
import pytest

from home_service_action_verifier.config import FeaturesConfig
from home_service_action_verifier.features import compute_feature_row
from home_service_action_verifier.tracking import interpolate_objects
from home_service_action_verifier.windows import build_contact_windows

from .test_schemas import annotation_dict


def synthetic_skeleton() -> pd.DataFrame:
    rows = []
    for frame in range(11):
        row = {"frame": frame, "time_s": float(frame), "pose_ok": True}
        for i in range(33):
            row[f"lm{i:02d}_x"] = 0.5
            row[f"lm{i:02d}_y"] = 0.5
            row[f"lm{i:02d}_z"] = 0.0
            row[f"lm{i:02d}_vis"] = 1.0
        row["lm11_x"] = 0.45
        row["lm12_x"] = 0.55
        row["lm23_x"] = 0.46
        row["lm24_x"] = 0.54
        row["lm15_x"] = 0.40 + 0.03 * min(frame, 5)
        row["lm16_x"] = 0.40 + 0.03 * min(frame, 5)
        row["lm27_y"] = 0.9
        row["lm28_y"] = 0.9
        rows.append(row)
    return pd.DataFrame(rows)


def test_context_return_distance_is_derived_from_geometry() -> None:
    annotation = annotation_dict()
    annotation["fps"] = 1.0
    annotation["work_area"] = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
    annotation["exit_point"] = [0.9, 0.5]
    annotation["objects"][0]["keyframes"] = [
        {"frame": 0, "bbox": [0.4, 0.5, 0.1, 0.1]},
        {"frame": 5, "bbox": [0.7, 0.5, 0.1, 0.1]},
        {"frame": 10, "bbox": [0.4, 0.5, 0.1, 0.1]},
    ]
    annotation["contacts"][0]["start_frame"] = 2
    annotation["contacts"][0]["end_frame"] = 6
    objects = interpolate_objects(annotation)
    window = build_contact_windows(annotation, [4])[0]
    row = compute_feature_row(synthetic_skeleton(), objects, annotation, window, FeaturesConfig())
    assert row["CTX__obj_return_dist"] == pytest.approx(0.0)
    assert row["CTX__obj_exit_velocity_mean"] < 0
