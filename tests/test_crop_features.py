from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd

from home_service_action_verifier.config import FeaturesConfig
from home_service_action_verifier.features.crop_feats import compute_crop_features
from home_service_action_verifier.tracking import interpolate_objects
from home_service_action_verifier.windows import build_contact_windows

from .test_schemas import annotation_dict


def test_crop_features_read_video_and_emit_hog(tmp_path) -> None:
    video_path = tmp_path / "s01_a1_unnat-carry_01.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 64),
    )
    for i in range(4):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (16 + i, 16), (48, 48), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    annotation = annotation_dict()
    annotation["video"] = str(video_path)
    annotation["fps"] = 5.0
    annotation["objects"][0]["keyframes"] = [
        {"frame": 0, "bbox": [0.5, 0.5, 0.6, 0.6]},
        {"frame": 3, "bbox": [0.5, 0.5, 0.6, 0.6]},
    ]
    annotation["contacts"][0]["start_frame"] = 0
    annotation["contacts"][0]["end_frame"] = 3
    objects = interpolate_objects(annotation)
    window = build_contact_windows(annotation, [0])[0]
    features = compute_crop_features(pd.DataFrame(), objects, annotation, window, FeaturesConfig())
    assert features["CROP__video_readable"] == 1.0
    assert math.isfinite(features["CROP__hog_000_mean"])
