from __future__ import annotations

import pytest

from home_service_action_verifier.tracking import interpolate_objects

from .test_schemas import annotation_dict


def test_tracking_linearly_interpolates_keyframes() -> None:
    annotation = annotation_dict()
    annotation["objects"][0]["keyframes"] = [
        {"frame": 0, "bbox": [0.0, 0.0, 0.2, 0.2]},
        {"frame": 10, "bbox": [1.0, 0.5, 0.2, 0.2]},
    ]
    df = interpolate_objects(annotation)
    row = df[df["frame"] == 5].iloc[0]
    assert row["cx"] == pytest.approx(0.5)
    assert row["cy"] == pytest.approx(0.25)
    assert bool(row["visible"]) is True


def test_tracking_null_keyframe_segment_is_invisible() -> None:
    annotation = annotation_dict()
    annotation["contacts"][0]["end_frame"] = 12
    annotation["objects"][0]["keyframes"] = [
        {"frame": 0, "bbox": [0.0, 0.0, 0.2, 0.2]},
        {"frame": 10, "bbox": None},
    ]
    df = interpolate_objects(annotation)
    start = df[df["frame"] == 0].iloc[0]
    assert bool(start["visible"]) is True
    assert bool(df[df["frame"] == 5].iloc[0]["visible"]) is False
    assert int(df["frame"].max()) == 12


def test_tracking_extends_last_visible_bbox_to_max_frame() -> None:
    annotation = annotation_dict()
    annotation["contacts"][0]["end_frame"] = 15
    annotation["objects"][0]["keyframes"] = [
        {"frame": 0, "bbox": [0.1, 0.2, 0.2, 0.2]},
        {"frame": 10, "bbox": [0.4, 0.5, 0.2, 0.2]},
    ]
    df = interpolate_objects(annotation)
    last = df[df["frame"] == 15].iloc[0]
    assert bool(last["visible"]) is True
    assert last["cx"] == pytest.approx(0.4)
    assert last["cy"] == pytest.approx(0.5)
