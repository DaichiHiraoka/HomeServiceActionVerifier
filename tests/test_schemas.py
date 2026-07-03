from __future__ import annotations

import pytest

from home_service_action_verifier.annotation import validate_annotation


def annotation_dict() -> dict:
    return {
        "seq_id": "s01_a1_unnat-carry_01",
        "video": "data/raw/s01/s01_a1_unnat-carry_01.mp4",
        "fps": 30.0,
        "label": "unnatural",
        "scenario": "unnat-carry",
        "actor": "a1",
        "work_area": [[0.55, 0.30], [0.95, 0.30], [0.95, 0.85], [0.55, 0.85]],
        "exit_point": [0.05, 0.50],
        "objects": [
            {
                "object_id": "wallet",
                "role": "non_task",
                "keyframes": [
                    {"frame": 0, "bbox": [0.42, 0.60, 0.06, 0.05]},
                    {"frame": 10, "bbox": [0.50, 0.60, 0.06, 0.05]},
                ],
            }
        ],
        "contacts": [{"object_id": "wallet", "start_frame": 2, "end_frame": 8}],
    }


def test_validate_annotation_accepts_minimal_valid_annotation() -> None:
    validated = validate_annotation(annotation_dict())
    assert validated["seq_id"] == "s01_a1_unnat-carry_01"


def test_validate_annotation_rejects_forbidden_semantic_keys() -> None:
    data = annotation_dict()
    data["objects"][0]["returned"] = False
    with pytest.raises(ValueError, match="forbidden semantic annotation key"):
        validate_annotation(data)
