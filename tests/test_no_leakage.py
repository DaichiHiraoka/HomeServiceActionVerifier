from __future__ import annotations

import pandas as pd

from home_service_action_verifier.dataset import build_dataset


def test_dataset_selects_only_feature_columns_and_meta_not_roles() -> None:
    features = pd.DataFrame(
        [
            {
                "seq_id": "s01",
                "actor": "a1",
                "scenario": "normal",
                "label": "normal",
                "contact_id": "s01:0",
                "ctx_sec": 5.0,
                "t_end_rel": float("inf"),
                "role": "non_task",
                "label_copy__bad": 1.0,
                "SKEL_SEQ__trunk_speed_mean": 0.1,
                "OBJ__speed_mean": 0.2,
                "REL__wrist_obj_dist_min": 0.3,
                "CTX__obj_return_dist": 0.0,
            }
        ]
    )
    dataset = build_dataset(features, "E", 5)
    assert "role" not in dataset.columns
    assert "label_copy__bad" not in dataset.columns
    assert "label" in dataset.columns
    assert "CTX__obj_return_dist" in dataset.columns


def test_dataset_excludes_truncated_windows_by_default() -> None:
    features = pd.DataFrame(
        [
            _feature_row("s01", float("inf"), 0.1),
            _feature_row("s01", 1.0, 0.9),
        ]
    )
    dataset = build_dataset(features, "E", 5)
    assert len(dataset) == 1
    assert dataset["t_end_rel"].iloc[0] == float("inf")


def _feature_row(seq_id: str, t_end_rel: float, value: float) -> dict[str, object]:
    return {
        "seq_id": seq_id,
        "actor": "a1",
        "scenario": "synthetic",
        "label": "normal",
        "contact_id": f"{seq_id}:0",
        "ctx_sec": 5.0,
        "t_end_rel": t_end_rel,
        "SKEL_SEQ__value": value,
        "OBJ__value": value,
        "REL__value": value,
        "CTX__value": value,
    }
