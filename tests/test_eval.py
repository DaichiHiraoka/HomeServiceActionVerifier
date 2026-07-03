from __future__ import annotations

import pandas as pd

from home_service_action_verifier.evaluate import compute_metrics, evaluate_lopo


def test_compute_metrics_perfect_prediction_has_f1_one() -> None:
    y = pd.Series(["normal", "unnatural", "normal", "unnatural"])
    metrics = compute_metrics(y, y.copy(), pd.Series([0.1, 0.9, 0.2, 0.8]))
    assert metrics["f1_unnatural"] == 1.0
    assert metrics["fpr"] == 0.0


def test_evaluate_lopo_keeps_actor_groups_out_of_training() -> None:
    dataset = pd.DataFrame(
        [
            _row("s1", "a1", "normal", 0.0),
            _row("s2", "a1", "unnatural", 1.0),
            _row("s3", "a2", "normal", 0.1),
            _row("s4", "a2", "unnatural", 0.9),
            _row("s5", "a3", "normal", 0.2),
            _row("s6", "a3", "unnatural", 0.8),
        ]
    )
    result = evaluate_lopo(dataset, "rf", seed=0)
    per_fold = result["per_fold"]
    assert set(per_fold["actor"]) == {"a1", "a2", "a3"}
    assert len(result["predictions"]) == len(dataset)


def _row(seq_id: str, actor: str, label: str, value: float) -> dict[str, object]:
    return {
        "seq_id": seq_id,
        "actor": actor,
        "scenario": "synthetic",
        "label": label,
        "contact_id": f"{seq_id}:0",
        "ctx_sec": 5.0,
        "t_end_rel": float("inf"),
        "SKEL_SEQ__value": value,
        "OBJ__value": value,
        "REL__value": value,
        "CTX__value": value,
    }
