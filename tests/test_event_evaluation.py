from __future__ import annotations

from home_service_action_verifier.event_evaluation import evaluate_event_predictions
from home_service_action_verifier.schemas import DetectionResult, EventToken


def test_event_evaluation_excludes_review_and_computes_metrics(tmp_path) -> None:
    events = [
        EventToken(event_id="N", start_sec=0, end_sec=1, action="inspect", ground_truth_label="normal"),
        EventToken(event_id="R", start_sec=1, end_sec=2, action="inspect", ground_truth_label="review"),
        EventToken(event_id="S", start_sec=2, end_sec=3, action="open", ground_truth_label="suspicious"),
        EventToken(
            event_id="H",
            start_sec=3,
            end_sec=4,
            action="pick_up",
            ground_truth_label="high_risk",
            same_action_pair_id="context_pair",
        ),
        EventToken(
            event_id="N2",
            start_sec=4,
            end_sec=5,
            action="pick_up",
            ground_truth_label="normal",
            same_action_pair_id="context_pair",
        ),
    ]
    results = [
        DetectionResult(event_id="N", method="test", predicted_label="normal", suspicion_score=0.0),
        DetectionResult(event_id="R", method="test", predicted_label="review", suspicion_score=0.3),
        DetectionResult(event_id="S", method="test", predicted_label="suspicious", suspicion_score=0.7),
        DetectionResult(event_id="H", method="test", predicted_label="high_risk", suspicion_score=1.0),
        DetectionResult(event_id="N2", method="test", predicted_label="normal", suspicion_score=0.0),
    ]

    metrics = evaluate_event_predictions(events, results, output_dir=tmp_path, review_policy="exclude")

    assert metrics.num_events == 4
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.average_precision == 1.0
    assert metrics.same_action_different_context_accuracy == 1.0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "per_event.csv").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "summary.md").exists()
