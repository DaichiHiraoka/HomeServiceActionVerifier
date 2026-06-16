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
    assert metrics.roc_auc is None
    assert metrics.average_precision is None
    assert metrics.same_action_different_context_accuracy == 1.0
    assert metrics.same_action_different_context_binary_accuracy == 1.0
    assert metrics.num_review_predictions == 1
    assert metrics.review_rate == 0.2
    assert metrics.num_events_without_prediction == 0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "per_event.csv").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "summary.md").exists()


def test_same_action_binary_accuracy_separates_detection_from_severity() -> None:
    events = [
        EventToken(
            event_id="P",
            start_sec=0,
            end_sec=1,
            action="photograph",
            ground_truth_label="suspicious",
            same_action_pair_id="photo_context",
        ),
        EventToken(
            event_id="N",
            start_sec=1,
            end_sec=2,
            action="photograph",
            ground_truth_label="normal",
            same_action_pair_id="photo_context",
        ),
    ]
    results = [
        DetectionResult(event_id="P", method="test", predicted_label="high_risk", suspicion_score=0.9),
        DetectionResult(event_id="N", method="test", predicted_label="normal", suspicion_score=0.0),
    ]

    metrics = evaluate_event_predictions(events, results, review_policy="exclude")

    assert metrics.same_action_different_context_accuracy == 0.5
    assert metrics.same_action_different_context_binary_accuracy == 1.0


def test_review_rate_and_missing_prediction_are_reported() -> None:
    events = [
        EventToken(event_id="A", start_sec=0, end_sec=1, action="inspect", ground_truth_label="normal"),
        EventToken(event_id="B", start_sec=1, end_sec=2, action="inspect", ground_truth_label="suspicious"),
        EventToken(event_id="C", start_sec=2, end_sec=3, action="inspect", ground_truth_label="normal"),
    ]
    results = [
        DetectionResult(event_id="A", method="test", predicted_label="review", suspicion_score=0.3),
        DetectionResult(event_id="B", method="test", predicted_label="suspicious", suspicion_score=0.7),
        DetectionResult(event_id="EXTRA", method="test", predicted_label="review", suspicion_score=0.3),
    ]

    metrics = evaluate_event_predictions(events, results, review_policy="exclude")

    assert metrics.num_review_predictions == 1
    assert metrics.review_rate == 0.5
    assert metrics.num_events_without_prediction == 1
    assert metrics.num_predictions_without_event == 1
    assert "warning: 1 event(s) had no prediction" in metrics.notes
    assert "warning: 1 prediction(s) did not match any annotation event" in metrics.notes


def test_missing_prediction_is_reported_but_not_scored_in_same_action_pairs() -> None:
    events = [
        EventToken(
            event_id="P",
            start_sec=0,
            end_sec=1,
            action="photograph",
            ground_truth_label="suspicious",
            same_action_pair_id="photo_context",
        ),
        EventToken(
            event_id="N",
            start_sec=1,
            end_sec=2,
            action="photograph",
            ground_truth_label="normal",
            same_action_pair_id="photo_context",
        ),
    ]
    results = [
        DetectionResult(event_id="P", method="test", predicted_label="suspicious", suspicion_score=0.7),
    ]

    metrics = evaluate_event_predictions(events, results, review_policy="exclude")

    assert metrics.num_events_without_prediction == 1
    assert metrics.same_action_different_context_accuracy is None
    assert metrics.same_action_different_context_binary_accuracy is None


def test_small_discrete_sample_omits_auc_and_average_precision() -> None:
    events = [
        EventToken(event_id=f"E{i}", start_sec=i, end_sec=i + 1, action="inspect", ground_truth_label=label)
        for i, label in enumerate(["normal", "suspicious", "normal", "high_risk"])
    ]
    results = [
        DetectionResult(event_id="E0", method="test", predicted_label="normal", suspicion_score=0.0),
        DetectionResult(event_id="E1", method="test", predicted_label="suspicious", suspicion_score=0.7),
        DetectionResult(event_id="E2", method="test", predicted_label="normal", suspicion_score=0.0),
        DetectionResult(event_id="E3", method="test", predicted_label="high_risk", suspicion_score=1.0),
    ]

    metrics = evaluate_event_predictions(events, results, review_policy="exclude")

    assert metrics.roc_auc is None
    assert metrics.average_precision is None
    assert "fewer than 30 binary-scored events" in metrics.notes
