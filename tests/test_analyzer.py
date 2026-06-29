from pathlib import Path

from home_service_action_verifier.analyzer import analyze_tracks
from home_service_action_verifier.io import load_object_tracks_csv, load_skeleton_csv, load_task_context


ROOT = Path(__file__).resolve().parents[1]


def test_sample_flags_high_risk_private_object() -> None:
    context = load_task_context(ROOT / "sample_data" / "task_context.json")
    skeleton = load_skeleton_csv(ROOT / "sample_data" / "skeleton.csv")
    objects = load_object_tracks_csv(ROOT / "sample_data" / "object_tracks.csv")

    results = analyze_tracks(skeleton, objects, context)
    by_id = {result.object_id: result for result in results}

    assert by_id["private_1"].predicted_label == "high_risk"
    assert by_id["private_1"].suspicion_score >= 0.8
    assert by_id["private_1"].first_alert_time is not None
    assert by_id["tool_1"].predicted_label == "normal"


def test_result_contains_explainable_reasons() -> None:
    context = load_task_context(ROOT / "sample_data" / "task_context.json")
    skeleton = load_skeleton_csv(ROOT / "sample_data" / "skeleton.csv")
    objects = load_object_tracks_csv(ROOT / "sample_data" / "object_tracks.csv")

    results = analyze_tracks(skeleton, objects, context)
    private_result = next(result for result in results if result.object_id == "private_1")

    reasons = " ".join(private_result.reasons)
    assert "高リスク物体" in reasons
    assert "物体が元の位置に戻っていない" in reasons

