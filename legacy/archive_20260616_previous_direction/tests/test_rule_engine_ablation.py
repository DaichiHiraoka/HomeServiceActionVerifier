from __future__ import annotations

from home_service_action_verifier.cli import main
from home_service_action_verifier.rule_engine import rule_based_detect, run_rule_based
from home_service_action_verifier.scenario import load_event_tokens, load_work_order
from home_service_action_verifier.schemas import EventToken


def _work_order():
    return load_work_order("configs/scenarios/router_repair.json")


def test_empty_ablation_matches_default_scores() -> None:
    events = load_event_tokens("data/real/router_trial_001_annotations.example.jsonl")
    work_order = _work_order()

    default_results = run_rule_based(events, work_order)
    empty_ablation_results = run_rule_based(events, work_order, ablation=frozenset())

    assert [result.suspicion_score for result in empty_ablation_results] == [
        result.suspicion_score for result in default_results
    ]
    assert [result.predicted_label for result in empty_ablation_results] == [
        result.predicted_label for result in default_results
    ]


def test_no_owner_breaks_resident_key_into_worker_bag_high_risk() -> None:
    event = EventToken(
        event_id="KEY_BAG",
        start_sec=0,
        end_sec=1,
        action="place_into_container",
        zone="work_area",
        object_class="key",
        object_owner="resident",
        container_class="tool_bag",
        container_owner="worker",
    )

    result = rule_based_detect(event, _work_order(), ablation=frozenset({"owner"}))

    assert result.suspicion_score < 0.8
    assert result.predicted_label != "high_risk"


def test_no_zone_reduces_private_drawer_open_below_suspicious() -> None:
    event = EventToken(
        event_id="DRAWER",
        start_sec=0,
        end_sec=1,
        action="open",
        zone="private_desk",
        object_class="drawer",
        object_owner="resident",
    )

    result = rule_based_detect(event, _work_order(), ablation=frozenset({"zone"}))

    assert result.suspicion_score < 0.5
    assert result.predicted_label == "review"


def test_no_photo_target_removes_photo_context_score_gap() -> None:
    work_order = _work_order()
    events = [
        EventToken(
            event_id="ROUTER_PHOTO",
            start_sec=0,
            end_sec=1,
            action="photograph",
            zone="work_area",
            object_class="router",
            target_object="router_label",
        ),
        EventToken(
            event_id="DOCUMENT_PHOTO",
            start_sec=1,
            end_sec=2,
            action="photograph",
            zone="work_area",
            object_class="router",
            target_object="document",
        ),
    ]

    default_scores = [result.suspicion_score for result in run_rule_based(events, work_order)]
    ablated_scores = [
        result.suspicion_score
        for result in run_rule_based(events, work_order, ablation=frozenset({"photo_target"}))
    ]

    assert default_scores[1] > default_scores[0]
    assert ablated_scores[0] == ablated_scores[1]


def test_cli_accepts_rule_based_ablation_method(tmp_path) -> None:
    output_dir = tmp_path / "ablation"

    exit_code = main(
        [
            "analyze-scenario",
            "--work-order",
            "configs/scenarios/router_repair.json",
            "--zones",
            "configs/zones/router_repair_zones.json",
            "--annotations",
            "data/real/router_trial_001_annotations.example.jsonl",
            "--method",
            "rule_based_no_owner",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    first_line = (output_dir / "event_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert '"method": "rule_based_no_owner"' in first_line
