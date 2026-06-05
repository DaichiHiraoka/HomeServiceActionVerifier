from __future__ import annotations

from home_service_action_verifier.rule_engine import rule_based_detect
from home_service_action_verifier.scenario import load_work_order
from home_service_action_verifier.schemas import EventToken


def _work_order():
    return load_work_order("configs/scenarios/router_repair.json")


def test_worker_tool_into_bag_is_normal() -> None:
    event = EventToken(
        event_id="T01",
        start_sec=0,
        end_sec=1,
        action="place_into_container",
        zone="work_area",
        object_class="screwdriver",
        object_owner="worker",
        container_class="tool_bag",
        container_owner="worker",
    )

    result = rule_based_detect(event, _work_order())

    assert result.predicted_label == "normal"
    assert result.suspicion_score == 0.0


def test_private_drawer_open_is_suspicious() -> None:
    event = EventToken(
        event_id="T02",
        start_sec=0,
        end_sec=1,
        action="open",
        zone="private_desk",
        object_class="drawer",
        object_owner="resident",
    )

    result = rule_based_detect(event, _work_order())

    assert result.predicted_label == "suspicious"
    assert "許可外エリアで行動している" in result.reasons


def test_private_document_photo_is_suspicious() -> None:
    event = EventToken(
        event_id="T04",
        start_sec=0,
        end_sec=1,
        action="photograph",
        zone="private_desk",
        object_class="document",
        object_owner="resident",
        target_object="document",
    )

    result = rule_based_detect(event, _work_order())

    assert result.predicted_label in {"suspicious", "high_risk"}
    assert "許可されていない対象を撮影している" in result.reasons


def test_resident_key_into_worker_bag_is_high_risk_and_capped() -> None:
    event = EventToken(
        event_id="T03",
        start_sec=0,
        end_sec=1,
        action="place_into_container",
        zone="work_area",
        object_class="key",
        object_owner="resident",
        container_class="tool_bag",
        container_owner="worker",
    )

    result = rule_based_detect(event, _work_order())

    assert result.predicted_label == "high_risk"
    assert result.suspicion_score == 1.0
