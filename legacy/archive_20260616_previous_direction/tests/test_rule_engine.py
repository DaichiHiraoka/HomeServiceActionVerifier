from __future__ import annotations

from home_service_action_verifier.baselines import proposed
from home_service_action_verifier.rule_engine import rule_based_detect
from home_service_action_verifier.scenario import load_work_order
from home_service_action_verifier.schemas import EventToken, RuleWeights


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


def test_default_rule_weights_match_implicit_defaults() -> None:
    event = EventToken(
        event_id="T05",
        start_sec=0,
        end_sec=1,
        action="open",
        zone="private_desk",
        object_class="drawer",
        object_owner="resident",
    )

    implicit = rule_based_detect(event, _work_order())
    explicit = rule_based_detect(event, _work_order(), weights=RuleWeights())

    assert explicit.suspicion_score == implicit.suspicion_score
    assert explicit.predicted_label == implicit.predicted_label


def test_custom_rule_weights_change_score_and_threshold_label() -> None:
    event = EventToken(
        event_id="T06",
        start_sec=0,
        end_sec=1,
        action="open",
        zone="private_desk",
        object_class="drawer",
        object_owner="resident",
    )
    weights = RuleWeights(
        forbidden_zone=0.1,
        unexpected_action=0.1,
        review_threshold=0.1,
        suspicious_threshold=0.9,
        high_risk_threshold=0.95,
    )

    result = rule_based_detect(event, _work_order(), weights=weights)

    assert result.suspicion_score == 0.2
    assert result.predicted_label == "review"


def test_proposed_ambiguous_band_uses_custom_thresholds() -> None:
    event = EventToken(
        event_id="T07",
        start_sec=0,
        end_sec=1,
        action="open",
        zone="private_desk",
        object_class="drawer",
        object_owner="resident",
    )
    weights = RuleWeights(
        review_threshold=0.1,
        suspicious_threshold=0.25,
        high_risk_threshold=0.5,
    )

    result = proposed.run([event], _work_order(), weights=weights)[0]

    assert result.suspicion_score == 0.65
    assert result.predicted_label == "high_risk"
    assert "proposed_vlm_status" not in result.evidence
