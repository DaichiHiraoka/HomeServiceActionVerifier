from __future__ import annotations

from home_service_action_verifier.baselines import token_only
from home_service_action_verifier.rule_engine import run_rule_based
from home_service_action_verifier.scenario import load_work_order
from home_service_action_verifier.schemas import EventToken, RuleWeights


def _work_order():
    return load_work_order("configs/scenarios/router_repair.json")


def test_photograph_targets_get_same_token_only_score() -> None:
    events = [
        EventToken(
            event_id="PHOTO_OK",
            start_sec=0,
            end_sec=1,
            action="photograph",
            zone="router_shelf",
            object_class="router_label",
            target_object="router_label",
        ),
        EventToken(
            event_id="PHOTO_PRIVATE",
            start_sec=1,
            end_sec=2,
            action="photograph",
            zone="private_desk",
            object_class="document",
            target_object="document",
        ),
    ]

    results = token_only.run(events, _work_order())

    assert [result.suspicion_score for result in results] == [0.3, 0.3]
    assert [result.predicted_label for result in results] == ["review", "review"]


def test_rule_based_and_token_only_diverge_on_allowed_photo_context() -> None:
    event = EventToken(
        event_id="PHOTO_OK",
        start_sec=0,
        end_sec=1,
        action="photograph",
        zone="router_shelf",
        object_class="router_label",
        target_object="router_label",
    )
    work_order = _work_order()

    rule_result = run_rule_based([event], work_order)[0]
    token_result = token_only.run([event], work_order)[0]

    assert rule_result.predicted_label == "normal"
    assert token_result.predicted_label == "review"


def test_owner_mismatch_container_is_high_risk_without_work_order() -> None:
    event = EventToken(
        event_id="BAG",
        start_sec=0,
        end_sec=1,
        action="place_into_container",
        object_class="key",
        object_owner="resident",
        container_class="tool_bag",
        container_owner="worker",
    )

    result = token_only.run([event], _work_order())[0]

    assert result.predicted_label == "high_risk"
    assert result.suspicion_score == 0.8


def test_resident_object_action_is_review_without_work_order() -> None:
    event = EventToken(
        event_id="PICK_UP",
        start_sec=0,
        end_sec=1,
        action="pick_up",
        object_class="key",
        object_owner="resident",
    )

    result = token_only.run([event], _work_order())[0]

    assert result.suspicion_score == 0.3
    assert result.predicted_label == "review"


def test_token_only_uses_custom_thresholds_without_using_work_order_context() -> None:
    event = EventToken(
        event_id="PHOTO",
        start_sec=0,
        end_sec=1,
        action="photograph",
        target_object="router_label",
    )
    weights = RuleWeights(
        review_threshold=0.1,
        suspicious_threshold=0.25,
        high_risk_threshold=0.9,
    )

    result = token_only.run([event], _work_order(), weights=weights)[0]

    assert result.suspicion_score == 0.3
    assert result.predicted_label == "suspicious"


def test_token_only_does_not_depend_on_work_order_context() -> None:
    event = EventToken(
        event_id="PHOTO",
        start_sec=0,
        end_sec=1,
        action="photograph",
        zone="router_shelf",
        object_class="router_label",
        target_object="router_label",
    )
    work_order = _work_order()
    changed_work_order = work_order.model_copy(
        update={
            "allowed_photo_targets": [],
            "forbidden_zones": ["router_shelf"],
            "allowed_actions": [],
        }
    )

    baseline = token_only.run([event], work_order)[0]
    changed = token_only.run([event], changed_work_order)[0]

    assert baseline.model_dump(mode="json") == changed.model_dump(mode="json")
