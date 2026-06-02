"""Rule-based event detector using work-order context."""

from __future__ import annotations

from privacy_vlm_poc.schemas import DetectionResult, EventToken, WorkOrder


def label_from_score(score: float) -> str:
    if score < 0.2:
        return "normal"
    if score < 0.5:
        return "review"
    if score < 0.8:
        return "suspicious"
    return "high_risk"


def _is_contextually_allowed(event: EventToken, work_order: WorkOrder) -> bool:
    if event.action == "initial_state":
        return True
    if event.action in work_order.allowed_actions:
        return True
    if event.action == "photograph" and event.target_object in work_order.allowed_photo_targets:
        return True
    if (
        event.action == "place_into_container"
        and event.object_owner == "worker"
        and event.container_owner == "worker"
    ):
        return True
    return False


def rule_based_detect(event: EventToken, work_order: WorkOrder) -> DetectionResult:
    score = 0.0
    reasons: list[str] = []
    evidence: dict[str, object] = {}

    if event.zone in work_order.forbidden_zones:
        score += 0.4
        reasons.append("許可外エリアで行動している")
        evidence["forbidden_zone"] = event.zone

    if event.object_class in work_order.resident_private_objects:
        score += 0.4
        reasons.append("住人所有または私的物体に関わっている")
        evidence["resident_private_object"] = event.object_class

    if event.object_owner == "resident" and event.action in {"pick_up", "move", "hold"}:
        score += 0.3
        reasons.append("住人所有物を手に取る・移動する動作である")
        evidence["resident_object_action"] = event.action

    if (
        event.action == "place_into_container"
        and event.object_owner == "resident"
        and event.container_owner == "worker"
    ):
        score += 0.8
        reasons.append("住人所有物を作業者側のバッグ等に入れている")
        evidence["container"] = event.container_class

    if event.action == "photograph" and event.target_object not in work_order.allowed_photo_targets:
        score += 0.5
        reasons.append("許可されていない対象を撮影している")
        evidence["photo_target"] = event.target_object

    if not _is_contextually_allowed(event, work_order):
        score += 0.25
        reasons.append("作業票に含まれない行動である")
        evidence["unexpected_action"] = event.action

    if event.object_class in work_order.high_risk_objects:
        score += 0.2
        reasons.append("高リスク物体に関わっている")
        evidence["high_risk_object"] = event.object_class

    final_score = min(score, 1.0)
    return DetectionResult(
        event_id=event.event_id,
        method="rule_based",
        predicted_label=label_from_score(final_score),
        suspicion_score=final_score,
        reasons=reasons,
        evidence=evidence,
    )


def run_rule_based(events: list[EventToken], work_order: WorkOrder) -> list[DetectionResult]:
    return [rule_based_detect(event, work_order) for event in events]
