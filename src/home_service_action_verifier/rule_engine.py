"""Rule-based event detector using work-order context."""

from __future__ import annotations

from home_service_action_verifier.schemas import DetectionResult, EventToken, RuleWeights, WorkOrder


RuleAblation = frozenset[str]
RULE_BASED_METHOD_ABLATIONS: dict[str, RuleAblation] = {
    "rule_based_no_owner": frozenset({"owner"}),
    "rule_based_no_zone": frozenset({"zone"}),
    "rule_based_no_photo_target": frozenset({"photo_target"}),
    "rule_based_no_action_allowlist": frozenset({"action_allowlist"}),
}
SUPPORTED_ABLATIONS = frozenset({"owner", "zone", "photo_target", "action_allowlist"})


def label_from_score(score: float, weights: RuleWeights | None = None) -> str:
    weights = weights or RuleWeights()
    if score < weights.review_threshold:
        return "normal"
    if score < weights.suspicious_threshold:
        return "review"
    if score < weights.high_risk_threshold:
        return "suspicious"
    return "high_risk"


def _normalize_ablation(ablation: set[str] | frozenset[str] | None) -> RuleAblation:
    normalized = frozenset(ablation or ())
    unsupported = sorted(normalized - SUPPORTED_ABLATIONS)
    if unsupported:
        msg = f"Unsupported rule ablation flags: {unsupported}. Supported: {sorted(SUPPORTED_ABLATIONS)}"
        raise ValueError(msg)
    return normalized


def _method_name(ablation: RuleAblation, method: str | None) -> str:
    if method:
        return method
    for candidate, candidate_ablation in RULE_BASED_METHOD_ABLATIONS.items():
        if ablation == candidate_ablation:
            return candidate
    if ablation:
        return "rule_based_ablation_" + "_".join(sorted(ablation))
    return "rule_based"


def _is_contextually_allowed(event: EventToken, work_order: WorkOrder, ablation: RuleAblation) -> bool:
    if event.action == "initial_state":
        return True
    if event.action in work_order.allowed_actions:
        return True
    if "photo_target" in ablation and event.action == "photograph":
        return True
    if event.action == "photograph" and event.target_object in work_order.allowed_photo_targets:
        return True
    if "owner" in ablation and event.action == "place_into_container":
        return True
    if (
        event.action == "place_into_container"
        and event.object_owner == "worker"
        and event.container_owner == "worker"
    ):
        return True
    return False


def rule_based_detect(
    event: EventToken,
    work_order: WorkOrder,
    *,
    ablation: set[str] | frozenset[str] | None = None,
    method: str | None = None,
    weights: RuleWeights | None = None,
) -> DetectionResult:
    ablation_flags = _normalize_ablation(ablation)
    weights = weights or RuleWeights()
    score = 0.0
    reasons: list[str] = []
    evidence: dict[str, object] = {}

    if "zone" not in ablation_flags and event.zone in work_order.forbidden_zones:
        score += weights.forbidden_zone
        reasons.append("許可外エリアで行動している")
        evidence["forbidden_zone"] = event.zone

    if "owner" not in ablation_flags and event.object_class in work_order.resident_private_objects:
        score += weights.resident_private_object
        reasons.append("住人所有または私的物体に関わっている")
        evidence["resident_private_object"] = event.object_class

    if (
        "owner" not in ablation_flags
        and event.object_owner == "resident"
        and event.action in {"pick_up", "move", "hold"}
    ):
        score += weights.resident_object_action
        reasons.append("住人所有物を手に取る・移動する動作である")
        evidence["resident_object_action"] = event.action

    if (
        "owner" not in ablation_flags
        and event.action == "place_into_container"
        and event.object_owner == "resident"
        and event.container_owner == "worker"
    ):
        score += weights.resident_into_worker_container
        reasons.append("住人所有物を作業者側のバッグ等に入れている")
        evidence["container"] = event.container_class

    if (
        "photo_target" not in ablation_flags
        and event.action == "photograph"
        and event.target_object not in work_order.allowed_photo_targets
    ):
        score += weights.disallowed_photo_target
        reasons.append("許可されていない対象を撮影している")
        evidence["photo_target"] = event.target_object

    if "action_allowlist" not in ablation_flags and not _is_contextually_allowed(event, work_order, ablation_flags):
        score += weights.unexpected_action
        reasons.append("作業票に含まれない行動である")
        evidence["unexpected_action"] = event.action

    if event.object_class in work_order.high_risk_objects:
        score += weights.high_risk_object
        reasons.append("高リスク物体に関わっている")
        evidence["high_risk_object"] = event.object_class

    final_score = min(score, 1.0)
    if ablation_flags:
        evidence["ablation"] = sorted(ablation_flags)
    return DetectionResult(
        event_id=event.event_id,
        method=_method_name(ablation_flags, method),
        predicted_label=label_from_score(final_score, weights),
        suspicion_score=final_score,
        reasons=reasons,
        evidence=evidence,
    )


def run_rule_based(
    events: list[EventToken],
    work_order: WorkOrder,
    *,
    ablation: set[str] | frozenset[str] | None = None,
    method: str | None = None,
    weights: RuleWeights | None = None,
) -> list[DetectionResult]:
    return [
        rule_based_detect(event, work_order, ablation=ablation, method=method, weights=weights)
        for event in events
    ]
