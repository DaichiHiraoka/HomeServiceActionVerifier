"""Token-only baseline for event-level comparisons.

This baseline intentionally ignores WorkOrder context. It only uses generic
EventToken fields, so it can show what cannot be distinguished without the
task-specific permission context.
"""

from __future__ import annotations

from home_service_action_verifier.rule_engine import label_from_score
from home_service_action_verifier.schemas import DetectionResult, EventToken, RuleWeights, WorkOrder

RESIDENT_OBJECT_ACTIONS = {"pick_up", "move", "hold", "open"}


def run(
    events: list[EventToken],
    work_order: WorkOrder,
    weights: RuleWeights | None = None,
) -> list[DetectionResult]:
    """Run token-only scoring.

    `work_order` is accepted for CLI compatibility but is deliberately unused.
    """

    results: list[DetectionResult] = []
    _ = work_order
    weights = weights or RuleWeights()
    for event in events:
        score = 0.0
        reasons: list[str] = []
        evidence: dict[str, object] = {"input": "event_token_only"}

        if event.action == "initial_state":
            pass

        if (
            event.action == "place_into_container"
            and event.object_owner is not None
            and event.container_owner is not None
            and event.object_owner != event.container_owner
        ):
            score += 0.8
            reasons.append("物体と収納先の所有者が異なる")
            evidence["owner_mismatch"] = {
                "object_owner": event.object_owner,
                "container_owner": event.container_owner,
            }

        if event.action in RESIDENT_OBJECT_ACTIONS and event.object_owner == "resident":
            score += 0.3
            reasons.append("住人所有物を操作している可能性がある")
            evidence["resident_object_action"] = event.action

        if event.action == "photograph":
            score += 0.3
            reasons.append("作業票を参照しないため撮影対象の許可文脈は判定できない")
            evidence["photo_context"] = "unknown_without_work_order"

        final_score = min(score, 1.0)
        results.append(
            DetectionResult(
                event_id=event.event_id,
                method="token_only",
                predicted_label=label_from_score(final_score, weights),
                suspicion_score=final_score,
                reasons=reasons,
                evidence=evidence,
            )
        )
    return results
