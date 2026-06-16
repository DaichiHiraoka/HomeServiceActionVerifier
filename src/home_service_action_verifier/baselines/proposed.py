"""Initial proposed event-level method.

The first implementation uses Rule-Based results directly. For ambiguous
events, VLM evidence can later be merged with:
final_score = max(rule_score, 0.7 * vlm_score)
This max rule keeps high-confidence work-order violations from being diluted.
"""

from __future__ import annotations

from home_service_action_verifier.rule_engine import label_from_score, run_rule_based
from home_service_action_verifier.schemas import DetectionResult, EventToken, RuleWeights, WorkOrder


def run(events: list[EventToken], work_order: WorkOrder, weights: RuleWeights | None = None) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    weights = weights or RuleWeights()
    for result in run_rule_based(events, work_order, weights=weights):
        reasons = list(result.reasons)
        evidence = dict(result.evidence)
        if weights.review_threshold <= result.suspicion_score <= weights.high_risk_threshold:
            reasons.append("曖昧イベントとしてVLM確認対象だが、初期実装ではRule-Basedスコアを採用")
            evidence["proposed_vlm_status"] = "not_connected"
        results.append(
            DetectionResult(
                event_id=result.event_id,
                method="proposed",
                predicted_label=label_from_score(result.suspicion_score, weights),
                suspicion_score=result.suspicion_score,
                reasons=reasons,
                evidence=evidence,
            )
        )
    return results
