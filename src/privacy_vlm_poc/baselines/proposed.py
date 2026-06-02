"""Initial proposed event-level method.

The first implementation uses Rule-Based results directly. For ambiguous
events, VLM evidence can later be merged with:
final_score = max(rule_score, 0.7 * vlm_score)
This max rule keeps high-confidence work-order violations from being diluted.
"""

from __future__ import annotations

from privacy_vlm_poc.rule_engine import label_from_score, run_rule_based
from privacy_vlm_poc.schemas import DetectionResult, EventToken, WorkOrder


def run(events: list[EventToken], work_order: WorkOrder) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    for result in run_rule_based(events, work_order):
        reasons = list(result.reasons)
        evidence = dict(result.evidence)
        if 0.2 <= result.suspicion_score <= 0.8:
            reasons.append("曖昧イベントとしてVLM確認対象だが、初期実装ではRule-Basedスコアを採用")
            evidence["proposed_vlm_status"] = "not_connected"
        results.append(
            DetectionResult(
                event_id=result.event_id,
                method="proposed",
                predicted_label=label_from_score(result.suspicion_score),
                suspicion_score=result.suspicion_score,
                reasons=reasons,
                evidence=evidence,
            )
        )
    return results
