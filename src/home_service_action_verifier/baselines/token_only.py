"""Token-only baseline for event-level comparisons.

The initial baseline uses the same deterministic rule set as Rule-Based but
emits `method=token_only`. It gives the comparison table a stable slot for
future token-only heuristics that intentionally ignore visual evidence.
"""

from __future__ import annotations

from home_service_action_verifier.rule_engine import label_from_score, run_rule_based
from home_service_action_verifier.schemas import DetectionResult, EventToken, WorkOrder


def run(events: list[EventToken], work_order: WorkOrder) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    for result in run_rule_based(events, work_order):
        evidence = dict(result.evidence)
        evidence["input"] = "event_token_only"
        results.append(
            DetectionResult(
                event_id=result.event_id,
                method="token_only",
                predicted_label=label_from_score(result.suspicion_score),
                suspicion_score=result.suspicion_score,
                reasons=list(result.reasons),
                evidence=evidence,
            )
        )
    return results
