"""Rule-based baseline wrapper."""

from __future__ import annotations

from home_service_action_verifier.rule_engine import run_rule_based
from home_service_action_verifier.schemas import DetectionResult, EventToken, WorkOrder


def run(events: list[EventToken], work_order: WorkOrder) -> list[DetectionResult]:
    return run_rule_based(events, work_order)
