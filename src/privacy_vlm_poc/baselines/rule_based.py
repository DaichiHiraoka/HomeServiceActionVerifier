"""Rule-based baseline wrapper."""

from __future__ import annotations

from privacy_vlm_poc.rule_engine import run_rule_based
from privacy_vlm_poc.schemas import DetectionResult, EventToken, WorkOrder


def run(events: list[EventToken], work_order: WorkOrder) -> list[DetectionResult]:
    return run_rule_based(events, work_order)
