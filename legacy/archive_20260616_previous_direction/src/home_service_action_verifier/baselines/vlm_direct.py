"""VLM-direct baseline placeholder for event-level comparisons.

This module intentionally avoids sending raw video externally. The initial
implementation raises a clear error until frame selection for individual event
windows is wired into the experiment runner.
"""

from __future__ import annotations

from home_service_action_verifier.schemas import EventToken

VLM_EVENT_PROMPT = """許可外行動の疑いだけをイベント単位で評価してください。犯罪や盗難を断定しないでください。
返答は event_id, predicted_label, suspicion_score, reasons, visible_evidence, limitations を含むJSONのみです。"""


def run_unimplemented(events: list[EventToken], method: str) -> None:
    event_count = len(events)
    msg = (
        f"{method} is not implemented yet for {event_count} event(s). "
        "Connect event-window frame extraction and ROI generation first. "
        "Raw video must not be sent to external APIs."
    )
    raise NotImplementedError(msg)
