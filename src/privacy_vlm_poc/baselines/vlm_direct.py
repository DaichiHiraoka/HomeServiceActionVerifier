"""VLM-direct baseline placeholder for event-level comparisons.

This module intentionally avoids sending raw video externally. The initial
implementation returns review-level placeholders until frame selection for
individual event windows is wired into the experiment runner.
"""

from __future__ import annotations

from privacy_vlm_poc.schemas import DetectionResult, EventToken

VLM_EVENT_PROMPT = """許可外行動の疑いだけをイベント単位で評価してください。犯罪や盗難を断定しないでください。
返答は event_id, predicted_label, suspicion_score, reasons, visible_evidence, limitations を含むJSONのみです。"""


def run_stub(events: list[EventToken], method: str) -> list[DetectionResult]:
    return [
        DetectionResult(
            event_id=event.event_id,
            method=method,
            predicted_label="review",
            suspicion_score=0.25,
            reasons=["VLM Direct baselineは初期実装では未接続です"],
            evidence={
                "limitations": "イベント単位のフレーム選択とROI切り出しを接続後に有効化します。",
                "safety": "raw videoは外部APIへ送信しません。",
            },
        )
        for event in events
    ]
