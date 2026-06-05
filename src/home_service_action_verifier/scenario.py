"""Scenario, zone, and event-token loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from home_service_action_verifier.schemas import EventToken, WorkOrder, ZoneConfig

ALLOWED_EVENT_LABELS = {"normal", "review", "suspicious", "high_risk"}


def _read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {json_path}: {exc}"
        raise ValueError(msg) from exc


def load_work_order(path: str | Path) -> WorkOrder:
    try:
        return WorkOrder.model_validate(_read_json(path))
    except ValidationError as exc:
        msg = f"Invalid work order file: {path}"
        raise ValueError(msg) from exc


def load_zone_config(path: str | Path) -> ZoneConfig:
    try:
        return ZoneConfig.model_validate(_read_json(path))
    except ValidationError as exc:
        msg = f"Invalid zone config file: {path}"
        raise ValueError(msg) from exc


def _event_payload_from_jsonl(line: str, line_number: int, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSONL at {path}:{line_number}: {exc}"
        raise ValueError(msg) from exc
    if "label" in payload and "ground_truth_label" not in payload:
        payload["ground_truth_label"] = payload.pop("label")
    label = payload.get("ground_truth_label")
    if label is not None and label not in ALLOWED_EVENT_LABELS:
        msg = (
            f"Invalid event label at {path}:{line_number}: {label!r}. "
            f"Allowed labels: {sorted(ALLOWED_EVENT_LABELS)}"
        )
        raise ValueError(msg)
    return payload


def load_event_tokens(path: str | Path) -> list[EventToken]:
    jsonl_path = Path(path)
    events: list[EventToken] = []
    for line_number, raw_line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = _event_payload_from_jsonl(line, line_number, jsonl_path)
        try:
            events.append(EventToken.model_validate(payload))
        except ValidationError as exc:
            msg = f"Invalid event token at {jsonl_path}:{line_number}"
            raise ValueError(msg) from exc
    return events
