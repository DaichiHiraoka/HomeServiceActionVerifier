from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import FORBIDDEN_ANNOTATION_KEYS, LABELS, OBJECT_ROLES, AnnotationJson, ensure_bbox

REQUIRED_KEYS = {
    "seq_id",
    "video",
    "fps",
    "label",
    "scenario",
    "actor",
    "work_area",
    "exit_point",
    "objects",
    "contacts",
}


def read_annotation(path: str | Path) -> AnnotationJson:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return validate_annotation(data)


def write_annotation(annotation: AnnotationJson, path: str | Path) -> None:
    validated = validate_annotation(annotation)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(validated, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_annotation(data: dict[str, Any]) -> AnnotationJson:
    if not isinstance(data, dict):
        raise ValueError("annotation root must be an object")
    _reject_forbidden_keys(data)
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(f"annotation missing required keys: {sorted(missing)}")
    if data["label"] not in LABELS:
        raise ValueError(f"label must be one of {sorted(LABELS)}")
    fps = float(data["fps"])
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not isinstance(data["work_area"], list) or len(data["work_area"]) < 3:
        raise ValueError("work_area must contain at least 3 points")
    for i, point in enumerate(data["work_area"]):
        _ensure_point(point, f"work_area[{i}]")
    _ensure_point(data["exit_point"], "exit_point")

    objects = data["objects"]
    if not isinstance(objects, list) or not objects:
        raise ValueError("objects must be a non-empty list")
    object_ids: set[str] = set()
    for i, obj in enumerate(objects):
        _validate_object(obj, i)
        object_ids.add(obj["object_id"])

    contacts = data["contacts"]
    if not isinstance(contacts, list) or not contacts:
        raise ValueError("contacts must be a non-empty list")
    for i, contact in enumerate(contacts):
        _validate_contact(contact, i, object_ids)

    normalized = dict(data)
    normalized["fps"] = fps
    return normalized  # type: ignore[return-value]


def _reject_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_ANNOTATION_KEYS:
                raise ValueError(f"forbidden semantic annotation key at {path}.{key}")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{i}]")


def _ensure_point(value: Any, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path} must be [x, y]")
    point = [float(value[0]), float(value[1])]
    if not all(0 <= v <= 1 for v in point):
        raise ValueError(f"{path} values must be normalized to [0, 1]")
    return point


def _validate_object(obj: Any, index: int) -> None:
    if not isinstance(obj, dict):
        raise ValueError(f"objects[{index}] must be an object")
    for key in ("object_id", "role", "keyframes"):
        if key not in obj:
            raise ValueError(f"objects[{index}] missing {key}")
    if obj["role"] not in OBJECT_ROLES:
        raise ValueError(f"objects[{index}].role must be one of {sorted(OBJECT_ROLES)}")
    keyframes = obj["keyframes"]
    if not isinstance(keyframes, list) or not keyframes:
        raise ValueError(f"objects[{index}].keyframes must be a non-empty list")
    prev_frame = -1
    for j, keyframe in enumerate(keyframes):
        frame = int(keyframe["frame"])
        if frame < 0 or frame <= prev_frame:
            raise ValueError(f"objects[{index}].keyframes must be sorted by increasing frame")
        ensure_bbox(keyframe.get("bbox"), f"objects[{index}].keyframes[{j}].bbox")
        prev_frame = frame


def _validate_contact(contact: Any, index: int, object_ids: set[str]) -> None:
    if not isinstance(contact, dict):
        raise ValueError(f"contacts[{index}] must be an object")
    object_id = contact.get("object_id")
    if object_id not in object_ids:
        raise ValueError(f"contacts[{index}].object_id not present in objects: {object_id}")
    start = int(contact.get("start_frame", -1))
    end = int(contact.get("end_frame", -1))
    if start < 0 or end < start:
        raise ValueError(f"contacts[{index}] must satisfy 0 <= start_frame <= end_frame")
