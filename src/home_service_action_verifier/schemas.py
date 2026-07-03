from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

LABELS = {"normal", "unnatural"}
OBJECT_ROLES = {"task", "non_task"}
FORBIDDEN_ANNOTATION_KEYS = {
    "returned",
    "is_returned",
    "return_label",
    "return_status",
    "not_returned",
    "restored",
    "is_restored",
    "suspicious_reason",
    "intent",
    "semantic_label",
}

META_COLUMNS = ["seq_id", "actor", "scenario", "label", "contact_id", "ctx_sec", "t_end_rel"]
LABEL_COLUMN = "label"
FEATURE_GROUPS = ["SKEL_POINT", "SKEL_SEQ", "OBJ", "REL", "CTX", "CROP"]


class BBoxKeyframe(TypedDict):
    frame: int
    bbox: list[float] | None


class AnnotatedObject(TypedDict):
    object_id: str
    role: Literal["task", "non_task"]
    keyframes: list[BBoxKeyframe]


class Contact(TypedDict):
    object_id: str
    start_frame: int
    end_frame: int


class AnnotationJson(TypedDict):
    seq_id: str
    video: str
    fps: float
    label: Literal["normal", "unnatural"]
    scenario: str
    actor: str
    work_area: list[list[float]]
    exit_point: list[float]
    objects: list[AnnotatedObject]
    contacts: list[Contact]


@dataclass(frozen=True)
class ValidationErrorDetail:
    path: str
    message: str


def feature_column(group: str, name: str) -> str:
    if group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {group}")
    return f"{group}__{name}"


def is_feature_column(name: str) -> bool:
    return "__" in name and name.split("__", 1)[0] in FEATURE_GROUPS


def ensure_bbox(value: Any, path: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{path} must be null or [cx, cy, w, h]")
    bbox = [float(v) for v in value]
    cx, cy, w, h = bbox
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        raise ValueError(f"{path} values must be normalized to [0, 1]")
    return bbox
