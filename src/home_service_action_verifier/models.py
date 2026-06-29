"""Data models for the current skeleton/object-context verifier."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Thresholds:
    touch_distance: float = 90.0
    return_distance: float = 70.0
    movement_distance: float = 80.0
    body_distance: float = 100.0
    review_threshold: float = 0.25
    suspicious_threshold: float = 0.55
    high_risk_threshold: float = 0.8


@dataclass(frozen=True)
class TaskContext:
    task_name: str
    work_areas: set[str] = field(default_factory=set)
    private_areas: set[str] = field(default_factory=set)
    target_objects: set[str] = field(default_factory=set)
    worker_objects: set[str] = field(default_factory=set)
    private_objects: set[str] = field(default_factory=set)
    high_risk_objects: set[str] = field(default_factory=set)
    thresholds: Thresholds = field(default_factory=Thresholds)


@dataclass(frozen=True)
class SkeletonFrame:
    timestamp: float
    left_wrist_x: float
    left_wrist_y: float
    right_wrist_x: float
    right_wrist_y: float
    torso_x: float
    torso_y: float
    pose_label: str = ""


@dataclass(frozen=True)
class ObjectFrame:
    timestamp: float
    object_id: str
    label: str
    role: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    zone: str = ""
    visible: bool = True
    crop_path: str = ""

    @property
    def center_x(self) -> float:
        return self.bbox_x + self.bbox_w / 2.0

    @property
    def center_y(self) -> float:
        return self.bbox_y + self.bbox_h / 2.0


@dataclass(frozen=True)
class DetectionResult:
    object_id: str
    object_label: str
    role: str
    predicted_label: str
    suspicion_score: float
    first_touch_time: float | None
    first_alert_time: float | None
    time_to_detection: float | None
    reasons: list[str]
    evidence: dict[str, float | str | bool | None]

