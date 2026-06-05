"""Shared pydantic schemas for the PoC pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SamplingMethod(str, Enum):
    UNIFORM = "uniform"
    MOTION = "motion"
    HYBRID = "hybrid"
    EVENT_WINDOW = "event_window"


class MaskMethod(str, Enum):
    NONE = "none"
    FACE_LIKE_TOP_MASK = "face_like_top_mask"
    BACKGROUND_BLUR_WITH_ROI = "background_blur_with_roi"
    LOWER_BODY_ONLY = "lower_body_only"
    OBJECT_AREA_ONLY = "object_area_only"
    HAND_OBJECT_ROI = "hand_object_roi"
    FULL_FRAME_BLUR_EXCEPT_ROI = "full_frame_blur_except_roi"
    TOKEN_ONLY = "token_only"


class VLMBackend(str, Enum):
    MOCK = "mock"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class ROI(BaseModel):
    """Pixel-space region of interest."""

    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "ROI":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            msg = "ROI requires x2 > x1 and y2 > y1"
            raise ValueError(msg)
        return self

    def clipped(self, width: int, height: int) -> "ROI":
        x1 = min(max(self.x1, 0), max(width - 1, 0))
        y1 = min(max(self.y1, 0), max(height - 1, 0))
        x2 = min(max(self.x2, x1 + 1), width)
        y2 = min(max(self.y2, y1 + 1), height)
        return ROI(x1=x1, y1=y1, x2=x2, y2=y2)


class VideoMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_path: Path
    fps: float = Field(ge=0)
    duration: float = Field(ge=0)
    total_frames: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class FrameInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    path: Path | None = None
    motion_score: float | None = Field(default=None, ge=0)


class VLMResponse(BaseModel):
    unauthorized_object_interaction_suspected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    target_object: str | None
    evidence_frames: list[int] = Field(default_factory=list)
    reason: str
    privacy_sensitive_description_included: bool
    limitations: str

    @field_validator("evidence_frames")
    @classmethod
    def evidence_frames_must_be_non_negative(cls, value: list[int]) -> list[int]:
        if any(frame < 0 for frame in value):
            msg = "evidence_frames must contain non-negative frame indices"
            raise ValueError(msg)
        return value


class AnalyzeConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    video_path: Path
    sampling_method: SamplingMethod
    num_frames: int = Field(gt=0)
    mask_method: MaskMethod
    roi: ROI | None = None
    vlm_backend: VLMBackend
    vlm_model: str | None = None
    resize_width: int | None = Field(default=None, gt=0)


class AnalyzeResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_dir: Path
    metadata: VideoMetadata
    selected_frames: list[FrameInfo]
    masked_frame_paths: list[Path]
    grid_path: Path
    result_path: Path
    report_path: Path
    config_path: Path
    vlm_response: VLMResponse
    processing_time_sec: float = Field(ge=0)


class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    selected_frame_recall: float
    average_num_selected_frames: float
    average_processing_time_sec: float
    num_videos: int
    notes: str
    output_dir: Path | None = None


class WorkOrder(BaseModel):
    scenario_id: str
    task_name: str
    authorized_zones: list[str]
    forbidden_zones: list[str]
    target_objects: list[str]
    worker_owned_objects: list[str]
    resident_private_objects: list[str]
    allowed_actions: list[str]
    forbidden_actions: list[str]
    allowed_photo_targets: list[str] = Field(default_factory=list)
    high_risk_objects: list[str] = Field(default_factory=list)


class ZoneDefinition(BaseModel):
    zone_id: str
    type: str
    bbox: tuple[int, int, int, int]

    @model_validator(mode="after")
    def validate_bbox(self) -> "ZoneDefinition":
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            msg = "ZoneDefinition bbox requires x2 > x1 and y2 > y1"
            raise ValueError(msg)
        return self


class ZoneConfig(BaseModel):
    video_width: int = Field(gt=0)
    video_height: int = Field(gt=0)
    zones: list[ZoneDefinition]


class EventToken(BaseModel):
    event_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    action: str
    zone: str | None = None
    object_class: str | None = None
    object_owner: str | None = None
    container_class: str | None = None
    container_owner: str | None = None
    target_object: str | None = None
    task_stage: str | None = None
    ground_truth_label: str | None = None
    same_action_pair_id: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_time_order(self) -> "EventToken":
        if self.start_sec >= self.end_sec:
            msg = "EventToken requires start_sec < end_sec"
            raise ValueError(msg)
        return self


class DetectionResult(BaseModel):
    event_id: str
    method: str
    predicted_label: str
    suspicion_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class EventEvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
    average_precision: float | None = None
    false_alarm_rate: float
    same_action_different_context_accuracy: float | None = None
    num_events: int
    num_positive_events: int
    num_negative_events: int
    notes: str
    output_dir: Path | None = None


def to_jsonable(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value
