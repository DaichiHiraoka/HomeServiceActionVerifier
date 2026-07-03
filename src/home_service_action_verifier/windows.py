from __future__ import annotations

from dataclasses import dataclass

from .schemas import AnnotationJson


@dataclass(frozen=True)
class WindowSpec:
    seq_id: str
    actor: str
    scenario: str
    label: str
    contact_id: str
    object_id: str
    start_frame: int
    end_frame: int
    contact_start_frame: int
    contact_end_frame: int
    ctx_sec: float
    t_end_rel: float


def build_contact_windows(
    annotation: AnnotationJson,
    ctx_seconds: list[float],
    truncate_seconds: list[float] | None = None,
) -> list[WindowSpec]:
    fps = float(annotation["fps"])
    windows: list[WindowSpec] = []
    for i, contact in enumerate(annotation["contacts"]):
        for ctx_sec in ctx_seconds:
            ctx_frames = int(round(ctx_sec * fps))
            full_start = max(0, int(contact["start_frame"]) - ctx_frames)
            full_end = int(contact["end_frame"]) + ctx_frames
            if truncate_seconds is None:
                windows.append(
                    _make_window(annotation, contact, i, full_start, full_end, ctx_sec, float("inf"))
                )
                continue
            for t_end_rel in truncate_seconds:
                truncated_end = int(round(int(contact["start_frame"]) + t_end_rel * fps))
                end_frame = min(full_end, max(full_start, truncated_end))
                windows.append(
                    _make_window(
                        annotation,
                        contact,
                        i,
                        full_start,
                        end_frame,
                        ctx_sec,
                        float(t_end_rel),
                    )
                )
    return windows


def _make_window(
    annotation: AnnotationJson,
    contact: dict[str, object],
    contact_index: int,
    start_frame: int,
    end_frame: int,
    ctx_sec: float,
    t_end_rel: float,
) -> WindowSpec:
    return WindowSpec(
        seq_id=annotation["seq_id"],
        actor=annotation["actor"],
        scenario=annotation["scenario"],
        label=annotation["label"],
        contact_id=f"{annotation['seq_id']}:{contact_index}",
        object_id=str(contact["object_id"]),
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        contact_start_frame=int(contact["start_frame"]),
        contact_end_frame=int(contact["end_frame"]),
        ctx_sec=float(ctx_sec),
        t_end_rel=t_end_rel,
    )
