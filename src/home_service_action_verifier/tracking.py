from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .annotation import read_annotation
from .schemas import AnnotationJson

OBJECT_COLUMNS = ["frame", "object_id", "cx", "cy", "w", "h", "visible"]


def interpolate_objects(annotation: AnnotationJson) -> pd.DataFrame:
    max_frame = max(contact["end_frame"] for contact in annotation["contacts"])
    for obj in annotation["objects"]:
        max_frame = max(max_frame, max(k["frame"] for k in obj["keyframes"]))

    rows: list[dict[str, object]] = []
    for obj in annotation["objects"]:
        rows.extend(_interpolate_one_object(obj["object_id"], obj["keyframes"], max_frame))
    return pd.DataFrame(rows, columns=OBJECT_COLUMNS)


def track_annotation(annotation_path: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    annotation = read_annotation(annotation_path)
    objects = interpolate_objects(annotation)
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        objects.to_parquet(out, index=False)
    return objects


def _interpolate_one_object(
    object_id: str, keyframes: list[dict[str, object]], max_frame: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sorted_keyframes = sorted(keyframes, key=lambda k: int(k["frame"]))
    if len(sorted_keyframes) == 1:
        only = sorted_keyframes[0]
        for frame in range(int(only["frame"]), max_frame + 1):
            rows.append(_bbox_row(frame, object_id, only.get("bbox")))
        return rows

    for current, nxt in zip(sorted_keyframes[:-1], sorted_keyframes[1:]):
        start = int(current["frame"])
        end = int(nxt["frame"])
        bbox0 = current.get("bbox")
        bbox1 = nxt.get("bbox")
        for frame in range(start, end):
            if bbox0 is None:
                rows.append(_bbox_row(frame, object_id, None))
            elif bbox1 is None:
                # The keyframe itself is still visible; occlusion starts after it.
                rows.append(_bbox_row(frame, object_id, bbox0 if frame == start else None))
            else:
                ratio = 0.0 if end == start else (frame - start) / (end - start)
                values = (1 - ratio) * np.array(bbox0, dtype=float) + ratio * np.array(
                    bbox1, dtype=float
                )
                rows.append(_bbox_row(frame, object_id, values))

    last = sorted_keyframes[-1]
    for frame in range(int(last["frame"]), max_frame + 1):
        rows.append(_bbox_row(frame, object_id, last.get("bbox")))
    return rows


def _bbox_row(frame: int, object_id: str, bbox: object) -> dict[str, object]:
    row: dict[str, object] = {"frame": frame, "object_id": object_id}
    if bbox is None:
        row.update({"cx": np.nan, "cy": np.nan, "w": np.nan, "h": np.nan, "visible": False})
        return row
    values = np.array(bbox, dtype=float)
    row.update(
        {
            "cx": float(values[0]),
            "cy": float(values[1]),
            "w": float(values[2]),
            "h": float(values[3]),
            "visible": True,
        }
    )
    return row
