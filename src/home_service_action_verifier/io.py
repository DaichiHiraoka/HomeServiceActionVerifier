"""File loading and result writing for the desktop prototype."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from home_service_action_verifier.models import DetectionResult, ObjectFrame, SkeletonFrame, TaskContext, Thresholds


def _as_set(payload: dict, key: str) -> set[str]:
    values = payload.get(key, [])
    if values is None:
        return set()
    if not isinstance(values, list):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    return {str(item) for item in values}


def _float(row: dict[str, str], key: str, default: float | None = None) -> float:
    value = row.get(key, "")
    if value == "" and default is not None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        msg = f"Invalid float for {key}: {value!r}"
        raise ValueError(msg) from exc


def _bool(row: dict[str, str], key: str, default: bool = True) -> bool:
    value = row.get(key, "")
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "visible"}


def load_task_context(path: str | Path) -> TaskContext:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds_payload = payload.get("thresholds", {}) or {}
    defaults = Thresholds()
    thresholds = Thresholds(
        touch_distance=float(thresholds_payload.get("touch_distance", defaults.touch_distance)),
        return_distance=float(thresholds_payload.get("return_distance", defaults.return_distance)),
        movement_distance=float(thresholds_payload.get("movement_distance", defaults.movement_distance)),
        body_distance=float(thresholds_payload.get("body_distance", defaults.body_distance)),
        review_threshold=float(thresholds_payload.get("review_threshold", defaults.review_threshold)),
        suspicious_threshold=float(thresholds_payload.get("suspicious_threshold", defaults.suspicious_threshold)),
        high_risk_threshold=float(thresholds_payload.get("high_risk_threshold", defaults.high_risk_threshold)),
    )
    return TaskContext(
        task_name=str(payload.get("task_name", "Untitled task")),
        work_areas=_as_set(payload, "work_areas"),
        private_areas=_as_set(payload, "private_areas"),
        target_objects=_as_set(payload, "target_objects"),
        worker_objects=_as_set(payload, "worker_objects"),
        private_objects=_as_set(payload, "private_objects"),
        high_risk_objects=_as_set(payload, "high_risk_objects"),
        thresholds=thresholds,
    )


def load_skeleton_csv(path: str | Path) -> list[SkeletonFrame]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frames = [
        SkeletonFrame(
            timestamp=_float(row, "timestamp"),
            left_wrist_x=_float(row, "left_wrist_x"),
            left_wrist_y=_float(row, "left_wrist_y"),
            right_wrist_x=_float(row, "right_wrist_x"),
            right_wrist_y=_float(row, "right_wrist_y"),
            torso_x=_float(row, "torso_x"),
            torso_y=_float(row, "torso_y"),
            pose_label=row.get("pose_label", ""),
        )
        for row in rows
    ]
    return sorted(frames, key=lambda item: item.timestamp)


def load_object_tracks_csv(path: str | Path) -> list[ObjectFrame]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frames = [
        ObjectFrame(
            timestamp=_float(row, "timestamp"),
            object_id=str(row.get("object_id", "")).strip(),
            label=str(row.get("label", "")).strip(),
            role=str(row.get("role", "")).strip(),
            bbox_x=_float(row, "bbox_x"),
            bbox_y=_float(row, "bbox_y"),
            bbox_w=_float(row, "bbox_w"),
            bbox_h=_float(row, "bbox_h"),
            zone=str(row.get("zone", "")).strip(),
            visible=_bool(row, "visible", True),
            crop_path=str(row.get("crop_path", "")).strip(),
        )
        for row in rows
    ]
    missing_ids = [frame for frame in frames if not frame.object_id]
    if missing_ids:
        msg = "object_tracks CSV contains rows without object_id"
        raise ValueError(msg)
    return sorted(frames, key=lambda item: (item.object_id, item.timestamp))


def make_run_dir(root: str | Path = "outputs/current_app") -> Path:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base / stamp
    counter = 1
    while path.exists():
        path = base / f"{stamp}_{counter:03d}"
        counter += 1
    path.mkdir(parents=True)
    return path


def write_results(output_dir: str | Path, results: Iterable[DetectionResult]) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_list = list(results)
    json_path = output / "results.json"
    csv_path = output / "results.csv"
    summary_path = output / "summary.md"

    json_path.write_text(
        json.dumps([asdict(result) for result in result_list], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "object_id",
            "object_label",
            "role",
            "predicted_label",
            "suspicion_score",
            "first_touch_time",
            "first_alert_time",
            "time_to_detection",
            "reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in result_list:
            writer.writerow(
                {
                    "object_id": result.object_id,
                    "object_label": result.object_label,
                    "role": result.role,
                    "predicted_label": result.predicted_label,
                    "suspicion_score": f"{result.suspicion_score:.3f}",
                    "first_touch_time": "" if result.first_touch_time is None else result.first_touch_time,
                    "first_alert_time": "" if result.first_alert_time is None else result.first_alert_time,
                    "time_to_detection": "" if result.time_to_detection is None else result.time_to_detection,
                    "reasons": "; ".join(result.reasons),
                }
            )

    lines = [
        "# Current App Analysis Summary",
        "",
        "| object_id | label | role | prediction | score | first_alert | reasons |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for result in result_list:
        first_alert = "" if result.first_alert_time is None else f"{result.first_alert_time:.2f}"
        lines.append(
            f"| {result.object_id} | {result.object_label} | {result.role} | "
            f"{result.predicted_label} | {result.suspicion_score:.2f} | {first_alert} | "
            f"{'; '.join(result.reasons)} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "summary": summary_path}
