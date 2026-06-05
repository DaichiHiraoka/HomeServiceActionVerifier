"""Lightweight pipeline and sampling evaluation helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from home_service_action_verifier.analyzer import analyze_video
from home_service_action_verifier.config import Settings, get_settings
from home_service_action_verifier.schemas import EvaluationMetrics, MaskMethod, ROI, SamplingMethod, VLMBackend


def parse_evidence_frames(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    frames: list[int] = []
    for item in text.split(";"):
        item = item.strip()
        if item:
            frames.append(int(item))
    return frames


def selected_frame_recall(selected_indices: list[int], evidence_frames: list[int], tolerance: int = 2) -> float | None:
    if not evidence_frames:
        return None
    matched = 0
    for evidence in evidence_frames:
        if any(abs(selected - evidence) <= tolerance for selected in selected_indices):
            matched += 1
    return matched / len(evidence_frames)


def _resolve_video_path(labels_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    relative_to_labels = labels_path.parent / candidate
    if relative_to_labels.exists():
        return relative_to_labels
    return candidate


def _make_eval_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("evaluation_%Y%m%d_%H%M%S")
    path = output_root / stamp
    counter = 1
    while path.exists():
        path = output_root / f"{stamp}_{counter:03d}"
        counter += 1
    path.mkdir(parents=True)
    return path


def evaluate_labels(
    labels_csv: str | Path,
    sampling_method: str | SamplingMethod = SamplingMethod.EVENT_WINDOW,
    num_frames: int = 8,
    mask_method: str | MaskMethod = MaskMethod.NONE,
    roi: ROI | tuple[int, int, int, int] | None = None,
    vlm_backend: str | VLMBackend = VLMBackend.MOCK,
    vlm_model: str | None = None,
    resize_width: int | None = None,
    settings: Settings | None = None,
) -> EvaluationMetrics:
    settings = settings or get_settings()
    labels_path = Path(labels_csv)
    dataframe = pd.read_csv(labels_path)
    required = {"video_path", "label", "evidence_frames"}
    missing = required - set(dataframe.columns)
    if missing:
        msg = f"labels.csv is missing columns: {sorted(missing)}"
        raise ValueError(msg)

    y_true: list[int] = []
    y_pred: list[int] = []
    frame_recalls: list[float] = []
    selected_counts: list[int] = []
    processing_times: list[float] = []
    rows: list[dict] = []

    for _, row in dataframe.iterrows():
        video = _resolve_video_path(labels_path, str(row["video_path"]))
        result = analyze_video(
            video_path=video,
            sampling_method=sampling_method,
            num_frames=num_frames,
            mask_method=mask_method,
            roi=roi,
            vlm_backend=vlm_backend,
            vlm_model=vlm_model,
            resize_width=resize_width,
            settings=settings,
        )
        true_label = int(row["label"])
        predicted_label = int(result.vlm_response.unauthorized_object_interaction_suspected)
        selected = [frame.frame_index for frame in result.selected_frames]
        evidence = parse_evidence_frames(row["evidence_frames"])
        recall_value = selected_frame_recall(selected, evidence)
        if recall_value is not None:
            frame_recalls.append(recall_value)

        y_true.append(true_label)
        y_pred.append(predicted_label)
        selected_counts.append(len(selected))
        processing_times.append(result.processing_time_sec)
        rows.append(
            {
                "video_path": str(video),
                "label": true_label,
                "predicted": predicted_label,
                "selected_frames": ";".join(str(item) for item in selected),
                "evidence_frames": ";".join(str(item) for item in evidence),
                "selected_frame_recall": recall_value,
                "run_dir": str(result.run_dir),
                "processing_time_sec": result.processing_time_sec,
            }
        )

    metrics = EvaluationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        precision=float(precision_score(y_true, y_pred, zero_division=0)) if y_true else 0.0,
        recall=float(recall_score(y_true, y_pred, zero_division=0)) if y_true else 0.0,
        f1=float(f1_score(y_true, y_pred, zero_division=0)) if y_true else 0.0,
        selected_frame_recall=float(sum(frame_recalls) / len(frame_recalls)) if frame_recalls else 0.0,
        average_num_selected_frames=float(sum(selected_counts) / len(selected_counts)) if selected_counts else 0.0,
        average_processing_time_sec=float(sum(processing_times) / len(processing_times)) if processing_times else 0.0,
        num_videos=len(y_true),
        notes=(
            "legacy mock clientの分類指標は精度評価としては意味が限定的です。"
            "この評価は主にパイプラインとサンプリング比較の確認用です。"
        ),
    )

    eval_dir = _make_eval_dir(settings.outputs_dir)
    metrics.output_dir = eval_dir
    (eval_dir / "metrics.json").write_text(
        json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(eval_dir / "per_video.csv", index=False)
    return metrics
