"""Run a sampling x masking comparison matrix for research verification."""

from __future__ import annotations

import argparse
import json
import runpy
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from home_service_action_verifier.analyzer import analyze_video
from home_service_action_verifier.config import get_settings
from home_service_action_verifier.evaluation import parse_evidence_frames, selected_frame_recall
from home_service_action_verifier.schemas import MaskMethod, SamplingMethod, VLMBackend

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
LABELS_PATH = SAMPLE_DIR / "labels.csv"


def _ensure_sample_data() -> None:
    required = [
        SAMPLE_DIR / "sample_suspicious.mp4",
        SAMPLE_DIR / "sample_normal.mp4",
        LABELS_PATH,
    ]
    if all(path.exists() for path in required):
        return
    runpy.run_path(str(ROOT / "scripts" / "generate_synthetic_video.py"), run_name="__main__")


def _parse_csv_list(value: str | None, defaults: list[str]) -> list[str]:
    if value is None or not value.strip():
        return defaults
    return [item.strip() for item in value.split(",") if item.strip()]


def _make_output_dir() -> Path:
    root = get_settings().outputs_dir
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"research_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    counter = 1
    while path.exists():
        path = root / f"research_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter:03d}"
        counter += 1
    path.mkdir(parents=True)
    return path


def run_matrix(args: argparse.Namespace) -> Path:
    _ensure_sample_data()
    output_dir = _make_output_dir()

    default_samplings = [item.value for item in SamplingMethod]
    default_masks = [item.value for item in MaskMethod]
    if args.quick:
        default_samplings = [SamplingMethod.UNIFORM.value, SamplingMethod.HYBRID.value, SamplingMethod.EVENT_WINDOW.value]
        default_masks = [MaskMethod.NONE.value, MaskMethod.BACKGROUND_BLUR_WITH_ROI.value, MaskMethod.LOWER_BODY_ONLY.value]

    samplings = _parse_csv_list(args.samplings, default_samplings)
    masks = _parse_csv_list(args.masks, default_masks)
    labels = pd.read_csv(LABELS_PATH)
    rows: list[dict] = []
    started = time.perf_counter()

    for _, label_row in labels.iterrows():
        video_path = ROOT / str(label_row["video_path"])
        true_label = int(label_row["label"])
        evidence = parse_evidence_frames(label_row["evidence_frames"])
        for sampling in samplings:
            for mask in masks:
                result = analyze_video(
                    video_path=video_path,
                    sampling_method=sampling,
                    num_frames=args.num_frames,
                    mask_method=mask,
                    vlm_backend=args.vlm_backend,
                    vlm_model=args.vlm_model,
                    resize_width=args.resize_width,
                )
                selected = [frame.frame_index for frame in result.selected_frames]
                rows.append(
                    {
                        "video_path": str(video_path),
                        "true_label": true_label,
                        "sampling_method": sampling,
                        "mask_method": mask,
                        "vlm_backend": args.vlm_backend,
                        "vlm_model": args.vlm_model,
                        "prediction": int(result.vlm_response.unauthorized_object_interaction_suspected),
                        "confidence": result.vlm_response.confidence,
                        "target_object": result.vlm_response.target_object,
                        "evidence_frames_truth": ";".join(str(item) for item in evidence),
                        "selected_frames": ";".join(str(item) for item in selected),
                        "selected_frame_recall": selected_frame_recall(selected, evidence),
                        "privacy_sensitive_description_included": result.vlm_response.privacy_sensitive_description_included,
                        "reason": result.vlm_response.reason,
                        "limitations": result.vlm_response.limitations,
                        "run_dir": str(result.run_dir),
                        "grid_path": str(result.grid_path),
                        "processing_time_sec": result.processing_time_sec,
                    }
                )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    grouped = (
        summary.groupby(["sampling_method", "mask_method"], dropna=False)
        .agg(
            average_confidence=("confidence", "mean"),
            average_selected_frame_recall=("selected_frame_recall", "mean"),
            average_processing_time_sec=("processing_time_sec", "mean"),
            num_runs=("video_path", "count"),
        )
        .reset_index()
    )
    grouped_path = output_dir / "by_condition.csv"
    grouped.to_csv(grouped_path, index=False)

    config = {
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model,
        "num_frames": args.num_frames,
        "resize_width": args.resize_width,
        "samplings": samplings,
        "masks": masks,
        "labels": str(LABELS_PATH),
        "elapsed_time_sec": time.perf_counter() - started,
        "notes": (
            "Mock backend results verify pipeline wiring only. Use --vlm-backend ollama "
            "after running the doctor command and pulling the selected VLM."
        ),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_markdown_report(config, summary, grouped), encoding="utf-8")
    return output_dir


def _markdown_report(config: dict, summary: pd.DataFrame, grouped: pd.DataFrame) -> str:
    per_run = summary[
        ["video_path", "sampling_method", "mask_method", "prediction", "confidence", "selected_frame_recall", "run_dir"]
    ]
    return f"""# Research Matrix Summary

## Config

```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

## Condition Summary

{_markdown_table(grouped)}

## Per Run

{_markdown_table(per_run)}

## Interpretation Notes

- `selected_frame_recall` is the primary sampling diagnostic.
- Compare `reason` and `limitations` in `summary.csv` to study explanation changes.
- Do not treat any output as a crime determination.
"""


def _markdown_table(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No rows._"
    columns = list(dataframe.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in dataframe.iterrows():
        values = [str(row[column]).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run research comparison matrix")
    parser.add_argument("--vlm-backend", choices=[item.value for item in VLMBackend], default=VLMBackend.MOCK.value)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--samplings", default=None, help="Comma-separated sampling methods")
    parser.add_argument("--masks", default=None, help="Comma-separated mask methods")
    parser.add_argument("--quick", action="store_true", help="Run a smaller matrix for smoke tests")
    return parser


def main() -> int:
    output_dir = run_matrix(build_parser().parse_args())
    print(f"Wrote research matrix outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
