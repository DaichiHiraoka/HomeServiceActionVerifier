"""End-to-end video analysis pipeline."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from home_service_action_verifier.config import Settings, get_settings
from home_service_action_verifier.masking import apply_mask, grid_image, read_image
from home_service_action_verifier.sampling import event_window_sampling, hybrid_sampling, motion_sampling, uniform_sampling
from home_service_action_verifier.schemas import (
    AnalyzeConfig,
    AnalyzeResult,
    FrameInfo,
    MaskMethod,
    ROI,
    SamplingMethod,
    VLMBackend,
    VLMResponse,
)
from home_service_action_verifier.video_io import get_video_metadata
from home_service_action_verifier.vlm_client import create_vlm_client


def _make_run_dir(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / stamp
    counter = 1
    while run_dir.exists():
        run_dir = root / f"{stamp}_{counter:03d}"
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _sample_frames(config: AnalyzeConfig, output_dir: Path) -> list[FrameInfo]:
    method = SamplingMethod(config.sampling_method)
    if method == SamplingMethod.UNIFORM:
        return uniform_sampling(config.video_path, config.num_frames, config.resize_width, output_dir=output_dir)
    if method == SamplingMethod.MOTION:
        return motion_sampling(config.video_path, config.num_frames, config.resize_width, output_dir=output_dir)
    if method == SamplingMethod.HYBRID:
        return hybrid_sampling(config.video_path, config.num_frames, config.resize_width, output_dir=output_dir)
    if method == SamplingMethod.EVENT_WINDOW:
        return event_window_sampling(config.video_path, config.num_frames, config.resize_width, output_dir=output_dir)
    msg = f"Unsupported sampling method: {config.sampling_method}"
    raise ValueError(msg)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_report(
    path: Path,
    config: AnalyzeConfig,
    metadata: dict,
    selected_frames: list[FrameInfo],
    response: VLMResponse,
    processing_time_sec: float,
) -> None:
    selected = ", ".join(f"{frame.frame_index} ({frame.timestamp:.2f}s)" for frame in selected_frames)
    report = f"""# Home Service Action Verifier Legacy VLM Report

## Summary

- video: `{config.video_path}`
- sampling_method: `{config.sampling_method}`
- mask_method: `{config.mask_method}`
- vlm_backend: `{config.vlm_backend}`
- vlm_model: `{config.vlm_model or "-"}`
- selected_frames: {selected}
- processing_time_sec: {processing_time_sec:.3f}

## Video Metadata

```json
{json.dumps(metadata, ensure_ascii=False, indent=2)}
```

## VLM Result

```json
{json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2)}
```

## Note

このレポートはlegacy動画解析の補助出力です。犯罪や不正行為の断定には使えません。
"""
    path.write_text(report, encoding="utf-8")


def analyze_video(
    video_path: str | Path,
    sampling_method: str | SamplingMethod = SamplingMethod.HYBRID,
    num_frames: int = 8,
    mask_method: str | MaskMethod = MaskMethod.NONE,
    roi: ROI | tuple[int, int, int, int] | None = None,
    vlm_backend: str | VLMBackend = VLMBackend.MOCK,
    vlm_model: str | None = None,
    output_root: str | Path | None = None,
    resize_width: int | None = None,
    settings: Settings | None = None,
) -> AnalyzeResult:
    settings = settings or get_settings()
    output_root = output_root or settings.outputs_dir
    resize_width = resize_width or settings.resize_width

    roi_model = ROI(x1=roi[0], y1=roi[1], x2=roi[2], y2=roi[3]) if isinstance(roi, tuple) else roi
    config = AnalyzeConfig(
        video_path=Path(video_path),
        sampling_method=SamplingMethod(sampling_method),
        num_frames=num_frames,
        mask_method=MaskMethod(mask_method),
        roi=roi_model,
        vlm_backend=VLMBackend(vlm_backend),
        vlm_model=vlm_model,
        resize_width=resize_width,
    )

    start = time.perf_counter()
    run_dir = _make_run_dir(output_root)
    selected_dir = run_dir / "selected_frames"
    masked_dir = run_dir / "masked_frames"
    selected_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)

    metadata = get_video_metadata(config.video_path)
    selected_frames = _sample_frames(config, selected_dir)
    if not selected_frames:
        msg = "No frames were selected from the video."
        raise ValueError(msg)

    masked_images = []
    masked_paths: list[Path] = []
    for frame_info in selected_frames:
        if frame_info.path is None:
            msg = f"Selected frame has no image path: {frame_info.frame_index}"
            raise ValueError(msg)
        image = read_image(frame_info.path)
        masked = apply_mask(image, config.mask_method, config.roi)
        masked_path = masked_dir / Path(frame_info.path).name
        ok = cv2.imwrite(str(masked_path), masked)
        if not ok:
            msg = f"Failed to write masked frame: {masked_path}"
            raise ValueError(msg)
        masked_images.append(masked)
        masked_paths.append(masked_path)

    grid = grid_image(masked_images, selected_frames)
    grid_path = run_dir / "grid.jpg"
    if not cv2.imwrite(str(grid_path), grid):
        msg = f"Failed to write grid image: {grid_path}"
        raise ValueError(msg)

    client = create_vlm_client(config.vlm_backend, settings)
    response = client.analyze(grid_path, selected_frames, metadata, config)

    processing_time_sec = time.perf_counter() - start
    result_path = run_dir / "result.json"
    config_path = run_dir / "config.json"
    report_path = run_dir / "report.md"

    _write_json(result_path, response.model_dump(mode="json"))
    config_payload = {
        "config": config.model_dump(mode="json"),
        "metadata": metadata.model_dump(mode="json"),
        "selected_frames": [frame.model_dump(mode="json") for frame in selected_frames],
        "masked_frame_paths": [str(path) for path in masked_paths],
        "grid_path": str(grid_path),
        "result_path": str(result_path),
        "report_path": str(report_path),
        "processing_time_sec": processing_time_sec,
    }
    _write_json(config_path, config_payload)
    _write_report(
        report_path,
        config,
        metadata.model_dump(mode="json"),
        selected_frames,
        response,
        processing_time_sec,
    )

    return AnalyzeResult(
        run_dir=run_dir,
        metadata=metadata,
        selected_frames=selected_frames,
        masked_frame_paths=masked_paths,
        grid_path=grid_path,
        result_path=result_path,
        report_path=report_path,
        config_path=config_path,
        vlm_response=response,
        processing_time_sec=processing_time_sec,
    )
