# HomeServiceActionVerifier

## Current Direction

The current research direction is contextual suspicious object-manipulation detection for unattended work in private homes.

This repository root is reserved for the new direction. Do not use the archived previous approach as the default reference for future work.

The active topic is:

> Estimate whether a visually similar object-manipulation action, such as picking up an object, belongs to a normal work sequence or a suspicious sequence by using worker skeleton/posture time series and localized image sequences around the target object.

For human information, the project uses skeleton and posture only. For object information, the project may use tracking and cropped frames limited to the target object and its immediate surroundings. The target is not to claim that hidden intent can be read directly. The practical target is to estimate contextual suspiciousness from observable posture transitions, object movement, and limited local object context.

## Archive Boundary

Previous work is isolated under:

```text
legacy/archive_20260616_previous_direction/
```

That archive contains the former implementation, documents, experiments, tests, configs, and local analysis assets.

Treat the archive as historical material only. Open it only when the user explicitly asks to inspect or recover previous work.

## Next Step

Build the new project structure around skeleton time series data, local object tracking/cropping, sequence labeling, transition/anomaly scoring, early warning timing, and uncertainty-aware evaluation.

## Running the Current Detector

The active prototype is packaged as `grip_detector`. The old script path is
kept as a compatibility wrapper.

```powershell
uv run .\grip_detector\app.py
uv run .\お試し\skeleton_grip_detector.py
uv run skeleton-grip-detector
uv run python -m grip_detector
```

The project is pinned to Python 3.12 because the current detector depends on
MediaPipe.

YOLO is the default object detector. To restrict detections to specific COCO
classes, pass comma-separated class names or class IDs:

```powershell
uv run .\grip_detector\app.py --yolo-classes "cell phone,book,bottle,cup"
```

Use the old motion detector only when needed:

```powershell
uv run .\grip_detector\app.py --object-detector motion
```

For target objects outside the pretrained COCO classes, use `--yolo-model` with
a custom trained `.pt` model.
