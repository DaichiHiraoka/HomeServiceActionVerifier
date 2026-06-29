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

The current desktop prototype is implemented as a Python/Tkinter app.

Run the desktop app:

```powershell
uv run home-service-verifier-app
```

Run the sample analysis without opening the GUI:

```powershell
uv run home-service-verifier-app --context sample_data/task_context.json --skeleton sample_data/skeleton.csv --objects sample_data/object_tracks.csv
```

Run the normal router-work trial generated from `data/raw_videos/IMG_7852.mp4`:

```powershell
uv run home-service-verifier-app --context data/trials/router_normal_img_7852/context.json --skeleton data/trials/router_normal_img_7852/skeleton.csv --objects data/trials/router_normal_img_7852/object_tracks.csv
```

Regenerate that trial from the raw video:

```powershell
uv run --no-project --python 3.11 --with "mediapipe==0.10.21" --with opencv-python-headless python scripts/prepare_router_trial.py --video data/raw_videos/IMG_7852.mp4 --out data/trials/router_normal_img_7852 --sample-seconds 1.0
```

Input file locations:

```text
data/contexts/       task context JSON files
data/skeleton/       skeleton CSV files
data/object_tracks/  object tracking CSV files
data/object_crops/   local object crop images
data/raw_videos/     raw videos, not used directly by the current app
data/trials/         per-trial grouped inputs
```

The active implementation is structured around skeleton time series data, local object tracking/cropping metadata, sequence labeling, transition/anomaly scoring, early warning timing, and uncertainty-aware evaluation.
