# HomeServiceActionVerifier

## Current Direction

The current research direction is contextual suspicious object-manipulation detection for unattended work in private homes.

This repository root is reserved for the new direction. Do not use the archived previous approach as the default reference for future work.

The active topic is:

> Estimate whether a visually similar object-manipulation sequence, especially moving an object while it appears to be held, belongs to a normal work sequence or a suspicious sequence by using worker skeleton/posture time series and localized image sequences around the target object.

For human information, the project uses skeleton and posture only. For object information, the project may use tracking and cropped frames limited to the target object and its immediate surroundings. The target is not to claim that hidden intent can be read directly. The practical target is to estimate contextual suspiciousness from observable posture transitions, hand-object contact, object movement, and limited local object context.

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

The active prototype is packaged as `grip_detector`. It no longer treats
"hand looks closed near an object" as the final event. The live score is for
"the hand is near/closed on a tracked object and that object moved." The old
script path is kept as a compatibility wrapper.

```powershell
uv run .\grip_detector\app.py
uv run .\お試し\skeleton_grip_detector.py
uv run skeleton-grip-detector
uv run python -m grip_detector
```

If `--source` is omitted, the detector opens an IP Webcam input GUI. Enter the
IP address and port with the keyboard or the on-screen numeric buttons. Pressing
Enter inserts `.` between IP address parts and then `:` before the port; Esc
cancels. The `IPmemory` button restores the last confirmed IP/port from the
local `ip_memory.json` file. That file is created automatically and is ignored
by Git.

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

`--no-object-tracking` is only for debugging the hand-pose pipeline. With
object tracking disabled, the prototype cannot judge held-object movement and
the movement event score stays at zero.

For target objects outside the pretrained COCO classes, use `--yolo-model` with
a custom trained `.pt` model.

Movement thresholds are intentionally exposed because the right values depend
on camera distance and resolution:

```powershell
uv run .\grip_detector\app.py --object-motion-speed-threshold 60 --object-motion-displacement-threshold 35
```

`--object-motion-speed-threshold` is the current movement speed in pixels per
second. `--object-motion-displacement-threshold` is the distance from the
object's initial tracked position. Both are used to suppress false positives
where a hand overlaps an object without moving it.

To reduce phantom movement from static textures such as desk grain, tracked
objects now become motion-eligible only after a position-based settle check.
Contact evidence is still shown before settling, but movement evidence is
rejected when the original location still matches the initial object appearance.
The overlay shows `S`, settle frame count, vacancy similarity, and `mv` so this
can be diagnosed live.
