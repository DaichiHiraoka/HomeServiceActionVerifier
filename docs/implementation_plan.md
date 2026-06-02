# Implementation Plan

## Implemented

- Added event-level schemas: `WorkOrder`, `ZoneConfig`, `EventToken`, `DetectionResult`, and `EventEvaluationMetrics`.
- Added `router_repair` work order, zone config, and example JSONL annotations.
- Added scenario loaders for work orders, zone configs, and event tokens.
- Added Rule-Based event detection with Japanese reasons and bounded suspicion scores.
- Added event-level metrics, confusion matrix, per-event CSV, and summary output.
- Added fixed-camera zone helpers and manual ROI extraction helpers.
- Added scenario CLI commands: `analyze-scenario`, `evaluate-events`, and `compare-methods`.
- Added tests for scenario loading, rule engine behavior, event evaluation, and CLI output.

## Priority Next Steps

1. Wire event-window frame extraction into `vlm_direct_full`.
2. Use `detectors.hand_roi.save_event_roi_frames` to enable `vlm_direct_roi`.
3. Connect VLM scoring into `baselines.proposed` for ambiguous events.
4. Add calibrated zone files for real captured videos.
5. Add richer annotations for task stage and hand/object ROI per event.

## Known Limits

- `vlm_direct_full` and `vlm_direct_roi` are explicit placeholders in the initial implementation.
- Proposed method currently uses Rule-Based scores and marks ambiguous events as future VLM-confirmation targets.
- Zone detection uses fixed rectangular regions and bbox center points.
- Hand/object ROI extraction is manual and does not run a hand detection model.
- The system does not infer identity or personal attributes and does not make crime conclusions.

## Future Comparison Table

| Method | Input | Status |
| --- | --- | --- |
| Rule-Based | EventToken + WorkOrder | implemented |
| VLM Direct Full | Full RGB selected event frames | placeholder |
| VLM Direct ROI | Hand/Object ROI selected event frames | placeholder |
| Token Only | EventToken only | covered by Rule-Based baseline |
| Proposed | EventToken + WorkOrder + ROI + VLM | initial Rule-Based scaffold |
