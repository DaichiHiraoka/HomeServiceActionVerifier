"""ROI helpers used by scenario experiments."""

from __future__ import annotations

from home_service_action_verifier.detectors.hand_roi import crop_roi_from_frame, save_event_roi_frames

__all__ = ["crop_roi_from_frame", "save_event_roi_frames"]
