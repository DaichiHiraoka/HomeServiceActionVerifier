from __future__ import annotations

from home_service_action_verifier.windows import build_contact_windows

from .test_schemas import annotation_dict


def test_truncated_windows_do_not_include_frames_after_t_end() -> None:
    annotation = annotation_dict()
    annotation["fps"] = 10.0
    annotation["contacts"][0]["start_frame"] = 20
    annotation["contacts"][0]["end_frame"] = 80
    windows = build_contact_windows(annotation, [1], truncate_seconds=[0, 1, 3])
    by_t = {window.t_end_rel: window for window in windows}
    assert by_t[0].end_frame == 20
    assert by_t[1].end_frame == 30
    assert by_t[3].end_frame == 50
