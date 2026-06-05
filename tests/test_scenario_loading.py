from __future__ import annotations

import pytest

from home_service_action_verifier.scenario import load_event_tokens, load_work_order, load_zone_config


def test_load_router_repair_work_order() -> None:
    work_order = load_work_order("configs/scenarios/router_repair.json")

    assert work_order.scenario_id == "router_repair_001"
    assert "router_shelf" in work_order.authorized_zones
    assert "private_desk" in work_order.forbidden_zones


def test_load_router_repair_zones() -> None:
    zone_config = load_zone_config("configs/zones/router_repair_zones.json")

    assert zone_config.video_width == 1280
    assert any(zone.zone_id == "router_shelf" for zone in zone_config.zones)


def test_load_annotations_maps_label_to_ground_truth() -> None:
    events = load_event_tokens("data/real/router_trial_001_annotations.example.jsonl")

    assert len(events) == 9
    assert events[0].ground_truth_label == "normal"
    assert events[-1].ground_truth_label == "high_risk"


def test_load_annotations_rejects_invalid_label(tmp_path) -> None:
    path = tmp_path / "bad_annotations.jsonl"
    path.write_text(
        '{"event_id":"BAD","start_sec":0,"end_sec":1,"label":"criminal","action":"inspect"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid event label"):
        load_event_tokens(path)
