"""Fixed-camera zone lookup helpers."""

from __future__ import annotations

from home_service_action_verifier.schemas import ZoneConfig


def find_zone_for_point(x: int, y: int, zone_config: ZoneConfig) -> str | None:
    for zone in zone_config.zones:
        x1, y1, x2, y2 = zone.bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            return zone.zone_id
    return None


def find_zone_for_bbox(bbox: tuple[int, int, int, int], zone_config: ZoneConfig) -> str | None:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return find_zone_for_point(center_x, center_y, zone_config)
