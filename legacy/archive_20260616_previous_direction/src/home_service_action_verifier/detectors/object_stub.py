"""Explicit placeholder for future object and hand/object association detection."""

from __future__ import annotations

from home_service_action_verifier.schemas import EventToken


def infer_object_from_event_token(event: EventToken) -> dict[str, str | None]:
    """Return annotation-derived object context until a detector is connected."""

    return {
        "object_class": event.object_class,
        "object_owner": event.object_owner,
        "container_class": event.container_class,
        "container_owner": event.container_owner,
        "status": "annotation_stub",
    }
