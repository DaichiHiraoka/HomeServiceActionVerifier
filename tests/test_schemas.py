from __future__ import annotations

import pytest
from pydantic import ValidationError

from home_service_action_verifier.schemas import VLMResponse
from home_service_action_verifier.vlm_client import apply_research_constraints


def test_vlm_response_schema_accepts_valid_json() -> None:
    response = VLMResponse.model_validate(
        {
            "unauthorized_object_interaction_suspected": False,
            "confidence": 0.2,
            "target_object": None,
            "evidence_frames": [],
            "reason": "根拠が不足しています。",
            "privacy_sensitive_description_included": False,
            "limitations": "限定されたフレームのみです。",
        }
    )
    assert response.confidence == 0.2


def test_vlm_response_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        VLMResponse.model_validate(
            {
                "unauthorized_object_interaction_suspected": True,
                "confidence": 1.2,
                "target_object": "object",
                "evidence_frames": [1],
                "reason": "invalid",
                "privacy_sensitive_description_included": False,
                "limitations": "invalid",
            }
        )


def test_research_constraints_reject_container_as_target_object() -> None:
    response = VLMResponse(
        unauthorized_object_interaction_suspected=True,
        confidence=0.8,
        target_object="bag",
        evidence_frames=[1, 2],
        reason="bag moved",
        privacy_sensitive_description_included=False,
        limitations="limited",
    )
    constrained = apply_research_constraints(response)
    assert constrained.unauthorized_object_interaction_suspected is False
    assert constrained.confidence <= 0.25
    assert constrained.target_object is None


def test_research_constraints_cap_false_confidence() -> None:
    response = VLMResponse(
        unauthorized_object_interaction_suspected=False,
        confidence=0.9,
        target_object=None,
        evidence_frames=[],
        reason="not enough evidence",
        privacy_sensitive_description_included=False,
        limitations="limited",
    )
    constrained = apply_research_constraints(response)
    assert constrained.confidence == 0.5
