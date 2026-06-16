from __future__ import annotations

import json
from pathlib import Path

from home_service_action_verifier.config import Settings
from home_service_action_verifier.schemas import AnalyzeConfig, FrameInfo, MaskMethod, SamplingMethod, VLMBackend, VideoMetadata
from home_service_action_verifier.vlm_client import OllamaVLMClient


def test_ollama_client_uses_per_run_model_override(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(url: str, payload: dict, headers: dict[str, str] | None = None, timeout: float = 60.0) -> dict:
        del headers, timeout
        captured["url"] = url
        captured["payload"] = payload
        return {
            "message": {
                "content": json.dumps(
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
            }
        }

    monkeypatch.setattr("home_service_action_verifier.vlm_client._post_json", fake_post_json)
    monkeypatch.setattr("home_service_action_verifier.vlm_client._image_to_base64", lambda path: "encoded-image")

    settings = Settings(ollama_enabled=True, ollama_host="http://localhost:11434", ollama_model="gemma3:4b")
    client = OllamaVLMClient(settings)
    config = AnalyzeConfig(
        video_path=Path("sample.mp4"),
        sampling_method=SamplingMethod.HYBRID,
        num_frames=8,
        mask_method=MaskMethod.NONE,
        vlm_backend=VLMBackend.OLLAMA,
        vlm_model="gemma3:12b",
        resize_width=640,
    )
    frames = [FrameInfo(frame_index=0, timestamp=0.0, path=Path("frame_000.jpg"))]
    metadata = VideoMetadata(
        video_path=Path("sample.mp4"),
        fps=10.0,
        duration=1.0,
        total_frames=10,
        width=640,
        height=480,
    )

    response = client.analyze(Path("grid.jpg"), frames, metadata, config)

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert isinstance(captured["payload"], dict)
    assert captured["payload"]["model"] == "gemma3:12b"
    assert response.confidence == 0.2
