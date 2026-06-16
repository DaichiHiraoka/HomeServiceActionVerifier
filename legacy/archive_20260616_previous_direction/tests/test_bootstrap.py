from __future__ import annotations

from pathlib import Path

from home_service_action_verifier.model_selection import ensure_local_env


def test_ensure_local_env_creates_runtime_defaults(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"

    created = ensure_local_env(env_path)

    assert created is True
    text = env_path.read_text(encoding="utf-8")
    assert "OLLAMA_ENABLED=true" in text
    assert "OLLAMA_MODEL=gemma3:4b" in text
    assert "OLLAMA_TIMEOUT_SEC=300" in text


def test_ensure_local_env_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OLLAMA_MODEL=custom\n", encoding="utf-8")

    created = ensure_local_env(env_path)

    assert created is False
    assert env_path.read_text(encoding="utf-8") == "OLLAMA_MODEL=custom\n"
