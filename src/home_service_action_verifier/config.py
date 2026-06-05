"""Environment-backed settings for the local PoC."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def load_env_file(path: str | Path = ".env") -> None:
    """Load a simple KEY=VALUE .env file without adding python-dotenv."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class Settings(BaseModel):
    resize_width: int = Field(default=640, gt=0)
    default_num_frames: int = Field(default=8, gt=0)
    outputs_dir: Path = Path("outputs/runs")

    ollama_enabled: bool = False
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_timeout_sec: float = Field(default=300.0, gt=0)

    openai_compatible_enabled: bool = False
    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_model: str = ""


def get_settings() -> Settings:
    """Read settings from environment variables without requiring python-dotenv."""

    load_env_file()
    return Settings(
        resize_width=_env_int("PRIVACY_VLM_RESIZE_WIDTH", 640),
        default_num_frames=_env_int("PRIVACY_VLM_DEFAULT_NUM_FRAMES", 8),
        outputs_dir=Path(os.getenv("PRIVACY_VLM_OUTPUTS_DIR", "outputs/runs")),
        ollama_enabled=_env_bool("OLLAMA_ENABLED", False),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        ollama_timeout_sec=_env_float("OLLAMA_TIMEOUT_SEC", 300.0),
        openai_compatible_enabled=_env_bool("OPENAI_COMPATIBLE_ENABLED", False),
        openai_compatible_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""),
        openai_compatible_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
        openai_compatible_model=os.getenv("OPENAI_COMPATIBLE_MODEL", ""),
    )
