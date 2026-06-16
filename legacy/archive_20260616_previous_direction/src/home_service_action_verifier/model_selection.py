"""Research model selection and local runtime diagnostics."""

from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from home_service_action_verifier.config import Settings, get_settings

RECOMMENDED_OLLAMA_MODEL = "gemma3:4b"
HIGHER_QUALITY_OLLAMA_MODEL = "gemma3:12b"
EFFICIENT_ALTERNATIVE_MODEL = "qwen2.5vl:3b"
UI_OLLAMA_MODELS = [RECOMMENDED_OLLAMA_MODEL, HIGHER_QUALITY_OLLAMA_MODEL]

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"
ENV_PATH = ROOT / ".env"


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    role: str
    expected_download_size: str
    rationale: str
    command: str


@dataclass
class OllamaDoctorResult:
    ollama_command_available: bool
    ollama_executable: str | None
    host: str
    host_reachable: bool
    configured_model: str
    configured_model_present: bool
    installed_models: list[str]
    recommended_model: str
    pull_command: str
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BootstrapResult:
    env_path: str
    env_created: bool
    sample_data_ready: bool
    ollama_executable: str | None
    host_reachable: bool
    requested_models: list[str]
    installed_before: list[str]
    pulled_models: list[str]
    already_present_models: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def find_ollama_executable() -> str | None:
    configured = os.getenv("OLLAMA_EXE")
    if configured and Path(configured).exists():
        return configured

    from_path = shutil.which("ollama")
    if from_path:
        return from_path

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        default_path = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if default_path.exists():
            return str(default_path)
    return None


def model_candidates() -> list[ModelCandidate]:
    return [
        ModelCandidate(
            name=RECOMMENDED_OLLAMA_MODEL,
            role="minimum_research_default",
            expected_download_size="about 3.3GB",
            rationale=(
                "Small local Gemma 3 vision model with text+image input, long context, and multilingual support. "
                "Use this as the default local model for repeated sampling/masking experiments."
            ),
            command=f"ollama pull {RECOMMENDED_OLLAMA_MODEL}",
        ),
        ModelCandidate(
            name=HIGHER_QUALITY_OLLAMA_MODEL,
            role="higher_quality_local",
            expected_download_size="about 8.1GB",
            rationale=(
                "Better quality target on machines that can tolerate more memory use and latency. "
                "Use it to confirm whether conclusions are stable under a stronger local VLM."
            ),
            command=f"ollama pull {HIGHER_QUALITY_OLLAMA_MODEL}",
        ),
        ModelCandidate(
            name=EFFICIENT_ALTERNATIVE_MODEL,
            role="cross_model_check",
            expected_download_size="about 3.2GB",
            rationale=(
                "Qwen2.5-VL small vision model. Use as a cross-model robustness check "
                "after Gemma 3 experiments are working."
            ),
            command=f"ollama pull {EFFICIENT_ALTERNATIVE_MODEL}",
        ),
    ]


def _required_sample_paths() -> list[Path]:
    return [
        SAMPLE_DIR / "sample_suspicious.mp4",
        SAMPLE_DIR / "sample_normal.mp4",
        SAMPLE_DIR / "labels.csv",
    ]


def ensure_local_env(path: Path = ENV_PATH) -> bool:
    """Create the local runtime .env used by the UI when it is missing."""

    if path.exists():
        return False
    path.write_text(
        "\n".join(
            [
                "PRIVACY_VLM_RESIZE_WIDTH=640",
                "PRIVACY_VLM_DEFAULT_NUM_FRAMES=8",
                "PRIVACY_VLM_OUTPUTS_DIR=outputs/runs",
                "",
                "OLLAMA_ENABLED=true",
                "OLLAMA_HOST=http://localhost:11434",
                f"OLLAMA_MODEL={RECOMMENDED_OLLAMA_MODEL}",
                "OLLAMA_TIMEOUT_SEC=300",
                "",
                "OPENAI_COMPATIBLE_ENABLED=false",
                "OPENAI_COMPATIBLE_BASE_URL=",
                "OPENAI_COMPATIBLE_API_KEY=",
                "OPENAI_COMPATIBLE_MODEL=",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return True


def ensure_sample_data() -> bool:
    """Generate bundled synthetic sample data if it is absent."""

    if all(path.exists() for path in _required_sample_paths()):
        return True
    runpy.run_path(str(ROOT / "scripts" / "generate_synthetic_video.py"), run_name="__main__")
    return all(path.exists() for path in _required_sample_paths())


def installed_ollama_models(settings: Settings | None = None) -> tuple[bool, list[str], str | None]:
    settings = settings or get_settings()
    try:
        data = _get_json(f"{settings.ollama_host.rstrip('/')}/api/tags")
        models = [item.get("name", "") for item in data.get("models", []) if item.get("name")]
        return True, models, None
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        return False, [], str(exc)


def _wait_for_ollama_host(settings: Settings, timeout_sec: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        reachable, _models, _error = installed_ollama_models(settings)
        if reachable:
            return True
        time.sleep(0.5)
    return False


def ensure_ollama_models(models: list[str], settings: Settings | None = None) -> BootstrapResult:
    settings = settings or get_settings()
    notes: list[str] = []
    env_created = ensure_local_env()
    sample_data_ready = ensure_sample_data()

    executable = find_ollama_executable()
    if not executable:
        return BootstrapResult(
            env_path=str(ENV_PATH),
            env_created=env_created,
            sample_data_ready=sample_data_ready,
            ollama_executable=None,
            host_reachable=False,
            requested_models=models,
            installed_before=[],
            pulled_models=[],
            already_present_models=[],
            notes=["Ollama command was not found. Install Ollama, then rerun this command."],
        )

    host_reachable, installed_before, host_error = installed_ollama_models(settings)
    if not host_reachable:
        notes.append(f"Ollama host was not reachable before pulling models: {host_error}")
        try:
            subprocess.Popen(
                [executable, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            notes.append(f"Failed to start ollama serve: {exc}")
        host_reachable = _wait_for_ollama_host(settings)

    if not host_reachable:
        return BootstrapResult(
            env_path=str(ENV_PATH),
            env_created=env_created,
            sample_data_ready=sample_data_ready,
            ollama_executable=executable,
            host_reachable=False,
            requested_models=models,
            installed_before=installed_before,
            pulled_models=[],
            already_present_models=[],
            notes=notes + ["Ollama host is still unreachable. Start Ollama, then rerun this command."],
        )

    pulled: list[str] = []
    already_present: list[str] = []
    installed = set(installed_before)
    for model in models:
        if model in installed:
            already_present.append(model)
            continue
        subprocess.run([executable, "pull", model], check=True)
        pulled.append(model)
        installed.add(model)

    return BootstrapResult(
        env_path=str(ENV_PATH),
        env_created=env_created,
        sample_data_ready=sample_data_ready,
        ollama_executable=executable,
        host_reachable=True,
        requested_models=models,
        installed_before=installed_before,
        pulled_models=pulled,
        already_present_models=already_present,
        notes=notes,
    )


def _get_json(url: str, timeout: float = 5.0) -> dict:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_doctor(settings: Settings | None = None) -> OllamaDoctorResult:
    settings = settings or get_settings()
    notes: list[str] = []
    executable = find_ollama_executable()
    command_available = executable is not None
    if not executable:
        notes.append("Ollama command is not on PATH. Install Ollama before running the local VLM backend.")

    host_reachable, installed_models, host_error = installed_ollama_models(settings)
    if host_error:
        notes.append(f"Ollama host is not reachable at {settings.ollama_host}: {host_error}")

    model_present = settings.ollama_model in installed_models
    if host_reachable and not model_present:
        notes.append(f"Configured model is not installed: {settings.ollama_model}")
    if not settings.ollama_enabled:
        notes.append("OLLAMA_ENABLED is false. The pipeline will not call Ollama until it is true.")

    return OllamaDoctorResult(
        ollama_command_available=command_available,
        ollama_executable=executable,
        host=settings.ollama_host,
        host_reachable=host_reachable,
        configured_model=settings.ollama_model,
        configured_model_present=model_present,
        installed_models=installed_models,
        recommended_model=RECOMMENDED_OLLAMA_MODEL,
        pull_command=f"ollama pull {settings.ollama_model or RECOMMENDED_OLLAMA_MODEL}",
        notes=notes,
    )
