"""VLM backend adapters.

Only selected and masked grid images are sent to optional external backends.
Raw video files are never transmitted by these clients.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from home_service_action_verifier.config import Settings, get_settings
from home_service_action_verifier.schemas import AnalyzeConfig, FrameInfo, VideoMetadata, VLMBackend, VLMResponse


SYSTEM_PROMPT = """あなたはスマートホーム内の限定された視覚情報から、許可外物品操作の疑いを説明する研究用PoCの解析器です。
犯罪や不正行為を断定してはいけません。legacy動画解析では unauthorized_object_interaction_suspected を内部フラグとしてのみ使います。
顔、年齢、性別、体型、服装の詳細、個人識別情報を推定・記述してはいけません。
判断根拠は対象物、手元、移動、消失、持ち去り疑いに限定してください。
疑いあり true にしてよいのは、人物ではない同一対象物が、前半フレームで存在し、その後に手元付近で移動・収納・消失した前後関係が見える場合だけです。
このPoCでは、対象物は机上や手元付近にある小さい持ち物です。バッグ、箱、容器、収納先、背景領域は target_object にしてはいけません。
バッグ、箱、容器、手元、人物領域、背景だけが動いて見える場合は true にしてはいけません。
一方で、小さい対象物が移動した後、表示されている後半フレーム内で見えなくなる場合は、収納先に入る瞬間が完全に見えなくても持ち去り疑いとして true を検討してください。
最終フレームの後に消えた、動画の外へ移動した、など表示されていない出来事を推測してはいけません。
対象物が最後まで同じ場所に残っている場合、または対象物の消失が確認できない場合は false または低い confidence にしてください。
判定できない場合は false または低い confidence にし、不確実性を limitations に書いてください。
privacy_sensitive_description_included は、人物属性や識別につながる記述を含めた場合だけ true にしてください。原則 false です。
必ず次のJSONオブジェクトだけを返してください。Markdownや説明文をJSONの外に出してはいけません。
{
  "unauthorized_object_interaction_suspected": true | false,
  "confidence": 0.0,
  "target_object": null,
  "evidence_frames": [0],
  "reason": "string",
  "privacy_sensitive_description_included": false,
  "limitations": "string"
}
"""


class VLMClient(Protocol):
    def analyze(
        self,
        grid_image_path: str | Path,
        frame_infos: list[FrameInfo],
        metadata: VideoMetadata,
        config: AnalyzeConfig,
    ) -> VLMResponse:
        ...


def _image_to_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None, timeout: float = 60.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _safe_parse_response(text: str) -> VLMResponse:
    try:
        return apply_research_constraints(VLMResponse.model_validate(_extract_json_object(text)))
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        return VLMResponse(
            unauthorized_object_interaction_suspected=False,
            confidence=0.0,
            target_object=None,
            evidence_frames=[],
            reason="VLM出力をJSONスキーマとして解釈できませんでした。",
            privacy_sensitive_description_included=False,
            limitations=f"Parse error: {exc}",
        )


def apply_research_constraints(response: VLMResponse) -> VLMResponse:
    """Keep backend output inside the PoC's non-person, non-container target definition."""

    if not response.unauthorized_object_interaction_suspected and response.confidence > 0.5:
        response = response.model_copy(
            update={
                "confidence": 0.5,
                "limitations": response.limitations
                + " PoC制約により、false判定のconfidenceは最大0.5に補正しました。",
            }
        )

    if response.target_object is None:
        return response
    target = response.target_object.lower()
    container_terms = ("bag", "バッグ", "container", "box", "箱", "容器", "収納", "storage")
    if not any(term in target for term in container_terms):
        return response
    return response.model_copy(
        update={
            "unauthorized_object_interaction_suspected": False,
            "confidence": min(response.confidence, 0.25),
            "target_object": None,
            "evidence_frames": [],
            "reason": (
                "VLMはバッグ・箱・容器などの収納先を対象物として返しました。"
                "このPoCでは収納先はtarget_objectにしないため、疑いあり判定を採用しません。"
            ),
            "limitations": response.limitations
            + " PoC制約により、バッグ・箱・容器を対象物にした出力はfalseへ補正しました。",
        }
    )


def build_user_prompt(frame_infos: list[FrameInfo], metadata: VideoMetadata, config: AnalyzeConfig) -> str:
    frame_summary = ", ".join(f"{item.frame_index}@{item.timestamp:.2f}s" for item in frame_infos)
    return (
        f"動画メタデータ: fps={metadata.fps:.2f}, duration={metadata.duration:.2f}s, "
        f"total_frames={metadata.total_frames}.\n"
        "入力は選択・マスク済みフレームのグリッド画像です。raw videoではありません。\n"
        "グリッド内の各セルは別々の時刻のフレームで、左上から右下へ時系列順に並んでいます。\n"
        f"sampling_method={config.sampling_method}, mask_method={config.mask_method}, "
        f"num_frames={len(frame_infos)}.\n"
        f"選択フレーム: {frame_summary}\n"
        "画像内の対象物、手元付近、対象物の移動や消失だけを根拠にしてください。"
        "バッグ、箱、容器、収納先、手元が見えるだけ、または人物領域だけが移動するだけでは疑いありにしないでください。"
        "target_object は机上や手元付近の小さい持ち物に限定し、バッグや容器を target_object にしないでください。"
        "小さい対象物が移動した後、表示されている後半フレーム内で見えなくなる場合は、収納先に入る瞬間が完全に見えなくても true を検討してください。"
        "最終フレーム後の出来事は推測せず、表示済みフレーム内で消失が見えない場合は false にしてください。"
        "対象物が最初から最後まで残っている、または対象物の消失が確認できない場合は false にしてください。"
        "人物の身元や属性は推定せず、許可外物品操作疑いの有無をJSONで返してください。"
    )


class LegacyMockClient:
    """Deterministic local backend for full-pipeline verification."""

    def analyze(
        self,
        grid_image_path: str | Path,
        frame_infos: list[FrameInfo],
        metadata: VideoMetadata,
        config: AnalyzeConfig,
    ) -> VLMResponse:
        del grid_image_path, metadata
        frame_count = len(frame_infos)
        sampling = getattr(config.sampling_method, "value", config.sampling_method)
        mask = getattr(config.mask_method, "value", config.mask_method)
        has_motion_focus = sampling in {"motion", "hybrid", "event_window"}
        suspected = has_motion_focus and frame_count >= 3
        confidence = 0.62 if suspected else 0.28
        if mask in {"lower_body_only", "object_area_only", "background_blur_with_roi"}:
            confidence = max(0.1, confidence - 0.08)
        evidence = [item.frame_index for item in frame_infos[: min(3, frame_count)]] if suspected else []
        return VLMResponse(
            unauthorized_object_interaction_suspected=suspected,
            confidence=round(confidence, 2),
            target_object="赤い矩形状の対象物" if suspected else None,
            evidence_frames=evidence,
            reason=(
                "Mock判定です。変化重視のサンプリングで複数フレームが選ばれているため、"
                "対象物の移動または消失を確認する比較実験用の疑いあり結果を返しています。"
                if suspected
                else "Mock判定です。入力条件では対象物の移動や消失を示す十分な根拠を仮定していません。"
            ),
            privacy_sensitive_description_included=False,
            limitations=(
                "legacy mock clientは画像内容を理解しません。分類精度の評価ではなく、"
                "パイプライン接続と比較実験の動作確認に使ってください。"
            ),
        )


class OllamaVLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def analyze(
        self,
        grid_image_path: str | Path,
        frame_infos: list[FrameInfo],
        metadata: VideoMetadata,
        config: AnalyzeConfig,
    ) -> VLMResponse:
        if not self.settings.ollama_enabled:
            return self._disabled_response()
        selected_model = config.vlm_model or self.settings.ollama_model
        try:
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(frame_infos, metadata, config),
                        "images": [_image_to_base64(grid_image_path)],
                    },
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            }
            url = f"{self.settings.ollama_host.rstrip('/')}/api/chat"
            data = _post_json(url, payload, timeout=self.settings.ollama_timeout_sec)
            message = data.get("message", {})
            return _safe_parse_response(str(message.get("content", "")))
        except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as exc:
            return _backend_error_response("Ollama", exc)

    @staticmethod
    def _disabled_response() -> VLMResponse:
        return VLMResponse(
            unauthorized_object_interaction_suspected=False,
            confidence=0.0,
            target_object=None,
            evidence_frames=[],
            reason="Ollama backend is disabled.",
            privacy_sensitive_description_included=False,
            limitations="Set OLLAMA_ENABLED=true and configure OLLAMA_HOST/OLLAMA_MODEL to use this backend.",
        )


class OpenAICompatibleVLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def analyze(
        self,
        grid_image_path: str | Path,
        frame_infos: list[FrameInfo],
        metadata: VideoMetadata,
        config: AnalyzeConfig,
    ) -> VLMResponse:
        if not self.settings.openai_compatible_enabled:
            return self._disabled_response()
        if not self.settings.openai_compatible_base_url or not self.settings.openai_compatible_model:
            return self._disabled_response("Base URL or model is not configured.")

        try:
            image_b64 = _image_to_base64(grid_image_path)
            payload = {
                "model": self.settings.openai_compatible_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": build_user_prompt(frame_infos, metadata, config)},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            }
            headers: dict[str, str] = {}
            if self.settings.openai_compatible_api_key:
                headers["Authorization"] = f"Bearer {self.settings.openai_compatible_api_key}"
            url = f"{self.settings.openai_compatible_base_url.rstrip('/')}/chat/completions"
            data = _post_json(url, payload, headers=headers)
            text = data["choices"][0]["message"]["content"]
            return _safe_parse_response(str(text))
        except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as exc:
            return _backend_error_response("OpenAI-compatible", exc)

    @staticmethod
    def _disabled_response(reason: str = "OpenAI-compatible backend is disabled.") -> VLMResponse:
        return VLMResponse(
            unauthorized_object_interaction_suspected=False,
            confidence=0.0,
            target_object=None,
            evidence_frames=[],
            reason=reason,
            privacy_sensitive_description_included=False,
            limitations=(
                "Set OPENAI_COMPATIBLE_ENABLED=true and configure "
                "OPENAI_COMPATIBLE_BASE_URL/API_KEY/MODEL to use this backend."
            ),
        )


def _backend_error_response(backend_name: str, exc: BaseException) -> VLMResponse:
    return VLMResponse(
        unauthorized_object_interaction_suspected=False,
        confidence=0.0,
        target_object=None,
        evidence_frames=[],
        reason=f"{backend_name} backend request failed.",
        privacy_sensitive_description_included=False,
        limitations=str(exc),
    )


def create_vlm_client(backend: str | VLMBackend, settings: Settings | None = None) -> VLMClient:
    selected = VLMBackend(backend)
    if selected == VLMBackend.MOCK:
        return LegacyMockClient()
    if selected == VLMBackend.OLLAMA:
        return OllamaVLMClient(settings)
    if selected == VLMBackend.OPENAI_COMPATIBLE:
        return OpenAICompatibleVLMClient(settings)
    msg = f"Unsupported VLM backend: {backend}"
    raise ValueError(msg)
