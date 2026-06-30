from __future__ import annotations

import sys
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

from .constants import MODEL_URL
from .runtime import cv2, mp, np

# ---------------------------------------------------------------------------
# 依存関係とモデル取得
# ---------------------------------------------------------------------------

def require_dependencies() -> None:
    """
    必須パッケージを確認し、足りない場合は具体的な導入コマンドを示します。
    """

    missing: List[str] = []

    if cv2 is None:
        missing.append("opencv-python")
    if mp is None:
        missing.append("mediapipe")
    if np is None:
        missing.append("numpy")

    if missing:
        package_list = " ".join(missing)
        raise RuntimeError(
            "必須パッケージが不足しています。\n"
            f"次を実行してください:\n"
            f"  {sys.executable} -m pip install {package_list}"
        )


def download_with_progress(url: str, destination: Path) -> None:
    """
    モデルを一時ファイルへダウンロードし、完了後に正式名へ置き換えます。

    途中で停止しても壊れたモデルファイルが残りにくい構成です。
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    # 同じディレクトリに一時ファイルを作り、os.replaceで原子的に置換します。
    with tempfile.NamedTemporaryFile(
        prefix="hand_landmarker_",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        print(f"MediaPipe Hand Landmarkerモデルを取得します:\n  {url}")

        def report_progress(
            block_count: int,
            block_size: int,
            total_size: int,
        ) -> None:
            if total_size <= 0:
                return

            downloaded = min(block_count * block_size, total_size)
            percentage = 100.0 * downloaded / total_size

            print(
                f"\rダウンロード中: {percentage:6.2f}% "
                f"({downloaded / 1024 / 1024:.1f} MB)",
                end="",
                flush=True,
            )

        urllib.request.urlretrieve(
            url,
            temporary_path,
            reporthook=report_progress,
        )
        print()

        # 異常に小さいファイルはHTMLエラー等の可能性があるため拒否します。
        if temporary_path.stat().st_size < 1_000_000:
            raise RuntimeError(
                "取得したモデルファイルが異常に小さいため使用を中止しました。"
            )

        os.replace(temporary_path, destination)

    except (urllib.error.URLError, OSError, RuntimeError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Hand Landmarkerモデルの取得に失敗しました。\n"
            f"URL: {url}\n"
            f"保存先: {destination}\n"
            f"原因: {error}"
        ) from error


def ensure_model(model_path: Path) -> Path:
    """
    モデルが存在すればそのまま使い、なければ公式配布元から取得します。
    """

    if model_path.exists() and model_path.stat().st_size >= 1_000_000:
        return model_path

    download_with_progress(MODEL_URL, model_path)
    return model_path
