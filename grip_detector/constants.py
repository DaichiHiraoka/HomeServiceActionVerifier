from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# MediaPipe公式配布モデル
# ---------------------------------------------------------------------------

# Google AI Edge が公式配布している Hand Landmarker の full モデルです。
# 初回実行時だけユーザーのキャッシュディレクトリへ自動保存します。
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def get_default_cache_dir(app_name: str) -> Path:
    """Return an OS-appropriate user cache directory for this application."""

    if os.name == "nt":
        cache_root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if cache_root:
            return Path(cache_root) / app_name

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / app_name

    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root) / app_name

    return Path.home() / ".cache" / app_name


# モデルの既定保存先です。
# Pythonファイルの隣へ勝手にファイルを置かず、OS標準に近いユーザーキャッシュへ保存します。
DEFAULT_MODEL_PATH = get_default_cache_dir(
    "skeleton_grip_detector"
) / "hand_landmarker.task"


# ---------------------------------------------------------------------------
# MediaPipe Hand Landmarker の21点番号
# ---------------------------------------------------------------------------

# 21点の意味:
#  0: 手首
#  1-4: 親指 CMC, MCP, IP, TIP
#  5-8: 人差し指 MCP, PIP, DIP, TIP
#  9-12: 中指 MCP, PIP, DIP, TIP
# 13-16: 薬指 MCP, PIP, DIP, TIP
# 17-20: 小指 MCP, PIP, DIP, TIP

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


# 各指の4点を、根元から先端までの順に定義します。
# 親指は関節構造が異なるため、主な4指とは別処理にします。
FINGER_JOINTS: Dict[str, Tuple[int, int, int, int]] = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


# 骨格を描画するための接続関係です。
# MediaPipeの21点番号を線で結び、手骨格を画面上へ表示します。
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # 掌
    (WRIST, THUMB_CMC),
    (WRIST, INDEX_MCP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, PINKY_MCP),
    (PINKY_MCP, WRIST),

    # 親指
    (THUMB_CMC, THUMB_MCP),
    (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP),

    # 人差し指
    (INDEX_MCP, INDEX_PIP),
    (INDEX_PIP, INDEX_DIP),
    (INDEX_DIP, INDEX_TIP),

    # 中指
    (MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_DIP, MIDDLE_TIP),

    # 薬指
    (RING_MCP, RING_PIP),
    (RING_PIP, RING_DIP),
    (RING_DIP, RING_TIP),

    # 小指
    (PINKY_MCP, PINKY_PIP),
    (PINKY_PIP, PINKY_DIP),
    (PINKY_DIP, PINKY_TIP),
)


# OpenCVの矩形を (x, y, width, height) で扱います。
BBox = Tuple[int, int, int, int]


# 物体との接触判定で使う指先IDです。
FINGERTIP_IDS: Tuple[int, ...] = (
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
)
