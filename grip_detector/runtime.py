from __future__ import annotations

# 外部依存は、未導入時にも説明付きエラーを出せるように try/except で読み込みます。
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
