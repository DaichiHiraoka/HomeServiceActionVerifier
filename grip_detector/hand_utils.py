from __future__ import annotations

from typing import Any, Sequence, Tuple

# ---------------------------------------------------------------------------
# MediaPipe結果の解釈
# ---------------------------------------------------------------------------

def get_handedness(
    hand_result: Sequence[Any],
) -> Tuple[str, float]:
    """
    MediaPipeの handedness 結果から左右ラベルと信頼度を取得します。
    """

    if not hand_result:
        return "Unknown", 0.0

    category = hand_result[0]

    # MediaPipeのバージョン差を吸収するため、候補属性を順番に確認します。
    name = (
        getattr(category, "category_name", None)
        or getattr(category, "display_name", None)
        or "Unknown"
    )
    score = float(getattr(category, "score", 0.0))

    return str(name), score


def build_hand_id(
    handedness: str,
    duplicate_count: int,
) -> str:
    """
    左右ラベルを時系列追跡IDとして使います。

    同一ラベルが同じフレームに複数出た場合のみ連番を付けます。
    """
    if duplicate_count == 0:
        return handedness

    return f"{handedness}_{duplicate_count + 1}"
