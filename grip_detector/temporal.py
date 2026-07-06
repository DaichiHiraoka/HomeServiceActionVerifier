from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .models import DetectorConfig, TemporalGripState

# ---------------------------------------------------------------------------
# 時系列判定
# ---------------------------------------------------------------------------

class GripTemporalFilter:
    """
    手ごとの時系列状態を管理します。

    ヒステリシス:
        開始閾値 > 解除閾値

    これにより、閾値付近で通常状態と把持中移動状態が高速反転する現象を抑えます。
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.states: Dict[str, TemporalGripState] = {}

    def reset(self) -> None:
        """全手の状態を初期化します。"""
        self.states.clear()

    def update(
        self,
        hand_id: str,
        raw_score: float,
        raw_mode: str,
        timestamp_sec: float,
    ) -> TemporalGripState:
        """
        1手分の生スコアを受け取り、平滑化と状態遷移を行います。
        """

        state = self.states.setdefault(hand_id, TemporalGripState())

        # 初回だけ、生スコアをそのまま平滑化スコアの初期値にします。
        if state.last_update_time is None:
            state.smoothed_score = raw_score
            delta_time = 0.0
        else:
            delta_time = max(0.0, timestamp_sec - state.last_update_time)

            # フレームレートに依存しにくいEMA係数を時定数から計算します。
            # alpha = 1 - exp(-dt / tau)
            tau = max(self.config.ema_time_constant_sec, 1e-6)
            alpha = 1.0 - math.exp(-delta_time / tau)

            state.smoothed_score += alpha * (
                raw_score - state.smoothed_score
            )

        state.last_update_time = timestamp_sec
        state.last_seen_time = timestamp_sec

        # 現在が非把持中移動状態の場合、開始閾値を一定時間超えたか確認します。
        if not state.is_grasping:
            state.exit_candidate_since = None

            if state.smoothed_score >= self.config.enter_threshold:
                if state.enter_candidate_since is None:
                    state.enter_candidate_since = timestamp_sec

                elapsed = timestamp_sec - state.enter_candidate_since

                if elapsed >= self.config.enter_delay_sec:
                    state.is_grasping = True
                    state.mode = raw_mode
                    state.enter_candidate_since = None
            else:
                # 閾値を下回ったら、継続時間の計測をやり直します。
                state.enter_candidate_since = None
                state.mode = "NONE"

        # 現在が把持中移動状態の場合、解除閾値を一定時間下回ったか確認します。
        else:
            state.enter_candidate_since = None

            if state.smoothed_score <= self.config.exit_threshold:
                if state.exit_candidate_since is None:
                    state.exit_candidate_since = timestamp_sec

                elapsed = timestamp_sec - state.exit_candidate_since

                if elapsed >= self.config.exit_delay_sec:
                    state.is_grasping = False
                    state.mode = "NONE"
                    state.exit_candidate_since = None
            else:
                # 解除条件を満たしていない間は、現在の優勢方式へ更新します。
                state.exit_candidate_since = None
                state.mode = raw_mode

        return state

    def remove_missing_hands(
        self,
        detected_hand_ids: Iterable[str],
        timestamp_sec: float,
    ) -> None:
        """
        一定時間見失った手の状態を削除します。

        手を画面外へ出した後、再入場時に以前の把持中移動状態が残ることを防ぎます。
        """

        detected = set(detected_hand_ids)
        delete_targets: List[str] = []

        for hand_id, state in self.states.items():
            if hand_id in detected:
                continue

            if state.last_seen_time is None:
                delete_targets.append(hand_id)
                continue

            missing_duration = timestamp_sec - state.last_seen_time

            if missing_duration >= self.config.missing_reset_sec:
                delete_targets.append(hand_id)

        for hand_id in delete_targets:
            del self.states[hand_id]
