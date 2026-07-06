# 仕様: 静的テクスチャ(机の木目等)の幽霊トラックによる偽「把持移動」判定の抑制

## 1. 背景と症状

机の木目などの静的テクスチャが物体トラックとして誤登録され、その上で手を動かすと
「把持して移動」判定(is_grasping=True)が誤発火する。

## 2. 根本原因(現行コードの参照)

対象: `grip_detector/object_tracking.py`, `grip_detector/models.py`

1. **幽霊トラックの発生**: YOLOの低confidence誤検出(既定 `yolo_confidence=0.35`)、
   またはmotion検出時の手の影・照明変化blobにより、木目領域にトラックが生まれる。
   木目は自己相似テクスチャのため `_template_match`(閾値0.58)がほぼ常に成立し、
   トラックが `object_max_missed_frames` で消えずに生き残る。
2. **偽の「移動」**: `_best_detection_index` はIoUゼロでも距離のみ
   (`object_association_distance_px=110` 以内、採用閾値0.20)で対応付けるため、
   手の動きに伴う手近傍の新検出が木目トラックへ対応付けられ bbox がジャンプする。
   これにより `object_displacement_px()` が閾値35pxを超え `motion_score=1` になる。
   類似木目へのテンプレートドリフトでも同様。
3. **本質的な穴**: 現行の移動証拠は「bboxが動いた」ことしか見ておらず、
   「**同一の物体**が元の場所から**居なくなり**、移動先に**現れた**」ことを検証していない。

## 3. 変更概要(要件一覧)

| ID | 内容 | 目的 |
|----|------|------|
| R1 | トラックの「静定(settled)」ゲートと再ベースライン | 生まれたて・ジッタ中のトラックを把持対象から除外 |
| R2 | 空き地チェック(vacancy check) | 「元の場所がまだ物体の見た目のまま」なら移動を偽と判定 |
| R3 | 移動先の同一性チェック(identity check) | bboxジャンプ・ドリフトを移動と認めない |
| R4 | 手近傍での新規トラック生成抑制 | 手の影・動きに誘発された幽霊トラックの発生防止 |
| R5 | 検出・対応付けの厳格化(設定変更) | 誤検出の入口を減らす |

R1〜R3が本命。R4/R5は補助。

## 4. データモデル変更

### 4.1 `TrackedObject`(models.py)に追加

```python
# 生成時(または再ベースライン時)の見た目。以後の _update_track では更新しない。
initial_template: Optional["np.ndarray"] = None

# 静定判定
settled: bool = False           # 把持対象として有効か
stationary_frames: int = 0      # 連続静止フレーム数

# 移動検証の結果(直近フレーム)。CSV/描画のデバッグ用
vacancy_similarity: float = 1.0   # 元位置と initial_template の一致度
identity_similarity: float = 1.0  # 現bboxと initial_template の一致度
```

注意: 既存の `template` は毎フレーム更新される追跡用。`initial_template` は
移動検証用の不変スナップショットであり、**別物として保持する**こと。

### 4.2 `DetectorConfig`(models.py)に追加

```python
# --- R1: 静定ゲート ---
# この速度未満を「静止」とみなす
object_settle_speed_px_s: float = 12.0
# 連続静止がこのフレーム数に達したら settled=True
object_settle_frames: int = 30
# 再ベースライン時、手がこの距離(px)以内にあれば再ベースラインしない
object_rebaseline_hand_distance_px: float = 120.0

# --- R2/R3: 移動検証 ---
# 元位置の一致度がこの値以上なら「元の場所に物体が残っている」= 移動は偽
object_vacancy_similarity_threshold: float = 0.60
# 移動先の一致度がこの値未満なら「同一物体ではない」= 移動は偽
object_identity_similarity_threshold: float = 0.45

# --- R4: 生成抑制 ---
# 手bbox(拡張後)からこの距離(px)以内では新規トラックを生成しない
object_birth_hand_distance_px: float = 90.0
```

### 4.3 `ObjectEvidence`(models.py)に追加(デバッグ可視化用)

```python
vacancy_similarity: float = 1.0
identity_similarity: float = 1.0
motion_valid: bool = False
```

## 5. 要件詳細

### R1: 静定ゲートと再ベースライン(object_tracking.py)

1. `_update_track` 内で速度を計算した直後に静止カウントを更新する:
   - `object_speed_px_s(track) < object_settle_speed_px_s` なら `stationary_frames += 1`、
     そうでなければ `stationary_frames = 0`。
   - `stationary_frames >= object_settle_frames` で `settled = True`。
2. **再ベースライン**: `settled` への遷移時(静止が規定フレーム続いた瞬間)、かつ
   全手bboxの中心からトラックbbox中心までの距離が
   `object_rebaseline_hand_distance_px` 超のとき:
   - `initial_bbox = bbox`
   - `initial_template = 現フレームgrayのbbox切り出し`
   - これにより「置き直した物体を再度掴んで移動」が正しく検出できる
     (現行は initial_bbox が最初の位置のまま固定で、displacement が張り付く)。
   - 手が近い間は再ベースラインを保留する(把持したまま静止しているケースを保護)。
3. `evaluate_hand_object_evidence` では **`settled=False` のトラックをスキップ**する
   (best候補の対象にしない)。
4. トラック生成時(`add_manual_track`)は `settled=False`、
   `initial_template = template` で初期化する。ただし `source="manual"`
   (ユーザー手動指定)の場合のみ `settled=True` で開始してよい。

### R2: 空き地チェック(object_tracking.py)

移動が本物なら、元の場所は空くはず。

1. 新関数を追加:

```python
def region_similarity(
    frame_gray: "np.ndarray",
    bbox: BBox,
    template: Optional["np.ndarray"],
) -> float:
    """bbox領域を切り出し、template と同サイズへリサイズして
    cv2.matchTemplate(TM_CCOEFF_NORMED) の単一値を返す。
    template が None または領域が小さすぎる場合は 1.0(判定不能=保守的に非移動扱い)。"""
```

2. 毎フレームの `update()` 末尾(トラック更新後)で、
   `displacement_px > object_motion_displacement_threshold_px` のトラックについて:
   - `vacancy_similarity = region_similarity(frame_gray, initial_bbox, initial_template)`
   - `vacancy_similarity >= object_vacancy_similarity_threshold` の場合、
     元位置にまだ同じ見た目が残っている = 木目等の偽移動。**移動を無効化**する。
3. 計算コスト削減のため、displacement が閾値未満のトラックでは計算しない
   (その場合 `vacancy_similarity = 1.0` のままでよい。移動証拠自体がないため)。

### R3: 移動先の同一性チェック(object_tracking.py)

1. R2と同じフレームタイミングで、displacement が閾値を超えたトラックについて:
   - `identity_similarity = region_similarity(frame_gray, bbox, initial_template)`
   - `identity_similarity < object_identity_similarity_threshold` の場合、
     現在のbboxは元の物体と見た目が違う = 対応付けジャンプ/ドリフト。**移動を無効化**する。

### R2/R3の統合: `object_motion_score` の変更

```python
motion_valid = (
    vacancy_similarity < config.object_vacancy_similarity_threshold   # 元位置が空いた
    and identity_similarity >= config.object_identity_similarity_threshold  # 同一物体
)
raw_motion = max(speed_score, displacement_score)  # 現行ロジック
motion_score = raw_motion if motion_valid else 0.0
```

- speedのみで displacement 未達の場合(掴んだ直後の動き出し)は、
  R2/R3の類似度が未計算(=1.0)になり motion_valid=False で移動が出遅れる。
  これを避けるため、**speed_score が0を超えた時点でもR2/R3を計算**すること
  (計算トリガーは「displacement または speed のいずれかが閾値の50%超」とする)。
- `ObjectEvidence` に `vacancy_similarity` / `identity_similarity` / `motion_valid`
  を格納し、`csv_logging.py` と `drawing.py` のデバッグ表示に追加する(任意だが推奨)。

### R4: 手近傍での新規トラック生成抑制(object_tracking.py `update()`)

未対応検出から新規トラックを作るループで、検出bbox中心と各手bbox
(`hand_bbox_from_landmarks`、margin=`object_hand_mask_padding_px`)の距離が
`object_birth_hand_distance_px` 以内ならスキップする。

トレードオフ: 手で置いた直後の物体は手が離れるまで登録されないが、
R1の静定ゲートによりどのみち即時把持対象にはならないため許容する。

### R5: 検出・対応付けの厳格化(設定変更のみ)

1. `yolo_confidence` 既定値を `0.35 → 0.50` に引き上げ。
2. `yolo_ignore_classes` に `"dining table"`, `"bench"`, `"bed"`, `"couch"` を追加。
3. `_best_detection_index`: IoUがゼロ(=重なりなし)の候補は、
   距離スコアに `0.80` ではなく `0.55` を乗じる(距離のみでの遠距離ジャンプを抑制)。
   ※ R3で偽移動自体は無効化されるため、これは保険。対応付けロジックの
   大規模な変更は行わない。

## 6. 受け入れ条件(手動テストシナリオ)

| # | シナリオ | 期待結果 |
|---|---------|---------|
| 1 | 机の木目の上で手を握って左右に動かす(物体なし) | is_grasping にならない。幽霊トラックが生まれても settled 前に把持対象外、移動しても vacancy check で motion_valid=False |
| 2 | コップを机に置き、約1秒(settle_frames)静止後に掴んで持ち上げて移動 | 従来どおり把持移動判定が出る。判定遅延の増加は settle 待ちを除き体感なし |
| 3 | コップを別の場所へ置いて手を離し、再度掴んで移動 | 再ベースラインにより2回目も判定が出る |
| 4 | 物体の上空で握って手だけ動かす(接触なし) | 判定なし(既存の持ち上げ検証と本仕様の併用) |
| 5 | 物体を掴んだまま静止(移動なし) | OBJECT_CONTACT_NO_MOVE のまま。誤って settled 再ベースラインされない(手が近いため保留) |

## 7. 非目標

- 対応付け(トラッキング)アルゴリズム自体の刷新は行わない。
- 深度カメラ・複数カメラ対応は本仕様の範囲外。
- YOLOモデルの変更・再学習は行わない。

## 8. 変更対象ファイル

- `grip_detector/models.py`: TrackedObject / ObjectEvidence / DetectorConfig への項目追加
- `grip_detector/object_tracking.py`: R1〜R5の本体、`region_similarity` 追加、
  `object_motion_score` の署名変更(vacancy/identity を受け取る)
- `grip_detector/csv_logging.py`, `grip_detector/drawing.py`: 新フィールドの出力(任意)
- `grip_detector/cli.py`: 新設定値のCLI引数追加(既存パターンに合わせる)
