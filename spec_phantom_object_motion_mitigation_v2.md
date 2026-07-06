# 仕様v2: 静的テクスチャ幽霊トラックによる偽「把持移動」判定の抑制

v1 (`spec_phantom_object_motion_mitigation.md`) の再設計版。v1は実装すると把持判定が一切出なくなったため、失敗分析に基づき全面改訂する。

## 0. v1の失敗分析(必読)

1. **静定ゲートが全トラックを殺した**: v1は「速度 < 閾値」の連続で settled を判定した。
   速度は隣接フレーム中心の差分であり、静止物体でもYOLO bboxは毎フレーム1〜5px揺れる
   → 30fpsで30〜150px/s に化ける → `stationary_frames` が毎回リセット → settled が一度も
   True にならない → `evaluate_hand_object_evidence` が全トラックをスキップ →
   evidence が常に空 → スコア恒常0。**判定が全く出ない直接原因。**
2. **motion_valid のANDゲート**: 既定False + 「vacancy合格 AND identity合格」の要求。
   identity(移動先と初期テンプレートの一致)は把持中の手の遮蔽・回転で実物でも
   容易に不合格になる。二重に判定を殺していた。

## 1. v2の設計原則

1. **ゲートは「拒否条件」として設計する**。正例(本物の把持移動)が通過できることを
   デフォルトとし、幽霊トラック特有のシグネチャ(元位置に同じ見た目が残っている)を
   検出した場合のみ拒否する。判定不能なら通す。
2. **静定判定は速度を使わない**。位置ベース(アンカー半径内滞留)で判定する。
3. **接触評価はゲートに依存させない**。settled前でも OBJECT_CONTACT_NO_MOVE は出す。
   ゲートは motion 経路のみに適用する。
4. **デバッグ可視化は必須要件**。どのゲートで落ちたかを画面で即特定できること。

## 2. データモデル変更

### 2.1 `TrackedObject`(models.py)に追加

```python
# 移動検証用の不変スナップショット。生成時と再ベースライン時のみ更新。
# 既存の template(毎フレーム更新の追跡用)とは別物として保持する。
initial_template: Optional["np.ndarray"] = None

# 静定判定(位置ベース)
anchor_center: Optional[Tuple[float, float]] = None
stationary_frames: int = 0
settled: bool = False   # 一度Trueになったらトラック削除まで False に戻さない(sticky)

# デバッグ用(直近フレームの値)
vacancy_similarity: float = 0.0
identity_similarity: float = 0.0   # ログ表示のみ。判定には一切使わない
motion_valid: bool = True          # 既定は True(= 移動を認める)
```

### 2.2 `DetectorConfig`(models.py)に追加

```python
# --- 静定判定(位置ベース) ---
# アンカーからの許容半径 = max(min_px, ratio * bbox対角長)
object_settle_radius_ratio: float = 0.15
object_settle_radius_min_px: float = 8.0
# 半径内に留まった連続フレーム数がこの値に達したら settled=True
object_settle_frames: int = 15
# 再ベースライン時、手bbox中心がこの距離(px)以内なら再ベースラインを保留
object_rebaseline_hand_distance_px: float = 120.0

# --- vacancy拒否 ---
# 元位置と initial_template の一致度がこの値以上なら偽移動として拒否
object_vacancy_similarity_threshold: float = 0.72

# --- 幽霊トラック発生抑制(v1 R4を継続) ---
object_birth_hand_distance_px: float = 90.0
```

既存値の変更: `object_max_missed_frames: 24 → 48`
(把持中は手の遮蔽で検出が途切れやすく、トラックが死ぬと判定不能になるため延命)

### 2.3 `ObjectEvidence`(models.py)に追加

```python
settled: bool = False
vacancy_similarity: float = 0.0
identity_similarity: float = 0.0
motion_valid: bool = True
```

## 3. 要件詳細

### R1: 位置ベースの静定判定(object_tracking.py `_update_track`)

```python
new_center = bbox_center(track.bbox)
if track.anchor_center is None:
    track.anchor_center = new_center

diag = (bbox幅**2 + bbox高**2) ** 0.5
radius = max(config.object_settle_radius_min_px,
             config.object_settle_radius_ratio * diag)

if bbox中心とanchor_centerの距離 <= radius:
    track.stationary_frames += 1
else:
    track.stationary_frames = 0
    track.anchor_center = new_center   # アンカーを現在位置へ再設定

if track.stationary_frames >= config.object_settle_frames:
    track.settled = True   # sticky: 以後どれだけ動いても False に戻さない
```

- ジッタは半径内に収まるためリセットされない。瞬間速度は静定判定に使わない。
- **settled は sticky**。物体が動いても維持する(v1失敗分析の教訓)。

### R2: 再ベースライン(R1と同じ箇所)

`stationary_frames` がちょうど `object_settle_frames` に達したフレーム
(初回静定および移動後の再静定)で、全手bbox中心とトラックbbox中心の距離が
`object_rebaseline_hand_distance_px` を超える場合のみ:

- `initial_bbox = bbox`
- `initial_template = 現フレームgrayのbbox切り出し`
- `motion_valid = True` にリセット

手が近い間は保留する(把持したまま静止しているケースを誤って基準化しない)。
これにより「置き直した物体を再度掴んで移動」も検出できる。

### R3: vacancy拒否(object_tracking.py `update()` 末尾)

v1の `region_similarity(frame_gray, bbox, template)` をそのまま使う
(切り出し→テンプレートサイズへリサイズ→`TM_CCOEFF_NORMED` の単一値)。

```python
displacement = object_displacement_px(track)
speed = object_speed_px_s(track)
should_check = (
    displacement > config.object_motion_displacement_threshold_px * 0.5
    or speed > config.object_motion_speed_threshold_px_s * 0.5
)
if should_check and track.initial_template is not None:
    track.vacancy_similarity = region_similarity(
        frame_gray, track.initial_bbox, track.initial_template)
    track.identity_similarity = region_similarity(
        frame_gray, track.bbox, track.initial_template)  # ログ専用
    # 元位置に同じ見た目が残っている = 木目等の偽移動 → 拒否
    track.motion_valid = (
        track.vacancy_similarity < config.object_vacancy_similarity_threshold
    )
else:
    # 判定材料がない場合は移動を認める(正例を落とさない)
    track.motion_valid = True
```

- **identity_similarity は計算してログに出すだけ。motion_valid には使わない。**
- 木目幽霊トラックは initial_template が必ず存在し、元位置の見た目が変わらないため
  vacancy_similarity が高止まりし、確実に拒否される。
- 本物の移動では元位置が空くか手で覆われるため vacancy_similarity は低くなり通過する。

### R4: motion_score へのゲート適用(object_tracking.py)

```python
raw_motion = max(speed_score, displacement_score)   # v0と同じ
motion_score = raw_motion if (track.settled and track.motion_valid) else 0.0
```

### R5: 接触評価はゲート不問(object_tracking.py `evaluate_hand_object_evidence`)

- **settled=False のトラックをスキップしない**(v1からの変更)。
  全トラックで contact_score を計算し、best選定も従来どおり行う。
- settled / motion_valid は motion_score(R4)経由でのみ効く。
- これにより「握って触れている」= OBJECT_CONTACT_NO_MOVE は settled 前でも
  必ず表示され、障害時の切り分けが画面上でできる。

### R6: 手近傍での新規トラック生成抑制(v1 R4と同じ)

未対応検出からの新規トラック生成時、検出bbox中心が手bbox
(margin=`object_hand_mask_padding_px`)から `object_birth_hand_distance_px`
以内ならスキップ。既存トラックの対応付け・更新は制限しない。

### R7: デバッグ可視化(必須。任意ではない)

- `drawing.py`: 各トラックbboxの脇に
  `S:{settled} f:{stationary_frames} v:{vacancy_similarity:.2f} mv:{motion_valid}`
  を描画。evidence 行に motion_score / motion_valid を追加。
- `csv_logging.py`: `object_settled`, `object_vacancy_similarity`,
  `object_identity_similarity`, `object_motion_valid` 列を追加。

## 4. 受け入れ条件

| # | シナリオ | 期待結果 |
|---|---------|---------|
| 1 | **回帰チェック(v1の失敗)**: 机上の静止物体が、起動から約1秒以内に画面表示で settled=True になる | 必須。これが通らない場合、以降の調整をせず静定判定の実装を疑う |
| 2 | 物体を掴んで静止(移動なし) | OBJECT_CONTACT_NO_MOVE が表示される(settled前でも) |
| 3 | 物体を掴んで持ち上げて移動 | 把持移動判定が出る |
| 4 | 机の木目の上で手を握って動かす(物体なし) | 判定なし。幽霊トラックが動いても vacancy拒否で motion_valid=False になることを画面で確認 |
| 5 | 物体を別の場所へ置いて手を離し、再度掴んで移動 | 再ベースライン後、2回目も判定が出る |
| 6 | 物体の上空で握って手だけ動かす | 判定なし |

トラブル時の切り分け手順: (a) トラックbboxが出ているか → 出なければ検出/生成抑制の問題、
(b) settled=True か → Falseなら静定判定、(c) OBJECT_CONTACT_NO_MOVE が出るか →
出なければ接触評価、(d) motion_valid → Falseなら vacancy閾値、の順に画面表示で確認する。

## 5. 非目標

- 対応付け(トラッキング)アルゴリズムの刷新、深度・複数カメラ、YOLO再学習は範囲外。
- v1のR5(yolo_confidence引き上げ、ignore_classes追加、対応付け係数変更)は
  任意の補助策として維持してよいが、本仕様の合否には含めない。

## 6. 変更対象ファイル

- `grip_detector/models.py`: TrackedObject / ObjectEvidence / DetectorConfig 項目追加
- `grip_detector/object_tracking.py`: R1〜R6、`region_similarity` 追加
- `grip_detector/drawing.py`, `grip_detector/csv_logging.py`: R7(必須)
- `grip_detector/cli.py`: 新設定値のCLI引数(既存パターンに合わせる)
