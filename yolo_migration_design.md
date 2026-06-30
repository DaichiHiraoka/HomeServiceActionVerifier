# YOLOを利用した物体検出構成への移行設計案

この文書は、現在のデスクトップアプリを、固定bbox注釈ではなくYOLO系の物体検出とトラッキングを使う構成へ変更するための設計案である。

ここでは実装は行わない。現行仕様と連動して変更するための設計メモとして扱う。

この文書の責務は、動画から物体トラックと局所クロップを生成し、骨格系列と同期できる形へ変換するところまでである。逸脱判定の正式仕様は `current_system_technical_overview.md` に置く。

YOLO自体は違反を判定しない。YOLOは、逸脱判定に必要な物体ID、位置、移動、zone、可視状態を供給する知覚層である。

## 1. 目的

現在の `router_normal_img_7852` 試行データでは、ルーター本体のbboxを手動で固定している。

この方法は固定された単一動画では動作するが、研究システムとしては次の弱点がある。

- 対象物の位置を人間が先に決める必要がある
- 動画ごとにbboxを作り直す必要がある
- 物体が移動した場合に追跡できない
- 「拾う」「持ち出す」「隠す」などの検出に必要な物体軌跡を自動で作れない

そのため、次の構成に移行する。

```text
raw video
  -> object detector
  -> multi-object tracker
  -> object_tracks.csv
  -> object_crops/
  -> existing analyzer
  -> results.json / results.csv / summary.md
```

## 2. 採用方針

第一候補は **Ultralytics YOLO + BoT-SORT または ByteTrack** とする。

理由は次の通り。

- Pythonから扱いやすい
- 動画入力、物体検出、トラッキング、切り出し処理へつなげやすい
- 速度が出しやすく、卒研プロトタイプとして実装しやすい
- YOLOの検出結果を、現在の `object_tracks.csv` 形式に変換しやすい
- Ultralyticsのtrack modeは、物体IDを維持する用途を想定しており、BoT-SORTやByteTrackを選べる

現時点では、検出器そのものの新規性を主張しない。

本研究の主対象は、検出器の改良ではなく、次の部分である。

```text
骨格系列 + 物体軌跡 + 作業文脈
  -> 文脈上不自然な物体操作の検出
```

## 3. 候補技術の整理

| 候補 | 役割 | 長所 | 弱点 | この研究での扱い |
| --- | --- | --- | --- | --- |
| YOLO + BoT-SORT | 物体検出 + ID追跡 | 実装しやすい、動画に強い、速度が出やすい | 未学習の家庭内小物は弱い | 第一候補 |
| YOLO + ByteTrack | 物体検出 + ID追跡 | 途切れた検出を拾いやすい | 誤検出も拾う可能性がある | 比較候補 |
| YOLO-seg | 物体検出 + mask | bboxより正確に対象を切れる | 学習データが必要になりやすい | 後続候補 |
| RT-DETR | 物体検出 | 精度とリアルタイム性の候補 | YOLOより重い場合がある | 比較候補 |
| Grounding DINO | テキスト指定によるopen-set検出 | 未定義クラスを探しやすい | 重い、安定運用が難しい | 研究比較・補助候補 |
| SAM / SAM系 | セグメンテーション | 対象周辺を細かく切れる | 物体名やID追跡は別途必要 | 後段の補助候補 |

基本構成はYOLOでよい。

ただし、財布・鍵・スマホ・ルーターなど家庭内小物は、COCO等の一般クラスだけでは十分に検出できない可能性が高い。その場合は、YOLOをそのまま使うのではなく、研究対象に必要な物体クラスだけを追加学習する。

## 4. 推奨アーキテクチャ

### 4.1 全体構成

```text
data/raw_videos/<video>.mp4
  |
  +--> pose estimator
  |      └--> skeleton.csv
  |
  +--> YOLO detector
         |
         v
       tracker
         |
         +--> object_tracks.csv
         +--> object_crops/<object_id>/*.jpg
         +--> detection_debug.mp4

skeleton.csv + object_tracks.csv
  |
  v
time synchronizer
  |
  v
interaction event generator
  |
  +--> approach
  +--> contact
  +--> lift
  +--> carry
  +--> place
  +--> return
  +--> disappear
  +--> exit
  |
  v
object-centric state sequence
  |
  v
task policy matcher
  |
  v
deviation result + evidence + alert
```

### 4.2 既存アプリとの接続

入力契約は次の3種類を維持する。

- `context.json`: 作業票、許可zone、許可状態遷移
- `skeleton.csv`: 骨格・姿勢系列
- `object_tracks.csv`: YOLOとトラッカーによる物体系列

変更点は、`object_tracks.csv`を固定bboxではなくYOLOとトラッカーから生成することに加え、骨格系列との同期後に観測イベント列を生成することである。

```text
旧方式:
  手動イベント表
  -> 作業票との照合

新方式:
  骨格系列 + YOLOトラック
  -> 観測イベント自動生成
  -> 物体状態遷移
  -> 作業票の許可状態遷移との照合
```

### 4.3 YOLO出力から逸脱判定へ渡す情報

YOLOとトラッカーは、各物体について次を出力する。

| 出力 | 逸脱判定での用途 |
| --- | --- |
| `object_id` | 同じ物体の状態遷移を継続して追う |
| `label` | 作業対象物、工具、保護対象物との照合 |
| `bbox`、中心座標 | 手首・胴体との距離を計算 |
| `track_id` | 物体の同一性を維持 |
| `confidence` | 観測の信頼度を計算 |
| `zone` | 許可zone外移動を検知 |
| `visible` | 消失候補を検知 |
| `crop_path` | 毀損・開封などの前後画像比較 |

YOLOのクラスラベルだけで所有者は決まらない。同じ`phone`でも、住人のスマートフォンと作業者のスマートフォンは別物である。

そのため、作業前の物体登録または設置位置との対応付けによって、各`object_id`へ`target`、`worker`、`protected`、`unknown`の役割を付与する。

## 5. 追加する処理

### 5.1 動画解析ジョブ

新しく、動画から試行データを作る処理を追加する。

想定コマンド名:

```powershell
uv run home-service-verifier-prepare-video --video data/raw_videos/IMG_7852.mp4 --context data/trials/router_normal_img_7852/context.json --out data/trials/router_normal_img_7852
```

このコマンドは次を行う。

1. 動画を一定fpsで読み込む
2. 各フレームでYOLO検出を実行する
3. trackerで `track_id` を維持する
4. 検出結果を `object_tracks.csv` に保存する
5. 対象物周辺クロップを `object_crops/` に保存する
6. 必要ならデバッグ動画を保存する

### 5.2 出力する `object_tracks.csv`

既存の必須列は維持する。

```csv
timestamp,object_id,label,role,bbox_x,bbox_y,bbox_w,bbox_h,zone,visible,crop_path
```

YOLO移行後は、追加列を持たせる。

```csv
timestamp,
object_id,
label,
role,
bbox_x,
bbox_y,
bbox_w,
bbox_h,
zone,
visible,
crop_path,
detector,
tracker,
track_id,
confidence,
class_id,
source_frame
```

既存のCSVローダーは必要列だけ読めばよい。追加列は無視できるようにする。

### 5.3 `object_id` の設計

`object_id` は、検出器のclass名だけでは足りない。

例えば、スマホが2台ある場合、どちらも `phone` になる。

そのため、次の形式にする。

```text
<label>_<track_id>
```

例:

```text
router_3
phone_8
key_12
```

`track_id` が途切れた場合は別IDになる。これは追跡精度評価の対象にする。

## 6. 設定ファイルの分離

作業上の許可条件と、YOLOの実行設定は分離する。

```text
context.json
  作業対象、保護対象、許可zone、許可行動、許可状態遷移、復元条件

detection_config.json
  YOLOモデル、信頼度閾値、IoU閾値、対象ラベル、トラッカー設定

zones.json
  zone名と画面座標上のpolygon
```

`context.json`の例:

```json
{
  "task_name": "router_repair",
  "objects": {
    "router": {
      "role": "target",
      "allowed_zones": ["router_work_area"],
      "allowed_actions": ["approach", "contact", "lift", "carry", "place", "return"],
      "must_return": true
    },
    "wallet": {
      "role": "protected",
      "allowed_zones": ["private_desk"],
      "allowed_actions": ["approach"],
      "alert_on": ["contacted", "lifted", "carried", "disappeared", "exited"]
    }
  }
}
```

`detection_config.json`の例:

```json
{
  "model": "models/yolo/home_service_objects.pt",
  "tracker": "botsort.yaml",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.7,
  "sample_fps": 5,
  "target_labels": [
    "router",
    "lan_cable",
    "power_cable",
    "screwdriver",
    "wallet",
    "key",
    "phone",
    "tool_bag"
  ]
}
```

## 7. zone判定

YOLOは物体bboxを出すが、作業エリア内かどうかは判断しない。

`zones.json`に画面座標上のpolygonを定義し、物体bboxの中心点および人物の足元座標がどのpolygonに属するかを判定する。

```json
{
  "zones": [
    {
      "name": "router_work_area",
      "polygon": [[420, 190], [770, 190], [770, 580], [420, 580]]
    },
    {
      "name": "private_storage",
      "polygon": [[0, 550], [500, 550], [500, 1080], [0, 1080]]
    },
    {
      "name": "exit_area",
      "polygon": [[780, 0], [1080, 0], [1080, 1080], [780, 1080]]
    }
  ]
}
```

zone遷移は、次の逸脱検知に使う。

- 保護対象物が私的zoneから作業zoneへ移動した
- 作業対象物が許可zone外へ移動した
- 保護対象物を保持した人物が出口zoneへ移動した
- 保護対象物が工具バッグzone付近で消失した

## 7.1 骨格系列との統合によるイベント生成

YOLOの移動情報だけでは、「人が物体を持った」のか、「カメラ揺れや検出誤差でbboxが動いた」のかを区別できない。

そのため、各物体トラックを同時刻の骨格フレームと統合する。

| イベント | 必要な条件 |
| --- | --- |
| `approach` | 手先距離が継続的に減少し、手を伸ばす姿勢がある |
| `contact` | 手先がbbox近傍に一定フレーム滞在する |
| `lift` | 接触後に物体が移動し、手首と物体の移動が同期する |
| `carry` | 物体と手首の相対距離を保ったまま人物が移動する |
| `place` | 物体移動が停止し、手先が離れる |
| `return` | 物体が初期位置または指定位置へ戻る |
| `disappear` | `carry`中または身体近傍で物体が一定時間見えなくなる |
| `exit` | 物体または保持人物が出口zoneへ移動する |

イベントは単一フレームで確定せず、連続フレーム数と検出信頼度を満たした場合に成立させる。

## 7.2 作業票との照合

イベント生成後、物体ごとの状態遷移列を作る。

```text
wallet_8:
idle -> approached -> contacted -> lifted -> carried -> disappeared
```

これを`context.json`の許可状態遷移と照合する。

```text
wallet:
許可 = idle -> approached -> idle
禁止・要確認 = contacted / lifted / carried / disappeared / exited
```

この不一致が逸脱である。

したがって、逸脱検知は「YOLOが財布を検出したから」ではなく、次の条件が連続して成立した結果として行う。

```text
walletとして登録された物体である
  +
手が接近・接触した
  +
接触後に物体と手が同期して移動した
  +
身体側または出口側へ移動した
  +
復元されない、または消失した
```

判定結果には、逸脱種別、物体ID、状態遷移、検知時刻、観測信頼度を含める。

## 8. 学習データの方針

家庭内小物は、一般物体データセットだけでは検出性能が不足する可能性が高い。完成形では、研究対象クラスに対する追加学習と、独立したテストデータによる評価を行う。

対象クラスは次の通りとする。

1. `router`
2. `lan_cable`
3. `power_cable`
4. `screwdriver`
5. `wallet`
6. `key`
7. `resident_phone`
8. `tool_bag`

データセットは、次の変動を含める。

- 複数の室内背景
- 明るさと影の変化
- カメラ距離と角度の変化
- 手や身体による部分遮蔽
- 同一クラスの複数物体
- 静止、持ち上げ、運搬、設置、消失直前の状態
- 正常作業と逸脱系列の両方

学習用、検証用、テスト用は、同一動画の隣接フレームが別集合へ入らないよう動画単位または撮影セッション単位で分割する。これを行わない場合、背景や物体外観の重複によって性能が過大評価される。

データ量は固定本数を先に決めるのではなく、各クラスのテストRecallと、最終逸脱判定の見逃し率が収束するまで追加する。特に`wallet`、`key`、`phone`のRecallを重視する。

事前学習済みYOLOは比較基準として使用し、追加学習モデルとの差をクラス別Precision、Recall、mAP、および最終逸脱判定性能で評価する。

## 9. 評価指標

検出器単体と、最終判定を分けて評価する。

### 9.1 物体検出の評価

- Precision
- Recall
- mAP
- 小物クラスごとのRecall
- 誤検出率

特に重要なのはRecallである。

財布や鍵を見逃すと、その後の不審行動判定ができないためである。

### 9.2 トラッキングの評価

- ID switch数
- track fragment数
- IDF1
- HOTA

厳密なMOT評価が難しい場合は、卒研用には次の簡易指標でもよい。

- 対象物が最後まで同じIDで追跡された割合
- 対象物が見えているのに消失扱いされた回数
- 別物体とIDが入れ替わった回数

### 9.3 最終判定の評価

- `normal` / `review` / `suspicious` / `high_risk` の分類精度
- normalを不審扱いする誤警告率
- 不審行動をnormal扱いする見逃し率
- first_alert_time
- time_to_detection

この研究では、検出器のmAPだけでなく、最終的に「どれだけ早く、どれだけ妥当に不審な物体操作を検知できたか」を評価する。

## 10. 実装構成

完成形は、次の処理を一つの再現可能なパイプラインとして実装する。

### 10.1 知覚処理

- 動画またはカメラ入力
- 骨格・姿勢推定
- YOLO物体検出
- BoT-SORTまたはByteTrackによるID追跡
- zone判定
- 対象物周辺クロップ生成

### 10.2 時系列統合

- 骨格フレームと物体トラックの時刻同期
- 手首・胴体と物体の距離計算
- 手・物体の移動同期計算
- 物体IDの再対応付け
- `approach`から`exit`までのイベント生成
- 物体ごとの状態遷移列生成

### 10.3 作業文脈照合

- `context.json`から許可状態遷移グラフを生成
- 対象、zone、行動、遷移、移動先、復元、消失、時間の各逸脱を判定
- ハード違反とソフト逸脱を区別
- 観測信頼度を含む不審度計算
- 判定根拠と検知時刻の保存

### 10.4 アプリケーション接続

GUIまたはCLIから、次を一括実行できるようにする。

- 元動画またはカメラの選択
- 作業票、zone定義、検出設定の読み込み
- 骨格推定、YOLO検出、追跡、逸脱判定
- デバッグ動画、CSV、JSON、Markdown結果の出力
- 警告と人手確認画面の表示

### 10.5 評価機能

- 物体検出評価
- トラッキング評価
- イベント生成評価
- 状態遷移評価
- 最終逸脱判定評価
- Time-to-detection評価
- 誤警告・見逃しの原因分析

各処理は中間CSVやJSONを保存できるようにし、誤判定が検出、追跡、イベント生成、文脈照合のどこで生じたかを追跡可能にする。

## 11. 想定ディレクトリ

```text
models/
  yolo/
    README.md
    home_service_objects.pt

configs/
  detection/
    router_yolo.json
  zones/
    router_room_zones.json

data/
  raw_videos/
    IMG_7852.mp4
  trials/
    router_normal_img_7852/
      context.json
      detection_config.json
      zones.json
      skeleton.csv
      object_tracks.csv
      object_crops/
      detection_debug.mp4
      video_metadata.json
```

動画、クロップ画像、モデルファイルは容量やプライバシーの問題があるため、原則Git追跡しない。

設定JSON、CSV、README、評価メモはGit追跡してよい。

## 12. 既存仕様への影響

既存の `analyzer.py` は、基本的にはそのまま使う。

変更が必要になる可能性が高い箇所:

- `ObjectFrame` に `confidence` や `track_id` を追加するかどうか
- `load_object_tracks_csv` が追加列を読むか無視するか
- `crop_path` をGUIで表示するか
- 同じlabelの複数物体をどう表示するか
- `zone` を手動入力ではなく `zones.json` から作るか

ただし、最初の移行では既存の必須列だけを維持し、追加情報はCSVの追加列として保持する。

この方がリワークが少ない。

## 13. 研究上の説明

YOLOを使う理由は、YOLO自体の性能を研究するためではない。

YOLOとトラッカーは、次の観測情報を安定して得るために使う。

- 対象物のID
- 対象物の位置
- 対象物の移動
- 対象物のzone遷移
- 対象物の消失候補
- 対象物周辺クロップ

骨格・姿勢推定は、次を得るために使う。

- 手の接近
- 接触候補
- 持ち上げ候補
- 保持状態
- 人物の移動方向
- 身体側への物体移動

両者を時刻同期し、物体ごとの状態遷移を生成する。

研究の主張は次のように置く。

> 本研究では、骨格・姿勢系列とYOLOによる物体軌跡から、人と物体の相互作用イベントを生成する。生成した物体状態遷移を、作業票から定義された許可状態遷移と照合することで、単発動作では区別できない作業対象外物品への接触、持ち上げ、運搬、消失、未復元を逸脱兆候として検出する。

なお、YOLOトラッキングだけでは毀損を判定できない。毀損を研究対象に含める場合は、局所クロップの操作前後比較または破損状態分類を別途組み込む。

## 14. リスク

### 14.1 小物検出が難しい

鍵、財布、ケーブルなどは小さく、遮蔽されやすい。

対策:

- 対象シナリオを限定する
- カメラ位置を固定する
- 高解像度フレームで検出する
- 必要なクラスだけ追加学習する
- 小物は検出confidenceを低めにして、追跡側で補う

### 14.2 IDが途切れる

手や身体で物体が隠れると、track_idが変わる可能性がある。

対策:

- BoT-SORTとByteTrackを比較する
- bbox中心の近さ、ラベル一致、時間差で後処理マージする
- 消失時間が短い場合は同一物体候補として扱う

### 14.3 誤検出が判定に影響する

誤検出した物体を高リスク物体として扱うと、誤警告につながる。

対策:

- 高リスク判定には一定フレーム以上の連続検出を要求する
- confidenceの平均値を使う
- 物体が手先に近づいたか、移動したかも合わせて判定する

## 15. 実装順序

完成形へ到達するため、依存関係に従って次の順序で実装する。

1. YOLOとトラッカーから`object_tracks.csv`と局所クロップを生成する。
2. 骨格系列と物体トラックを共通時刻へ同期する。
3. 手・物体距離、移動同期、zone遷移、可視状態を計算する。
4. `approach`、`contact`、`lift`、`carry`、`place`、`return`、`disappear`、`exit`を生成する。
5. 物体ごとの状態遷移列を構築する。
6. `context.json`から物体ごとの許可状態遷移グラフを生成する。
7. 観測遷移と許可遷移を照合し、逸脱種別と根拠を出力する。
8. 物体ID再対応付けと観測信頼度による誤警告抑制を実装する。
9. GUI、CLI、リアルタイム警告へ統合する。
10. 検出、追跡、イベント、逸脱判定を個別およびE2Eで評価する。

固定bbox版は性能比較用の基準として保存するが、完成システムの入力方式には使用しない。

## 16. 参考

- Ultralytics YOLO tracking documentation: https://docs.ultralytics.com/modes/track/
- Ultralytics tracking datasets documentation: https://docs.ultralytics.com/datasets/track/
- Ultralytics RT-DETR documentation: https://docs.ultralytics.com/models/rtdetr/
- Grounding DINO official repository: https://github.com/IDEA-Research/GroundingDINO
- Grounding DINO 1.5 API repository: https://github.com/IDEA-Research/Grounding-DINO-1.5-API
