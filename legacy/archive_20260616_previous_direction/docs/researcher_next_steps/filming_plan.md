# 撮影計画(初回トライアル: 50秒 × 2本)

## 方針

比較発表のため、動画を2本に分けて撮影する。

- **Video A `router_trial_normal_001`**: 作業票どおりの正常作業のみ(約50秒)。
- **Video B `router_trial_violation_001`**: 正常作業の流れの中に要件外行動を含む(約50秒)。

Video B には同一動作・異文脈ペア(`photo_context`、`bag_context`)の正常側と違反側を両方入れる。`same_action_different_context_accuracy` は同一アノテーションファイル内でペアを結ぶため、Video B 単体でこの指標を評価できる。

## 撮影者が用意すべき機材

### 必須機材

- 固定撮影できるカメラまたはスマートフォン1台。
  - 横向き撮影。
  - 1280x720 以上。可能なら 1920x1080 で撮影し、評価用に 1280x720 へ変換してもよい。
  - 30fps 程度で十分。手ぶれ補正よりも固定を優先する。
- 三脚、スマートフォンホルダー、または机上固定用のクランプ。
- 充電器、モバイルバッテリー、延長コード。
- 5分以上の空き容量がある記録媒体。
- 秒単位で確認できるタイマーまたはストップウォッチ。
- 撮影開始を合わせるためのスレート代わりの紙。
  - 例: `Video A router_trial_normal_001`、`Video B router_trial_violation_001` と大きく書く。
  - 冒頭で紙を1秒見せるか、手を1回叩いて同期点を作る。
- 床や机に領域を示す養生テープ、マスキングテープ、付箋。
  - `router_shelf`
  - `work_area`
  - `private_desk`
  - `tool_bag_area`
- 印刷または別端末で表示するイベント表。
  - このファイルの Video A / Video B の表を撮影者と作業者が見られる状態にする。

### 小道具

- 作業対象:
  - ルーター本体。
  - ルーター型番ラベル。実機ラベルを使う場合は個人情報や契約情報を隠す。研究用には `DUMMY ROUTER LABEL` などの模擬ラベルを貼る。
  - LANケーブル。
  - 電源アダプタ。
- 作業者の持ち物:
  - 工具バッグ。
  - ドライバー。
  - 作業者用スマートフォン。写真撮影動作を見せるために使う。実際に写真を保存する必要はない。
- 住人側の模擬私物:
  - 模擬書類。本文には `DUMMY DOCUMENT`、`SAMPLE` などだけを書き、実名・住所・電話番号・口座番号を入れない。
  - 模擬の鍵。実鍵を使わない。キーホルダーや番号札にも個人情報を付けない。
  - 模擬財布または小箱。
  - 引き出し役の箱、ケース、または机の引き出し。
- プライバシー対策:
  - 室内の実在書類、郵便物、画面、写真、表札、カレンダーの予定欄を隠す紙や布。
  - 協力者同意の確認メモ。

## カメラ・セット準備

- 固定カメラ1台、1280x720 推奨(`configs/zones/router_repair_zones.json` の前提解像度)。
- 画面内に4領域が入る構図にする(現在の仮座標):
  - `router_shelf`(許可): 画面左、ルーター・LANケーブル・電源アダプタを置く。
  - `work_area`(許可): 画面中央、作業者が立つ位置。
  - `private_desk`(禁止): 画面右、模擬の書類・鍵・財布を置く。
  - `tool_bag_area`: 画面中央下、工具バッグを置く。
- 撮影後、実際の画角に合わせて `zones.json` の bbox を更新する。
- 小道具: ルーター(型番ラベル付き)、LANケーブル、工具バッグ、ドライバー、模擬書類、模擬の鍵、スマートフォン(撮影動作用)。
- 顔を中心に写さない。模擬物体のみ使用し、実在の私的情報を写さない。協力者の同意を得る。

## 撮影前チェック

撮影者は本番前に以下を確認する。

1. カメラを固定し、撮影中に画角が動かないことを確認する。
2. 横向きで撮影し、画面内に `router_shelf`、`work_area`、`private_desk`、`tool_bag_area` が同時に入っていることを確認する。
3. ルーター、工具バッグ、模擬書類、模擬鍵が、それぞれのイベントで手元と物体の動きが見える位置にあることを確認する。
4. 露出とピントをできるだけ固定する。スマートフォンの場合は長押しで AE/AF ロックできる機種ならロックする。
5. 10秒のテスト撮影を行い、再生して以下を確認する。
   - 手元の動作が見える。
   - 模擬書類の内容が研究用のダミーである。
   - 実在の個人情報が画面内にない。
   - 顔が中心に大きく写っていない。
   - 音声が入っても問題ない環境である。不要なら無音でもよい。
6. `configs/zones/router_repair_zones.json` の前提は 1280x720 なので、撮影解像度が違う場合は後で bbox を実画角に合わせて更新する前提で撮る。
7. 作業者に「表の event_id 順に演じる」「失敗したら途中編集せず最初から撮り直す」と共有する。

## 撮影の詳細手順

### 共通手順

1. 撮影者はカメラを固定し、録画ボタンを押す。
2. 冒頭1秒でスレート紙を画面に入れる。
   - Video A: `router_trial_normal_001`
   - Video B: `router_trial_violation_001`
3. スレートを下げ、何も動かさない初期状態を5秒程度撮る。
4. 作業者は下のイベント表どおりに動く。撮影者は原則として声で指示しない。必要な場合は撮影外から小声で「次」とだけ合図する。
5. 各イベントの間に0.5秒から1秒の間を作る。連続して動きすぎると、後で `start_sec` / `end_sec` を切りにくくなる。
6. 物を持つ、置く、撮る、開ける動作は、手と対象物が画面内に入った状態で行う。
7. 違反イベントでも実在の私物は使わない。必ず模擬書類、模擬鍵、模擬財布を使う。
8. 途中でイベント順を間違えた場合、または物体が画面外に出た場合は、その動画を採用せず最初から撮り直す。
9. 録画終了後、すぐに再生して全イベントが見えるか確認する。
10. 採用動画を `video.mp4` として保存し、対応するフォルダへ置く。

### Video A の撮影手順

1. `A00 initial_state`: 作業者を画面外に出し、ルーター、工具バッグ置き場、私物机が見える状態を6秒撮る。
2. `A01 enter`: 作業者が入室し、工具バッグを `tool_bag_area` または `work_area` の見える位置に置く。
3. `A02 inspect`: 作業者が `router_shelf` の前に移動し、ルーターと配線を見る。指差しや軽い確認動作を入れる。
4. `A03 unplug_cable`: LANケーブルを一度抜き、すぐに戻す。ケーブルとルーター端子が見えるようにゆっくり行う。
5. `A04 photograph`: 作業者用スマートフォンをルーター型番ラベルに向け、撮影する動作を行う。画面上は「ルーターを撮っている」と分かればよく、実際の保存は不要。
6. `A05 place_into_container`: 作業者所有のドライバーを手に取り、工具バッグへ戻す。住人側の物に触れない。
7. `A06 exit`: 作業者が工具バッグを持って退出する。退出後に1秒程度、空の画面を残してから録画を止める。

### Video B の撮影手順

1. `B00 initial_state`: Video A と同じ画角で、無人の初期状態を5秒撮る。
2. `B01 enter`: 作業者が入室し、工具バッグを置く。
3. `B02 photograph`: 正常文脈として、ルーター型番ラベルをスマートフォンで撮影する動作を行う。
4. `B03 open`: 作業者が `private_desk` に移動し、模擬の引き出しまたは箱を開ける。ここから要件外行動であることが分かるように、ルーターから明確に離れた場所で行う。
5. `B04 photograph`: 模擬書類にスマートフォンを向け、撮影する動作を行う。書類にはダミー文字だけを使う。
6. `B05 pick_up`: 模擬鍵を手に取る。鍵を持った手が画面内に残るようにする。
7. `B06 place_into_container`: 模擬鍵を作業者の工具バッグに入れる。バッグの口と鍵が見えるようにゆっくり行う。
8. `B07 place_into_container`: 正常文脈として、作業者所有のドライバーを工具バッグに戻す。`B06` と同じ `place_into_container` だが、対象物の所有者が違うことが比較点になる。
9. `B08 exit`: 作業者が退出する。退出後に1秒程度、空の画面を残してから録画を止める。

## 撮影後の確認

撮影者は各動画について以下を記録・確認する。

- 動画ファイル名を最終的に `video.mp4` にする。
- 採用した動画の実測イベント時刻をメモし、下の `annotations.jsonl` 雛形の `start_sec` / `end_sec` を更新する。
- 画角がテスト時から変わっていないか確認する。変わった場合はその動画専用に `zones.json` の bbox を測り直す。
- `Video A` は正常作業だけで構成されているか確認する。
- `Video B` は `photo_context` と `bag_context` の正常側・違反側が両方入っているか確認する。
- 実在の個人情報、顔の大写し、画面内の通知、住所や氏名が写っていないか確認する。
- 採用しないテイクは、混同を避けるため `rejected_take_001.mp4` のように別名へ退避する。

## Video A: 正常作業(約50秒)

すべて Rule-Based スコア 0.0 → `normal` 予測になる想定。

| event_id | 目安時間 | action                                    | zone         | 内容                                                       |
| -------- | -------- | ----------------------------------------- | ------------ | ---------------------------------------------------------- |
| A00      | 0–6s    | `initial_state`                         | work_area    | 無人の初期状態                                             |
| A01      | 6–12s   | `enter`                                 | entrance     | 入室し工具バッグを置く                                     |
| A02      | 12–20s  | `inspect`                               | router_shelf | ルーターと配線を確認                                       |
| A03      | 20–27s  | `unplug_cable`                          | router_shelf | LANケーブルを抜き差しして確認                              |
| A04      | 27–34s  | `photograph` (target: `router_label`) | router_shelf | ルーター型番を撮影 ※pair: photo_context                   |
| A05      | 34–42s  | `place_into_container`                  | work_area    | 作業者所有ドライバーを工具バッグに戻す ※pair: bag_context |
| A06      | 42–50s  | `exit`                                  | entrance     | 退出                                                       |

アノテーション雛形(撮影後に実測時刻へ更新する):

```jsonl
{"event_id":"A00","start_sec":0,"end_sec":6,"label":"normal","action":"initial_state","zone":"work_area","object_class":null,"object_owner":null,"notes":"初期状態"}
{"event_id":"A01","start_sec":6,"end_sec":12,"label":"normal","action":"enter","zone":"entrance","object_class":"tool_bag","object_owner":"worker","notes":"入室し工具バッグを置く"}
{"event_id":"A02","start_sec":12,"end_sec":20,"label":"normal","action":"inspect","zone":"router_shelf","object_class":"router","object_owner":"resident","notes":"作業対象物の確認"}
{"event_id":"A03","start_sec":20,"end_sec":27,"label":"normal","action":"unplug_cable","zone":"router_shelf","object_class":"lan_cable","object_owner":"resident","notes":"LANケーブルの抜き差し確認"}
{"event_id":"A04","start_sec":27,"end_sec":34,"label":"normal","action":"photograph","zone":"router_shelf","object_class":"router_label","object_owner":"resident","target_object":"router_label","same_action_pair_id":"photo_context","notes":"ルーター型番を撮影する正常行動"}
{"event_id":"A05","start_sec":34,"end_sec":42,"label":"normal","action":"place_into_container","zone":"work_area","object_class":"screwdriver","object_owner":"worker","container_class":"tool_bag","container_owner":"worker","same_action_pair_id":"bag_context","notes":"作業者所有の工具をバッグに戻す正常行動"}
{"event_id":"A06","start_sec":42,"end_sec":50,"label":"normal","action":"exit","zone":"entrance","object_class":null,"object_owner":null,"notes":"退出"}
```

## Video B: 要件外行動を含む(約52秒)

正常イベントの間に違反イベントを挟み、ペアの両文脈を1本に収める。

| event_id | 目安時間 | action                                    | zone         | 正解ラベル | 内容                                                   |
| -------- | -------- | ----------------------------------------- | ------------ | ---------- | ------------------------------------------------------ |
| B00      | 0–5s    | `initial_state`                         | work_area    | normal     | 無人の初期状態                                         |
| B01      | 5–10s   | `enter`                                 | entrance     | normal     | 入室し工具バッグを置く                                 |
| B02      | 10–16s  | `photograph` (target: `router_label`) | router_shelf | normal     | ルーター型番を撮影 ※pair: photo_context               |
| B03      | 16–22s  | `open`                                  | private_desk | suspicious | 私物机の引き出しを開ける ※pair: drawer_context        |
| B04      | 22–28s  | `photograph` (target: `document`)     | private_desk | suspicious | 私的書類を撮影 ※pair: photo_context                   |
| B05      | 28–34s  | `pick_up`                               | private_desk | suspicious | 住人所有の鍵を手に取る                                 |
| B06      | 34–40s  | `place_into_container`                  | work_area    | high_risk  | 鍵を作業者バッグに入れる ※pair: bag_context           |
| B07      | 40–46s  | `place_into_container`                  | work_area    | normal     | 作業者所有ドライバーをバッグに戻す ※pair: bag_context |
| B08      | 46–52s  | `exit`                                  | entrance     | normal     | 退出                                                   |

アノテーション雛形:

```jsonl
{"event_id":"B00","start_sec":0,"end_sec":5,"label":"normal","action":"initial_state","zone":"work_area","object_class":null,"object_owner":null,"notes":"初期状態"}
{"event_id":"B01","start_sec":5,"end_sec":10,"label":"normal","action":"enter","zone":"entrance","object_class":"tool_bag","object_owner":"worker","notes":"入室し工具バッグを置く"}
{"event_id":"B02","start_sec":10,"end_sec":16,"label":"normal","action":"photograph","zone":"router_shelf","object_class":"router_label","object_owner":"resident","target_object":"router_label","same_action_pair_id":"photo_context","notes":"ルーター型番を撮影する正常行動"}
{"event_id":"B03","start_sec":16,"end_sec":22,"label":"suspicious","action":"open","zone":"private_desk","object_class":"drawer","object_owner":"resident","same_action_pair_id":"drawer_context","notes":"作業対象外の私物引き出しを開ける"}
{"event_id":"B04","start_sec":22,"end_sec":28,"label":"suspicious","action":"photograph","zone":"private_desk","object_class":"document","object_owner":"resident","target_object":"document","same_action_pair_id":"photo_context","notes":"私的書類を撮影する"}
{"event_id":"B05","start_sec":28,"end_sec":34,"label":"suspicious","action":"pick_up","zone":"private_desk","object_class":"key","object_owner":"resident","notes":"住人所有の鍵を手に取る"}
{"event_id":"B06","start_sec":34,"end_sec":40,"label":"high_risk","action":"place_into_container","zone":"work_area","object_class":"key","object_owner":"resident","container_class":"tool_bag","container_owner":"worker","same_action_pair_id":"bag_context","notes":"住人所有の鍵を作業者バッグに入れる"}
{"event_id":"B07","start_sec":40,"end_sec":46,"label":"normal","action":"place_into_container","zone":"work_area","object_class":"screwdriver","object_owner":"worker","container_class":"tool_bag","container_owner":"worker","same_action_pair_id":"bag_context","notes":"作業者所有の工具をバッグに戻す正常行動"}
{"event_id":"B08","start_sec":46,"end_sec":52,"label":"normal","action":"exit","zone":"entrance","object_class":null,"object_owner":null,"notes":"退出"}
```

## Rule-Based の予測見込み(rule_engine.py で机上検証済み)

- Video A: 全イベント score 0.0 → `normal`。false alarm が出ないことの確認材料になる。
- Video B:
  - B03(引き出し): 0.65 → `suspicious`(正解と一致)。
  - B04(書類撮影)/ B05(鍵を手に取る): スコアが 1.0 まで加算され `high_risk` 予測になる見込み。正解ラベルは `suspicious` なので4クラスでは不一致だが、二値評価(suspicious/high_risk を positive)では正解扱い。発表時は「過剰側に外れる」点として考察に使える。
  - B06(鍵をバッグへ): 1.0 → `high_risk`(正解と一致)。

## フォルダ構成

撮影後、`uploadfiles/template` をコピーして2フォルダを作る。

```text
uploadfiles/
  router_trial_normal_001/
    work_order.json      (configs/scenarios/router_repair.json をコピー)
    zones.json           (実画角に合わせて bbox を更新)
    annotations.jsonl    (Video A 雛形を実測時刻に更新)
    video_path.txt       (video.mp4 と1行書く)
    video.mp4
  router_trial_violation_001/
    work_order.json
    zones.json
    annotations.jsonl    (Video B 雛形を実測時刻に更新)
    video_path.txt
    video.mp4
```

## 評価の流れ

1. 各フォルダごとに `compare-methods` を実行する(または Streamlit UI でフォルダを指定する)。
2. Video A では false_alarm_rate が 0 に近いこと、Video B では recall が高いことを比較表で示す。
3. `same_action_different_context_accuracy` は Video B 側のアノテーションで評価する(photo_context / bag_context のペアが両文脈を含むため)。
4. 必要なら A と B の annotations.jsonl を結合した1ファイルでも評価し、全体指標を出す(event_id が A/B で重複しないようにしてある)。

## 撮影時の注意(再掲)

- 顔や個人識別情報を中心に撮影しない。
- 研究協力者の同意を得た環境で撮影する。
- 実験用の模擬物体を使い、実在の私的情報が写らないようにする。
- raw video を外部APIへ送信しない。
