# 撮影計画(初回トライアル: 50秒 × 2本)

## 方針

比較発表のため、動画を2本に分けて撮影する。

- **Video A `router_trial_normal_001`**: 作業票どおりの正常作業のみ(約50秒)。
- **Video B `router_trial_violation_001`**: 正常作業の流れの中に要件外行動を含む(約50秒)。

Video B には同一動作・異文脈ペア(`photo_context`、`bag_context`)の正常側と違反側を両方入れる。`same_action_different_context_accuracy` は同一アノテーションファイル内でペアを結ぶため、Video B 単体でこの指標を評価できる。

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

## Video A: 正常作業(約50秒)

すべて Rule-Based スコア 0.0 → `normal` 予測になる想定。

| event_id | 目安時間 | action | zone | 内容 |
| --- | --- | --- | --- | --- |
| A00 | 0–6s | `initial_state` | work_area | 無人の初期状態 |
| A01 | 6–12s | `enter` | entrance | 入室し工具バッグを置く |
| A02 | 12–20s | `inspect` | router_shelf | ルーターと配線を確認 |
| A03 | 20–27s | `unplug_cable` | router_shelf | LANケーブルを抜き差しして確認 |
| A04 | 27–34s | `photograph` (target: `router_label`) | router_shelf | ルーター型番を撮影 ※pair: photo_context |
| A05 | 34–42s | `place_into_container` | work_area | 作業者所有ドライバーを工具バッグに戻す ※pair: bag_context |
| A06 | 42–50s | `exit` | entrance | 退出 |

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

| event_id | 目安時間 | action | zone | 正解ラベル | 内容 |
| --- | --- | --- | --- | --- | --- |
| B00 | 0–5s | `initial_state` | work_area | normal | 無人の初期状態 |
| B01 | 5–10s | `enter` | entrance | normal | 入室し工具バッグを置く |
| B02 | 10–16s | `photograph` (target: `router_label`) | router_shelf | normal | ルーター型番を撮影 ※pair: photo_context |
| B03 | 16–22s | `open` | private_desk | suspicious | 私物机の引き出しを開ける ※pair: drawer_context |
| B04 | 22–28s | `photograph` (target: `document`) | private_desk | suspicious | 私的書類を撮影 ※pair: photo_context |
| B05 | 28–34s | `pick_up` | private_desk | suspicious | 住人所有の鍵を手に取る |
| B06 | 34–40s | `place_into_container` | work_area | high_risk | 鍵を作業者バッグに入れる ※pair: bag_context |
| B07 | 40–46s | `place_into_container` | work_area | normal | 作業者所有ドライバーをバッグに戻す ※pair: bag_context |
| B08 | 46–52s | `exit` | entrance | normal | 退出 |

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
