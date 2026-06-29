# router_normal_img_7852 baseline analysis

このファイルは、現在のアプリ実装で `data/trials/router_normal_img_7852/` を分析した結果のメモです。

## 入力

- 元動画: `data/raw_videos/IMG_7852.mp4`
- 作業文脈: `context.json`
- 骨格CSV: `skeleton.csv`
- 物体追跡CSV: `object_tracks.csv`

## 生成データ

- 動画長: 143.58 秒
- skeleton rows: 128
- object track rows: 144
- object crops: 144 枚

## 判定結果

| object_id | label | role | prediction | score | first_touch_time | first_alert_time |
| --- | --- | --- | --- | ---: | ---: | --- |
| `router_1` | `router` | `target` | `normal` | 0.00 | 25.0 | なし |

## 判定理由

- ルーターは作業対象物として定義されている。
- 物体bboxは作業エリア内に留まっている。
- 物体移動距離は 0.0 で、作業エリア外への持ち出しはない。
- 追跡中に対象物が消失していない。
- そのため、現在のルールでは不審行動には該当しない。

## 注意

この試行データでは、ルーターbboxは固定注釈である。汎用の物体検出器やトラッカーで自動推定したものではない。

MediaPipe Pose の推定には、人物が明確に映る前の誤検出が少し含まれる。ただし、この正常系判定では対象物が移動していないため、最終スコアには影響していない。
