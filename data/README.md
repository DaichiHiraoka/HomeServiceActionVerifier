# Data Directory

このディレクトリは、現在のデスクトップアプリで使う入力ファイルを置く場所です。

## 基本配置

| 場所 | 置くもの | アプリで選ぶ入力 |
| --- | --- | --- |
| `data/contexts/` | 作業文脈 JSON | 作業文脈 JSON |
| `data/skeleton/` | 骨格 CSV | 骨格 CSV |
| `data/object_tracks/` | 物体追跡 CSV | 物体追跡 CSV |
| `data/object_crops/` | 対象物体周辺クロップ画像 | 物体追跡 CSV の `crop_path` から参照 |
| `data/raw_videos/` | 元動画 | 現時点のアプリでは直接入力しない |
| `data/trials/` | 実験ごとの一式 | 必要に応じて各ファイルを選択 |

## 注意

- 顔や部屋全体が映る元動画は `data/raw_videos/` に置く。
- 対象物体周辺クロップは `data/object_crops/` に置く。
- 元動画とクロップ画像はプライバシー情報を含む可能性があるため、原則 Git 追跡しない。
- Git には、CSV、JSON、README、空ディレクトリ維持用の `.gitkeep` だけを置く。

## 迷った場合

最初は `data/trials/template/` をコピーして、実験名に変更して使う。

例:

```text
data/trials/router_pickup_001/
  context.json
  skeleton.csv
  object_tracks.csv
  object_crops/
  raw_videos/
```

