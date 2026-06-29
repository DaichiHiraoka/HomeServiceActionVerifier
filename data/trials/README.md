# trials

実験ごとの入力一式をまとめて置く場所です。

`data/contexts/`、`data/skeleton/`、`data/object_tracks/` に分けて置く代わりに、1つの実験フォルダへまとめたい場合に使います。

推奨構成:

```text
data/trials/<trial_name>/
  context.json
  skeleton.csv
  object_tracks.csv
  object_crops/
  raw_videos/
```

アプリでは、各ファイルを個別に選択します。

## 現在の試行データ

- `router_normal_img_7852/`: `data/raw_videos/IMG_7852.mp4` から作成した正常系ルーター作業データ。

この試行データは `scripts/prepare_router_trial.py` で再生成できます。
