# router_normal_img_7852 analysis notes

この試行データは、正常系のルーター向け作業動画から作成したものです。

## 入力動画

- 動画: `data/raw_videos/IMG_7852.mp4`
- 解像度: 1920 x 1080
- 長さ: 143.58 秒
- FPS: 29.970

## 作成したファイル

- `context.json`: ルーター作業を正常な作業文脈として定義
- `skeleton.csv`: MediaPipe Poseで推定した手首と胴体中心の時系列
- `object_tracks.csv`: ルーター本体を作業対象として固定bboxで追跡した時系列
- `object_crops/router_1/`: ルーター本体と周辺のみを切り出したフレーム画像
- `video_metadata.json`: 動画と生成処理のメタデータ

## 注意

ルーターのbboxは今回の動画に対する固定注釈です。物体検出器が自動でルーターを認識しているわけではありません。
このデータは、現行アプリの正常系入力を作るための初期データとして使います。

## 件数

- skeleton rows: 128
- object track rows: 144

## 実行例

試行データを再生成する。

```powershell
uv run --no-project --python 3.11 --with "mediapipe==0.10.21" --with opencv-python-headless python scripts/prepare_router_trial.py --video data/raw_videos/IMG_7852.mp4 --out data/trials/router_normal_img_7852 --sample-seconds 1.0
```

通常のアプリCLIで分析する。

```powershell
uv run home-service-verifier-app --context data/trials/router_normal_img_7852/context.json --skeleton data/trials/router_normal_img_7852/skeleton.csv --objects data/trials/router_normal_img_7852/object_tracks.csv
```

既存 `.venv` がロックされている場合は、リポジトリの仮想環境を触らずに直接実行する。

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; python -m home_service_action_verifier.app --context data/trials/router_normal_img_7852/context.json --skeleton data/trials/router_normal_img_7852/skeleton.csv --objects data/trials/router_normal_img_7852/object_tracks.csv
```
