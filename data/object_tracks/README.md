# object_tracks

物体追跡 CSV を置く場所です。

アプリの「物体追跡 CSV」で選択します。

必要な列:

```text
timestamp,object_id,label,role,bbox_x,bbox_y,bbox_w,bbox_h,zone,visible,crop_path
```

`crop_path` には、必要に応じて `data/object_crops/` 配下の画像パスを書きます。

例は `sample_data/object_tracks.csv` を参照してください。

