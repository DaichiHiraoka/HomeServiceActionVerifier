# Upload Folder Template

Copy this directory under `uploadfiles/` and rename the copy for each real trial.

Required files:

- `work_order.json`
- `zones.json`
- `annotations.jsonl`
- `video_path.txt`

`video_path.txt` may be empty/comment-only when the experiment only needs event-token evaluation. If a video should be recorded with the run, put a relative path such as `video.mp4` on the first non-comment line and place that video in the same trial folder.
