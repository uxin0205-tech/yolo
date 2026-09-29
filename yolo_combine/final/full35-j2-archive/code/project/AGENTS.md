## Agent skills

### Issue tracker

Issues and specs are tracked in GitHub Issues for `uxin0205-tech/yolo`. See `docs/agents/issue-tracker.md`.

### Domain docs

This repository uses a single-context domain documentation layout. See `docs/agents/domain.md`.

## BBAT5 正式資料

- 所有新 Pose、ball/bat Detect 診斷與融合實驗只使用 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml` 宣告的 `bbat5-v1`。
- `/home/uxin/yolo/original/pose/dataset`、`detect_dataset` 與 `artifacts/datasets/bbt5_pose_basic` 只屬歷史血緣，禁止作新 run 輸入。
- `artifacts/datasets/bbat5-v1-runtime` 是可重建、symlink-only 的 cache 隔離 View，不得擁有自己的 split 或 label 修補。
- 歷史 checkpoint 與報告保留當時 basic split 的資料血緣，不回溯改寫其 metrics。
