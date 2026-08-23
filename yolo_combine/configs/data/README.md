# 資料設定

`yolo_combine` 不擁有 BBAT5 資料版本。所有新 run 先讀取全域 registry：

```text
/home/uxin/yolo/configs/datasets/bbat5-v1.yaml
```

- `bbt5-pose.yaml`、`bbt5-detect.yaml` 是便於閱讀的本專案對照檔；正式 variant 直接引用 registry 鎖定的 canonical YAML。
- `coco2017.yaml` 是 COCO80 Detect 主任務，和 ball/bat BBAT5 Task View 不可混稱。
- runtime View 固定建立在各 variant 的 `artifacts/cache-views/bbat5-v1/`，影像與 labels 都是 symlink，只有 cache 與 runtime manifest 屬於 workspace；正式資料仍只在 `original/pose/`。
- `bbt5_pose_basic` 是 legacy 名稱，新程式與新 YAML 禁止引用。
