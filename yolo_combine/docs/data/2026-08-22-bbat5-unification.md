# BBAT5 資料統一紀錄

## 決定

`yolo_combine` 的所有新 Pose、ball/bat Detect 診斷及融合 run 統一使用
`/home/uxin/yolo/configs/datasets/bbat5-v1.yaml` 所鎖定的 `bbat5-v1`。原始
`dataset/`、`detect_dataset/` 與 `bbt5_pose_basic` 只保留歷史血緣語意。

## 為什麼不用 basic split

| 項目 | Legacy `bbt5_pose_basic` | Canonical `bbat5-v1` |
| --- | ---: | ---: |
| Train images | 6,080 | 5,964 |
| Val images | 567 | 683 |
| 總影像 | 6,647 | 6,647 |
| `.rf.` 同源群組跨 split | 10 | 0 |
| 負 keypoint 座標 | 已在 View 中臨時修 4 筆 | 正式 lineage 固定修 4 筆 |
| split/patch manifest | 不完整 | 完整且有 SHA256 |

逐檔核對舊 View 與 `bbat5-v1`：6,647 張影像與 6,647 個 Pose labels 全部可對應，
missing image/label 都是 0，label content mismatch 是 0。兩者差異是 split assignment
與正式 lineage，不是樣本遺失。

## 舊 View 證據

- 舊路徑：`/home/uxin/yolo/yolo_combine/artifacts/datasets/bbt5_pose_basic`
- manifest SHA256：`71db159e9943a183ae47c6bf241da7dbe5ee71d6b1c6535834bc77f1f4b4adfd`
- data YAML SHA256：`9b762b3778e8607c4bebbb20541a2559a18b5b20a0f46e668ef572457b63c466`
- 內容：6,647 個 image symlink、6,647 個獨立 label、2 個可重建 Ultralytics
  `labels.cache`、2 個 metadata 檔，0 broken symlink；沒有權重或 checkpoint。
- 舊 split 的 10 個 overlap groups：`Untitled_mov-13_jpg`、`Untitled_mov-17_jpg`、
  `Untitled_mov-9_jpg`、`frame1363_jpg`、`frame1461_jpg`、`frame1473_jpg`、
  `youtube-187_jpg`、`youtube-247_jpg`、`youtube-400_jpg`、`youtube-407_jpg`。

舊 View 完全可由唯讀原始來源與既有程式重建，且所有 labels 已由 canonical 版本涵蓋；
因此本輪移除其生成檔，避免新訓練誤用。歷史 checkpoint、metrics、baseline JSON 與上述
hash 證據保留，不回溯改寫。

## 新 runtime View

新程式只接受全域 registry，先驗證 task YAML 與五份 lineage manifests 的 SHA256，再建立
`artifacts/cache-views/bbat5-v1`。影像與 labels 都是 symlink；Ultralytics cache 留在 workspace，正式
資料保持唯讀。偵測到 registry hash 漂移、不同資料路徑、跨 split group 或負座標時會
fail closed。

## 最終 CPU 驗證

- Detect 與 Pose 各自從 registry 重建 6,647 images／6,647 labels 的 runtime View；
  兩個 task 合計 26,588 symlinks、0 broken links，實體檔只有 4 個 YAML／manifest。
- paired audit：Detect label derivation mismatch 0、train/val group overlap 0、負座標 0。
- Full35 與 Partial75 audit 都引用 `bbat5-v1`，沒有 missing path 或 hash mismatch。
- 完整測試：38 passed、3 CUDA-only skipped；本輪未使用 GPU。
