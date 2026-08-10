# YOLO11m P2 整理摘要

整理時間：2026-08-10T13:29:40+08:00

## 保留原則

- 保留完整重訓程式、COCO labels/annotations/影像連結與 seed 0。
- 保留 A0、A1 best、A2 Stage 1 best、A2 final best 四個有效權重及 SHA-256。
- 保留正式報告、comparison.png、COCO metrics、benchmark 與正式訓練歷史。

## 移除前容量

| 類別 | Bytes | GiB |
| --- | ---: | ---: |
| linked worktree total | 3,927,925,689 | 3.658 |
| invalidated experiments | 888,917,054 | 0.828 |
| gate runs | 329,509,279 | 0.307 |
| health runs | 351,845,483 | 0.328 |
| download archives | 252,907,541 | 0.236 |
| training/controller logs | 269,444,907 | 0.251 |
| COCO label caches | 247,583,389 | 0.231 |
| validation predictions | 231,293,845 | 0.215 |
| duplicate last checkpoints | 616,825,640 | 0.574 |

各分類可能互相包含，例如 linked worktree total 包含其下所有分類，因此不可直接相加。

## 移除項目

- linked worktree 中未搬入正式封存的所有 runtime 產物。
- invalidated、gate、health、下載 ZIP、大量 logs、predictions JSON、last.pt、初始化與重複權重。
- Python、pytest、Ruff 與 COCO label cache。
- 已被正式 P2_PLAN 取代的主 checkout 舊計畫，以及過期頂層 coco2017.yaml。

## 最終狀態

- 最終 apparent size：1,062,650,032 bytes（0.990 GiB）。
- 最終實際磁碟占用：1,360,134,144 bytes（1.267 GiB，`du -sh` 顯示 1.3G）。
- 已移除 linked worktree：3,927,925,689 bytes（3.658 GiB）。
- 整理前觀測值約 4.0G，整理後為 1.3G，實際磁碟約減少 2.7G。
- Git worktree 僅剩 `/home/uxin/yolo/yolo_p2`。
- 四個 checkpoint 的最終 SHA-256 與 archive manifest 一致，且全部可由 Ultralytics 載入。
- A0 為 strides 8/16/32；A1、A2 Stage 1、A2 為 strides 4/8/16/32。
- focused tests：6 passed。
- Ruff：All checks passed。
- Markdown links、COCO 數量、seed 0、deterministic、正式 epochs 與 benchmark 樣本數均通過檢查。

已刪除的 invalidated、gate、health、logs、ZIP、predictions、last checkpoints、cache、舊 worktree 與臨時 staging 不再能從本專案目錄復原。
