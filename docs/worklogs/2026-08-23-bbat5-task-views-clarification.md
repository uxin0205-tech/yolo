# 2026-08-23：釐清 BBAT5 Pose／Detect Task View

## 任務與範圍

回答 `bbat5-v1` 是 Pose 或 Detect 資料集的疑問，並同步整理全域、原始資料目錄、
canonical 版本、`yolo_combine` 與 `architecture_2` 的中文入口說明。本輪只調整文件與
領域用語，沒有修改資料、split、labels、YAML、manifest 或權重。

## 變更內容與原因

- 將 `bbat5-v1` 明確定義為「正式資料版本容器」，不是單一任務資料集。
- 明確列出 Pose 使用 `configs/pose.yaml`，ball/bat 二類 Detect 使用
  `configs/detect.yaml`；兩者共用 6,647 張影像及 5,964／683 grouped split。
- 明確區分 COCO80／person Detect：它使用 `/home/uxin/yolo/coco2017.yaml`，不能改用
  `nc=2` 的 BBAT5 Detect YAML。
- 說明融合流程同時記錄 COCO 與 BBAT5 契約，並透過 BBAT5 registry 解析 Pose／二類
  Detect Task View。

## 驗證方式與結果

- 核對全域 registry 同時包含 `tasks.pose` 與 `tasks.detect_2class`。
- 核對 Pose YAML 定義 `kpt_shape=[2,3]`；Detect YAML 只有 ball／bat 兩類。
- 核對 `architecture_2` 正式規格仍定義 Detect 主線為 COCO80、Pose 主線為 BBAT5，
  BBAT5 Detect 只作配對診斷。
- 沒有執行訓練或使用 GPU。

## 困難與解法

「唯一正式資料集」容易被理解成只服務一個 YOLO task。改用「正式資料版本容器」並在每個
入口放置任務選擇表，讓資料版本、Task View 與 COCO80 主線不再混用。

## 未解事項或風險

無。未來若資料或 split 改變仍需建立 `bbat5-v2`，不得覆寫 v1。
