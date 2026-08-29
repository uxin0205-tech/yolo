# 2026-08-25：GPU 使用狀態唯讀檢查

## 任務與範圍

依使用者要求，在不干擾既有 GPU 任務的前提下確認 GPU 是否有人使用。本輪只執行一次
`nvidia-smi` 快照及既有 PID 的 `ps`／`pstree` 查詢；沒有建立 CUDA context、啟動 benchmark、
送出 signal、調整 clock／power、終止程序或啟動本專案訓練。

## 檢查時間與結果

- 時間：`2026-08-25T20:39:22+08:00`。
- GPU 0：NVIDIA GeForce RTX 5090，57°C、85% GPU utilization、302.20 W。
- 顯存：21,633 / 32,607 MiB 已使用。
- 主要 compute PID：`3170826`，GPU 顯存約 21,280 MiB；單次 `pmon` 為 SM 67%、memory 42%。
- 擁有者：`vicchen`。
- 工作：`MambaPose/tools/train.py`，設定
  `crowdpose-s-v1-no-cycling.py`，work directory 為
  `MambaPose/work_dirs/reproduction/runs/crowdpose-s-v1-no-cycling`。
- 啟動時間：2026-08-25 18:26:52；檢查時已運行約 2 小時 12 分。
- cgroup：`mambapose-reproduction.service`；程序狀態為 running，並有 PyTorch data workers。
- Xorg／GNOME 等 graphics processes 也存在，但主要 compute workload 是上述 MambaPose training。

結論：GPU 正在被其他使用者的訓練工作大量使用，本專案目前不應啟動 GPU training、benchmark
或大量 CUDA 工作。


## 23:52 後續唯讀快照

- 時間：`2026-08-25T23:52:16+08:00`。
- GPU 0：62°C、100% utilization、365.36 W，顯存仍為 21,633 / 32,607 MiB。
- compute PID 仍為 `3170826`，使用約 21,280 MiB；`ps` 顯示擁有者仍為 `vicchen`、程序狀態
  `Rsl`，同一 MambaPose CrowdPose training 已累計運行約 5 小時 25 分。
- 本次同樣只使用 `nvidia-smi` 快照與 CPU `ps`；沒有輪詢、CUDA 工作或程序操作。
- 結論不變：GPU 正在滿載，本專案不應啟動 GPU 任務。

## 驗證方式

1. `nvidia-smi --query-gpu=...`：取得 utilization、memory、temperature 與 power 快照。
2. `nvidia-smi --query-compute-apps=...` 與一次 `nvidia-smi pmon -c 1`：定位 compute PID。
3. `ps`／`pstree`：唯讀確認使用者、命令、啟動時間、cgroup 與 worker tree。

## 困難與解法

- 初次命令因 sandbox 的 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` 失敗；改用
  經核准的沙箱外唯讀查詢。沒有以任何方式操作 GPU 程序。
- 補記第二次快照時，第一個精確 pattern 未命中，後續又因 Perl `#` delimiter 與 Markdown 標題
  衝突而出現 parser error；兩次都沒有改動檔案。改用不衝突的 delimiter 後成功，並立即唯讀核對。

## 未解事項或風險

- 此結果只是單一時間點快照，GPU 狀態可能隨時改變；啟動任何後續 GPU 工作前必須重新確認。
- 沒有估算既有工作剩餘時間，也沒有讀取或修改其他使用者的 training logs。
