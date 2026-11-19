# 2026-11-08：Stage8 進度查核與 SiLU 訓練中斷恢復

## 任務與範圍

使用者查詢目前狀況。第 8 階段五組候選仍屬已授權範圍；異常處理沿用既定配方，完成本階段後停止。`yolo_actshape/` 僅有下一階段的工作區說明，未啟動實驗。

## 已完成結果

Rep17、Rep20、Rep17＋20 各完成 5 個 epoch，最佳部署結果皆為 E1；非零分支 folding 與重新載入後完整 AP 驗證通過。三組未達既定延長條件，沒有追加訓練。

| 候選 | COCO AP | person AP | BBAT Pose AP | 相對 Stage7 的主要未過門檻項目 |
|---|---:|---:|---:|---|
| Rep17 | 0.514681 | 0.643073 | 0.900236 | ball Pose 降低 0.2265 個百分點 |
| Rep20 | 0.514086 | 0.643344 | 0.899160 | ball Pose 降低 0.4460 個百分點，另有 COCO bat／整體 Pose 退步 |
| Rep17＋20 | 0.514347 | 0.642796 | 0.901389 | bat box 降低 0.1401 個百分點 |

以上皆為 qSiLU／D1 部署版本 AP50–95。逐項保護門檻為相對 Stage7 降低不超過 0.1 個百分點；尚未採用新候選，舊最佳交接模型保留。

## 中斷證據與診斷

SiLU 候選在 2026-11-07 22:46:37 收到 SIGSEGV，worker PID 3718471，queue 隨後記錄 `exit=-11` 並停止。最後進度為第 1／5 個 epoch、第 200／463 個 macro，尚未完成任何 epoch，也沒有 `resume.pt`。ACB 尚未開始。

主代理依 diagnosing-bugs 流程先確認實際故障證據，讀取系統 apport crash report，僅將 core 暫時解壓至 `/tmp`，不複製環境變數或完整記憶體內容到專案。GDB 顯示主執行緒停在 `libcuda.so.1`，呼叫路徑包含 `cuStreamSynchronize`、`cudaStreamSynchronize`、`at::native::copy_kernel_cuda`。這確認故障位置，尚不能證明是驅動、硬體或上游呼叫的根因。

未建立可在數秒內穩定重現的最小案例，因此不宣稱完成根因修復或以簡單 smoke 取代原始故障驗證。採原配方的一次恢復執行並增加 Python fault handler，觀察實際訓練路徑。原始 core 由系統保管；專案只保存小型堆疊與事件資料。

查核時 GPU 無 compute process、顯存約 619 MiB，系統可用 RAM 約 114 GiB、磁碟剩餘 369 GiB。未取得可讀的系統 kernel log，不能據此排除過去的 GPU／系統異常。

## 恢復前參考與決策

再次唯讀核對 `yolo_optimize/experiments/pose_only_rep_v1/config.json`、`run_arm.py`，及既有 `stage8/reference-review.json` 的最新 `retraining_review`。33 項參考 SHA、410 項來源鎖及共同 parent 均通過核對。optimize 的正式實作使用 qSiLU＋Float PWL；本輪只將兩個 Rep 相加後的 activation 改為 SiLU，屬已記錄的獨立延伸候選，其他算子與既定訓練配方維持不變。

共同 Stage7 parent SHA：`2f91ec1862ad62304f1e2727e9b9ee64a27c5a46a45137f83a74dc7f14c46a69`。SiLU 初始權重 SHA：`15e5feebe42d87c5e8424d7073d803482e4c354c72218db7f7033cc39f76c45e`。

恢復保留原 5 個 epoch、Rep LR 1e-5／heads 5e-6、warmup／cosine、固定 BN running、EMA、Pose weight 0.24、batch、資料、evaluator 與條件式延長／硬體適應規則。沒有 epoch 續訓檔，故從相同初始權重重新執行第 1 個 epoch；不冒稱恢復第 200 步狀態。已完成的三組及 SiLU 校準／smoke 直接重用。

只在程序環境新增 `PYTHONFAULTHANDLER=1`，不改程式、來源鎖、protocol 或正在使用的 job 配置。GPU 預估約 8 GiB；queue 每 600 秒記錄監測、每 60 秒檢查其他 GPU 工作並避讓。

事件證據：`yolo_achitechure/artifacts/stage8/repconv-variants-20261107/recovery-20261108/incident.json` 與 `cuda-stack.txt`。啟動與恢復後驗證續記如下。

## 恢復執行與驗證結果

20 次小型 CUDA 矩陣運算、GPU→CPU 複製與同步檢查通過；只確認當下 CUDA 可用，不代表已重現或修復原故障。環境為 PyTorch 2.11.0+cu128、CUDA build 12.8、NVIDIA driver 595.71.05。

2026-11-08 01:04:05 以原入口重新啟動背景 queue，supervisor PID 3724621、SiLU worker PID 3724626。01:05:17 確認進度已前進至第 1／5 個 epoch、第 10／463 個 macro，EMA updates 10、當次 AMP retries 0，shared／Detect head／Pose head 梯度皆存在。GPU 使用約 96%、7721 MiB，僅有本輪 compute process。

恢復後再次確認 410 項來源鎖，以及 source／config／job SHA 完全不變；三組已完成的 result／verify 與部署 checkpoint SHA 亦一致。fault handler 已實際繼承至 worker。完整檢查保存於 `recovery-check.json`。尚未重新到達原中斷步數，不宣稱根因已排除。

已直接刪除本次建立的暫存 core 與 GDB 原始輸出，共 5,561,259,074 bytes（約 5.18 GiB）；保留小型堆疊與事件證據，沒有 archive／backup，未改動系統 apport 原檔。

依已完成候選每 epoch 約 30 分鐘、剩餘 SiLU／ACB 共 10 個 epoch，加上前置、完整 AP 與交接驗證，無新增異常或條件式延長時估計 11 月 8 日 06:30–08:00 完成。SiLU 硬體適應或低 LR 延長另計。依使用者偏好，在確認背景訓練持續前進後退出 working；原 600 秒監測機制持續執行，第 9 階段仍未啟動。

## 困難與解法

worker 沒有 Python traceback；使用系統 core 定位 CUDA 同步／複製呼叫，增加 fault handler 供再次異常時記錄。臨時證據整理腳本曾混用絕對及相對路徑，改為先解析絕對路徑後完成；未改動任何模型或實驗資料。

## 01:07：確認 SiLU 與 ACB 排程

依使用者要求重新查核：SiLU worker 3724626 與 supervisor 3724621 均存活，重啟後紀錄由 macro 1、10、20、30 持續前進至 40／463（第 1／5 個 epoch）。已記錄 loss 全為有限值，兩個 heads 與 shared 梯度存在，最新 AMP retries 0，worker log 沒有新增 CUDA error／SIGSEGV／Python traceback。GPU 約 95%、7721 MiB、59°C。

ACB 已通過既有 CPU 初始化、非零分支 folding、重載與算子契約驗證，仍在 SiLU 之後排隊，尚未啟動 GPU 前置或正式訓練。兩組依序使用 GPU，不能將排程正常誤寫為兩組皆已完成 GPU 驗證。SiLU 此時尚未再次到達原中斷步數。410 項來源鎖仍通過，沒有修改配方、重啟程序或插入新實驗；無新增執行困難。

查核資料：`recovery-20261108/status-confirmation-0107.json`。原 600 秒監測與 Stage8 完成後停止規則維持。

## 未解事項與風險

SIGSEGV 根因尚未確認，恢復執行不能當作根因已修復。SiLU 與 ACB 尚無訓練後完整 AP；若觸發既定延長或硬體算子適應，完成時間將順延。沒有 RTL／HLS 或實板量測。
