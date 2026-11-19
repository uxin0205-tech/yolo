# Stage9 CUDA 原生崩潰查核與一次恢復

日期：2026-11-09，16:55 使用者查詢進度。當時 GPU 已無 compute 程序，初輪與擴充 supervisor 均停止。384×768 的 20 個完整 epoch、正式 AP 與重載驗證保持；512×1024 尚無完整 E1 結果，擴充三尺寸尚未啟動。

## 事件與已知事實

12:32 系統重啟後的恢復確實執行到 E1 step240/463；apport 記錄 12:53:20 SIGSEGV，初輪 supervisor 於 12:54:24 記錄 worker exit=-11。因沒有完整 epoch resume，原政策不自動重試；擴充隨後停止等待。上午只確認新 macro 持續前進，並未保證長時間故障已排除。

Python fault handler 表面停在 `CompletePWLSoftmax.weight_codes` 的 `torch.isfinite(scores).all()`。暫時解壓本使用者的 apport CoreDump，GDB 原生堆疊為 `libcuda.so.1 → cuStreamSynchronize → cudaStreamSynchronize → _local_scalar_dense_cuda → tensor bool`。這是故障浮現的同步呼叫路徑，不足以認定 PWL 數值公式、driver 本身、loader 或硬體是根因。Stage8 過去在不同算子的 CPU copy 同步處也曾留下相近 libcuda 堆疊。

CUDA 非同步錯誤可能到同步點才浮現；`CUDA_LAUNCH_BLOCKING=1` 可用於定位，會改變執行時間，不應當作正式可靠性修復。[NVIDIA 官方說明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)、[PyTorch CUDA 說明](https://docs.pytorch.org/docs/stable/notes/cuda.html)。

環境：PyTorch 2.11.0+cu128、CUDA build 12.8、cuDNN 91900、NVIDIA driver 595.71.05。查核時 RAM 可用約 121 GiB，磁碟剩餘約 375 GiB；不能以此排除故障當時的系統問題。上次開機 kernel journal 無可讀條目。當天 12:52 有系統套件設定紀錄，但 libcuda 檔案未在該時段變動；沒有足夠證據將系統更新判定為本次根因。

## 動手前參考與診斷設計

先唯讀核對 optimize `pose_only_rep_v1/run_arm.py` 的完整 epoch snapshot／恢復邊界、config，以及其 BitTrue PWL normalization 實作；已有 reference-review 的 parent、實際配方與結果繼續適用。optimize 的舊 PWL 保留浮點 normalization，D1 完整整數 reciprocal 是本專案既有延伸，不能把這次 CUDA 問題當成改回浮點除法的理由。

原來源 436 項、擴充 450 項再次驗證通過，14 個已完成 job、兩份完成 resume、parent、best、job/config 與既有來源 SHA 未改。原 qSiLU／D1／B2／S1／folded Rep17＋20 契約、資料、LR、batch、BN／EMA 與上限 100E 均保留。

新增來源鎖外的 `yolo_actshape/operations/diagnose_cuda.py`，僅隔離測試 D1，沒有改正式算子或儲存模型權重。先做 CPU／GPU probability 逐位比對及 NaN／超出 token 上限的拒絕檢查，再以實際 batch8、4 heads、512×512 scores 重複前向、反向、GPU→CPU scalar 同步；常規及同步除錯環境分開執行。假設排序與預算事前記於 `artifacts/segv-20261109/incident.json`。

## 驗證結果

100 次初測通過，耗時僅約 0.41 秒，故事前補記延伸壓力檢查：常規與同步模式各 10,000 次。常規耗時 37.04 秒，同步模式 52.60 秒；兩者 CPU／GPU probability 均逐位相同，梯度有限且非零、非法輸入均拒絕、最大 row-sum 誤差約 0.0019517，峰值 allocated 847,269,888 bytes。沒有用最後浮點除法重新正規化，也沒有改 D1 位寬或常數。

這 20,000 次診斷未重現完整訓練的間歇性崩潰，不能視為根因修復或任務模型長訓穩定性保證。它也沒有重現完整資料載入、雙任務與所有卷積的互動。這是目前診斷的實際限制，沒有為了取得簡單綠燈而把 smoke 當成原故障的 regression test。

## 恢復決策與執行

在 GPU 短診斷、數值檢查、來源與既有成果 SHA 全部通過後，記錄本事件僅一次的人工恢復額度，再以原 `run_stage9.sh run` 入口啟動。原第一輪沒有 checkpoint，因此仍從相同 parent 的 E1 開頭重算；沒有額外增加完整 epoch 預算。

正式訓練不啟用 CUDA_LAUNCH_BLOCKING，不修改 driver、系統服務、loader、precision 或正式訓練程式。只清掉該中斷 job 的過期即時 progress.json；原值保存於 incident JSON，progress.jsonl 與正式歷史不動，避免再觸發「兩小時無進度」誤判。

17:06:39 啟動初輪 supervisor PID 19974。先確認 GPU worker 寫入新 macro 心跳，再恢復擴充等待程序。原政策仍是有完整且 SHA／config／source 相符的 epoch resume 才最多自動重試一次；沒有 resume 再失敗則停止，不無限重啟。

## 清理與剩餘事項

本次 `/tmp` 的 5,467,738,112-byte CoreDump 及 GDB 原始輸出已刪除，只保留小型 `cuda-stack.txt`、診斷程式與 JSON 證據；沒有 archive、沒有更動系統 apport 原檔，也沒有刪除唯一權重。

困難：CUDA 原生故障尚無快速穩定重現案例，因此未確認根因。原 ETA 已失效；若這次能持續執行，剩餘約 34 小時，粗估 11 月 11 日，需待 512×1024 首輪完整保存及後續實測更新。每 600 秒監測仍維持，下一階段不啟動。恢復後進度與驗收續記如下。

## 恢復後確認

2026-11-09T17:09:49.273882+08:00：初輪 supervisor 19974、GPU worker 19979 正常；E1 step30/463、EMA updates 30，AMP retries 0，共享／雙 head 梯度存在。擴充 supervisor 20067 已恢復 waiting_initial_queue。兩個最新 monitor 均在 600 秒內。

目前只證明這次實際訓練已恢復前進，尚未通過之前約 240 步的故障區間，也未保存完整 E1，因此沒有新增任務 AP，不宣稱長訓穩定或根因修復。按既有政策留在背景監測；若無再中斷，暫估 11 月 11 日，待首輪完整保存後再更新。

## 17:32 持續執行查核

使用者詢問是否繼續。確認原 supervisor 19974、worker 19979 與擴充 supervisor 20067 均存活，沒有重新啟動。512×1024/adapt10 已由前次 E1 step30 前進至 step290/463、EMA updates290，超過先前約 step240 的故障區間；最新進度約39秒前更新，AMP retries0、共享與雙 head 梯度存在。GPU使用率95%、62°C、總記憶體6998 MiB，compute只有原worker。兩個monitor距查核約322秒／205秒，擴充仍waiting_initial_queue。

本次只查核並記錄，未改配方、來源或預算。新困難：無。首輪尚未完成，尚無完整E1 checkpoint或新增AP；通過先前步數不代表間歇性CUDA根因已修復，背景繼續每600秒監測。

## 20:17 完整 checkpoint 與 AP 進度

2026-11-09T20:18:36.963022+08:00：同一 worker 19979 與 supervisor 19974 已連續運行超過 3 小時，沒有新增重啟。512×1024 完成 E1–E4 的完整 COCO5000／BBAT683 AP，正在第一段 E5/10、step340/463；之後仍有半 LR 10E。GPU查核時93%、59°C、總記憶體6968 MiB，沒有其他compute worker。擴充supervisor20067正常等待，600秒監測持續。

已實際驗證 E4 resume.pt 的 SHA、config SHA、source SHA，next_epoch=5；best-e004.pt 與 sidecar SHA 也通過。若再次中斷，已有完整E4狀態可以恢復，不再需要從E1重算。原程式／配方／資料未改，原生CUDA故障根因仍未確認。

目前綜合最佳是同一份 E4，十項 AP 如下（百分點差異相對原Stage8 parent）。

| 指標 | 原 parent | E4 | 差異 pp |
|---|---:|---:|---:|
| coco_AP | 51.4347% | 51.8553% | +0.4206 |
| person_AP | 64.2796% | 64.6954% | +0.4158 |
| sports_ball_AP | 51.5782% | 52.0212% | +0.4430 |
| baseball_bat_AP | 46.9576% | 46.7692% | -0.1885 |
| box_AP50_95 | 61.4371% | 61.6512% | +0.2142 |
| pose_AP50_95 | 90.1389% | 89.9204% | -0.2185 |
| ball_box_AP50_95 | 49.9764% | 50.4882% | +0.5118 |
| ball_pose_AP50_95 | 86.4546% | 86.9347% | +0.4801 |
| bat_box_AP50_95 | 72.8977% | 72.8142% | -0.0835 |
| bat_pose_AP50_95 | 93.8233% | 92.9062% | -0.9171 |

COCO、person、球與BBAT box有提升，但COCO球棒、整體Pose及球棒Pose仍未通過逐項保護門檻，暫不更新交接主線。該形狀Conv＋PV為60.7111 GMAC，比固定640×640高28.06%；成本不能當成相對原rect640動態輸入的硬體加速。

已完成整輪平均40.84分鐘；若無新中斷，512×1024原20E與驗證粗估2026-11-10T06:48:47.033038+08:00完成，整體含擴充仍約11月11日。小型數據見 `yolo_actshape/artifacts/status-20261109-2017.json`。新困難：無。繼續既有背景queue，無新增訓練或測試。

## 22:55 持續訓練與最佳保留

2026-11-09T22:55:09.387298+08:00：同一 supervisor19974／worker19979 連續執行約5小時48分鐘，沒有新增SIGSEGV重試或資源衝突。512×1024/adapt10已完成8輪，正在E9/10 step270/463；EMA updates3974，AMP retries0、共享／雙head梯度存在。GPU查核95%、62°C，總顯存7160MiB；擴充supervisor20067仍等待初輪完成，兩個monitor均在600秒內。

依既定rank，目前綜合最佳為E4；best checkpoint與sidecar SHA核對通過。最新完整resume已驗SHA／config／source，next_epoch=9，代表可從下一輪恢復。

| 指標 | 原 parent | 目前最佳 | 差異 pp |
|---|---:|---:|---:|
| coco_AP | 51.4347% | 51.8553% | +0.4206 |
| person_AP | 64.2796% | 64.6954% | +0.4158 |
| sports_ball_AP | 51.5782% | 52.0212% | +0.4430 |
| baseball_bat_AP | 46.9576% | 46.7692% | -0.1885 |
| box_AP50_95 | 61.4371% | 61.6512% | +0.2142 |
| pose_AP50_95 | 90.1389% | 89.9204% | -0.2185 |
| ball_box_AP50_95 | 49.9764% | 50.4882% | +0.5118 |
| ball_pose_AP50_95 | 86.4546% | 86.9347% | +0.4801 |
| bat_box_AP50_95 | 72.8977% | 72.8142% | -0.0835 |
| bat_pose_AP50_95 | 93.8233% | 92.9062% | -0.9171 |

目前均輪40.72分鐘，第一段估計2026-11-09T23:54:08.091641+08:00結束，再由既有queue從最佳權重校準後接半LR10E。512×1024兩段及驗證粗估2026-11-10T06:46:22.536976+08:00完成，整體仍約11月11日。小型資料見 `yolo_actshape/artifacts/status-20261109-2255.json`。

本次只讀取既有結果、驗證新checkpoint並更新紀錄；沒有改研究方法、配方或執行中來源。新困難：無。CUDA原生故障根因仍未確認，不因連續執行就宣稱已修復。
