# 2026-11-08：Stage8 完成、無 Rep 比較推薦與中斷恢復

最終更新：Stage8 於 14:33 完成，選定 rep17_20_silu E1，部署使用 qSiLU／D1；GPU 已釋放，Stage9 未開始。下方分時段記錄先前進度，最後的比較與查核為現行交付結論。

## 任務與範圍

使用者要求顯示全部結果及優劣。整理 Stage7 起點、Stage8 舊輪三組、新輪五組的已完成結果，分開完整驗證、逐輪暫定 AP 與尚未開始的候選；不提前宣告 Stage8 完成，不啟動 Stage9。

## 動手前參考與查核

再次唯讀查閱 `yolo_optimize/experiments/pose_only_rep_v1/RESULTS.md`、`run_arm.py` 與本輪固定 protocol／reference-review，410 項來源鎖和 33 項參考 SHA 均通過。optimize 保存完整 epoch snapshot，從 next epoch 恢復；本輪沿用此恢復邊界。optimize 沒有已驗證的 SIGSEGV 自動恢復配方；本次外層程序監測屬執行管理延伸，不改模型、數值運算、訓練預算或實驗來源鎖。

## 查核時的結果與異常

09:24 查得新輪三組 qSiLU Rep 均完成 5E；SiLU 完成 E1–E4，第 5 輪最後紀錄為 340／463 macro，於 03:26:14 再次 SIGSEGV，queue 停止，ACB 尚未啟動。此前預估 06:30–08:00 完成已不適用。兩次停止都由既有 supervisor 記錄，但沒有自動恢復，因此直到本次查詢才繼續處理。

本次 Python fault handler 顯示當前呼叫在 `ultralytics/utils/loss.py` 建立 CUDA imgsz tensor；系統 core 的 GDB 堆疊仍指向 `libcuda.so.1`／`cuStreamSynchronize`，與前次位置一致。這只定位故障，不能證明 SiLU、驅動或硬體的根因。

完整 E4 resume SHA：`8ea9b45b2ae988fea5d10509f8996a863b73cb590127aa753121ffc338ee1675`。CPU 載入核對 config、來源鎖、4 輪 history、EMA updates 1852、6 組 optimizer 及 next_epoch=5 一致。E1 最佳多分支權重 SHA：`5d84d66891f9918c972986a617ba5244cdf36b1374b6c87d541e6af1f533012e`。

## 事前固定的恢復方式

從 E4 的完整狀態重做 E5，保留原資料順序、LR／scheduler、optimizer、EMA、算子與模型。GPU 預估約 8 GiB；剩餘 E5 約 30 分鐘，ACB 原定 5E 約 2.5 小時，SiLU 可能依既定條件再接 3E 硬體適應，實際時間另計。這次不從頭重跑已完成的 4 輪。

新增外層恢復程序，放在來源鎖以外的 `yolo_achitechure/operations/`，獨立記錄自身 SHA。首次恢復後至多再自動重試一次，而且必須同時符合：本輪 train worker 的 SIGSEGV、無 STOP、原來源鎖未變、config／job 相符、完整 epoch resume SHA 與 next_epoch 合格。數值錯誤、驗證失敗、其他 exit code、沒有續訓檔或超過上限都停止；既有 GPU 避讓與 600 秒監測維持。此管理層不代表根因修復。

## 結果整理與驗證

進行中：建立同一 checkpoint 的十項 AP 比較、訓練與部署差距、逐輪曲線、參數成本與可離線重畫的小型資料。SiLU E1 暫時通過十項 AP 保護門檻、平衡分數高於舊最佳，但最終重載驗證尚未完成；不能提前更新 handoff。

## 使用者補充：ACB 也盡量避免同類問題

ACB 亦納入同一外層監測，啟動前仍依序執行原定 E0、train-only 梯度校準與短訓。GPU 維持單一 compute worker，有其他工作則避讓。這些措施可避免已知的資源競爭並降低中斷後的損失，但不能宣稱已消除 CUDA 根因。

查閱 [NVIDIA 同步與非同步錯誤說明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)：錯誤可能在較晚同步點浮現，`CUDA_LAUNCH_BLOCKING=1` 用於定位且可能顯著降低速度。因此未將同步除錯模式直接套用全程訓練，也未任意更動 driver、CUDA、batch 或精度。查到相同 driver 版本的圖形／RTX renderer 個案，但不是本次 CUDA 訓練呼叫，不能當作相同根因。

另核對 [PyTorch multiprocessing 說明](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)：需區分子程序使用 CUDA 的 fork 問題與本次主執行緒故障；目前資料 worker 使用 CPU，證據不足以更改正式 loader 的啟動方式。

外層恢復守門測試 9 passed：完整 epoch SHA、STOP、其他 exit code、毀損續訓檔、配方與來源鎖變動均涵蓋；未改鎖定實驗程式。測試先在缺少實作時失敗，再完成實作通過。新增管理層的程式 SHA 另寫入執行紀錄。

## 交付與恢復後驗證

已建立 [全部結果與優劣報告](../research/stage8-comparison-2026-11-08.md)，包含 9 列候選／起點、完整十項 AP、34 個已完成 epoch、同 checkpoint 訓練／部署差距、5 組參數成本及 optimize 原三組 E5 參考。資料截點為 09:34:48，ACB AP 空欄代表未取得結果，不是零。每列固定一個最佳 checkpoint，沒有逐欄挑選不同 epoch。

小型資料與 `redraw.py` 位於 `docs/research/data/stage8-20261108/`，約 0.66 MiB，無 GPU／checkpoint 可重畫。已實際重畫 2 張圖的 PNG／SVG，目視確認文字、圖例、數值及未完成標示可讀；9 列 CSV 與 JSON 精確一致、34 個 epoch 數量及 10 個本機連結檢查通過。同步研究索引、根層／專項 README、RUNBOOK 與工作紀錄索引。

新增外層完整生命週期測試，模擬第一次 SIGSEGV、重試成功或再次失敗，確認最多 2 次啟動（首次恢復＋1 次自動重試），並保留兩次事件證據。合計 11 passed（0.04 秒）。這是恢復控制測試，不是 CUDA 根因修復測試。

09:32:58 啟動外層 PID 3747815、原 queue PID 3747816，SiLU worker PID 3747822。09:40:25 確認 E5 已到 120／463 macro，EMA updates 1972 = 原 1852＋120，當次 AMP retries 0，shared／兩 heads 梯度存在。重啟後 worker log 無新錯誤，來源鎖與 E1 最佳權重 SHA 未變，ACB 尚在後續 queue。證據位於 `recovery-20261108-0925/verification.json`。

本次暫存 core／GDB 輸出共 6,631,176,303 bytes（約 6.18 GiB）已直接刪除，只保留小型定位證據；未動系統 apport 原檔，沒有另建 archive。沒有修改 `yolo_optimize`、正式資料、模型配方或 source.json。

若 E1 仍被選定，SiLU 的部署球棒 AP 落差將依原 protocol 觸發 3E 硬體適應；連同 ACB 5E 與驗證，暫估 11 月 8 日 14:30–16:00 完成，低 LR 延長或再次異常另計。交接模型目前仍是舊 Rep17＋20，不能把暫定 SiLU 結果當作已完成採用。依使用者偏好交由背景 queue 繼續，Stage9 未啟動。

## 困難與解法

CUDA 原生程式間歇性崩潰尚無可快速穩定重現的案例；用實際 core 與 fault handler 留下定位證據，恢復完整 epoch 狀態並設定有限重試。這次不調整學習率或 activation 來掩蓋執行錯誤。

## 14:05：更新全部結果與優劣

使用者再次要求顯示全部結果及優劣。唯讀核對 source guard、固定選型規則與本輪最新資料；沒有改動任何執行中配方。SiLU 在 09:40 再遇第三次 SIGSEGV，外層於 09:40:53 自動重試，之後順利完成原 5E。另完成原定 3E qSiLU／D1 適應，但依 rank 保留適應前 E1；選定模型 folding 及重載完整十項 AP 驗證 passed，AP 差異全部為 0，部署 SHA 為 `4335968378626630747e0424f2a346fef70c82e96333f65bd6854f7d506e4493`。

SiLU 適應段最佳 E1 的球 Pose 為 86.687213%、整體 Pose 為 90.186786%，但 COCO 球棒從適應前 47.512050% 降至 46.970516%；球棒 box 與球棒 Pose 也退步，平衡分數從 64.717811 降至 64.679711，故未採用。適應段的三輪完整 AP 另列，不冒稱已完成獨立最終重載驗證，也不和共同 parent 的獨立候選混成一條曲線。

ACB 的 E0、梯度校準及短訓通過，已完成 E1–E4，14:05 正在 E5、120／463 macro。最佳仍 E1：COCO 51.424385%、整體 box 61.557864%、整體 Pose 90.115353%、球棒 Pose 93.931342%。雖平衡分數 64.763296 高於 SiLU，球 Pose 相對 Stage7 降 0.173735 pp，未通過逐項門檻。ACB worker log 沒有 SIGSEGV、CUDA error 或 traceback；GPU 正常使用，未觀察到同類中斷。外層自動重試額度已用完，不重設。

更新同一份報告與資料，現在有 10 列方案（含未採用的硬體適應）、42 個已完成 epoch。SiLU 改為已驗證，ACB 改為已有暫定 AP；重畫主候選曲線與 AP 差異圖，新增適應段獨立曲線，均提供 PNG／SVG。同步根層／專項 README、RUNBOOK 與既有索引入口，沒有保存整套舊快照或複製權重。

整個 Stage8 仍在執行，現用 handoff 尚未改指向新模型；待 ACB 完整評估後按既定規則選型、交付並停止。未開始 Stage9。新的資料與圖表驗證結果續記於下。

14:09 驗證完成：10 列 CSV 與 JSON 精確一致、42 個已完成 epoch、6 個圖檔與 12 個本機連結通過檢查；已目視檢查 AP 差異圖與獨立硬體適應曲線。SiLU 部署 checkpoint SHA 及重載 AP 差異 0 再核對一致。ACB E5 已前進至 190／463 macro、EMA updates 2042、當次 AMP retries 0，shared／兩 heads 梯度存在；410 項來源鎖仍通過。證據保存於 `status-review-20261108-1405.json`。若未觸發額外延長且無新異常，剩餘訓練、驗證與交付估計約 14:35–15:00 完成。

## 15:04：Stage8 完成與無 RepConv 的推薦比較

使用者要求整理原本未加 RepConv 與各方法的推薦。先唯讀核對 optimize 原 Rep 結果、固定 protocol、410 項來源 SHA 及 reference-review 的 26 項主要參考 SHA。沒有更動研究方法、模型、配方、資料或啟動新 GPU 任務。

原 supervisor 與外層恢復均已 complete，完成時間 14:33:07。五組新候選各完成 5E，SiLU 另有 3E 硬體適應；五組最佳皆 E1，重載完整十項 AP 的差異全為 0。可攜包五組 CPU 160／640 逐值相同並拒絕錯誤 SiLU activation。五份部署權重 SHA 與 manifest／verify 一致。新主線為 rep17_20_silu，handoff 已由自動交付改指 result/stage8-retrain/handoff；本次只核對與整理，不重複複製權重。

加入 Stage7 無 Rep PolyQuality E10 參考，與無 Rep qSiLU E8 分開。維持 qSiLU 部署契約時推薦所選 Rep：COCO 球棒 +0.6085 pp、BBAT 球棒 box／Pose +0.1070／+0.1782 pp，但 person、球 box／Pose 微降。原模型已包含 B2／S1／D1；Rep folding 後單 3×3 卷積 MAC 與無 Rep 相同，不宣稱額外硬體加速。

PolyQuality 比所選 Rep 高 9／10 項，但 COCO 球棒較低 0.9018 pp；它是不同 activation 的既有模型，不是 Stage8 同預算控制組。實作需 4 次資料相乘形成二至五次冪及一般係數乘法，qSiLU 是 1 次平方與 dyadic 分段式；這是局部運算 proxy，不能當作整網成本比。PolyQuality 無局部整數 emulator，不自動改變 qSiLU 主線要求。

將同一報告更新為最終版本，11 列方案、43 個已完成 epoch，ACB 改為 5E 且最終驗證通過。新增十項 AP 的無 Rep／所選 Rep／PolyQuality 對照與推薦，補上無 Rep 26,453,936、雙 Rep folded 26,453,168 個參數的意義。同步根層 README、研究索引與計畫、專項 README／RUNBOOK、工作紀錄索引；清除文件的執行中敘述，保留原始歷史事件不改寫。沒有刪除唯一權重、產生備份或發布到外部。

本次整理的困難：無。前次 CUDA SIGSEGV 的根因仍未確認，ACB 已完成且未觀察到相同中斷；不能把重試成功當作根因修復。資料與圖表最後檢查結果續記於下。

15:09 最終查核通過：11 列 AP CSV／JSON 與 43 筆 epoch 一致，410 項來源鎖、35 份引用資料 SHA 及 7 份可用 checkpoint SHA 相符；主報告 12 個本機連結有效。三張圖各輸出 PNG／SVG，共 6 檔，已重畫並目視核對；重畫程式語法通過。補正參考列的 10E 明確屬於既有 Stage7，未新增控制訓練。本次未重跑 GPU AP 或模型測試，核對的是已完成的正式驗證證據。詳見 [驗證紀錄](../research/data/stage8-20261108/verification.json)。

## 未解事項與風險

單 seed、validation 選型、無等預算 control 加訓；未做獨立 test、RTL／HLS／實板量測。所選模型球 Pose 僅以約 0.001604 pp 的餘裕通過保護門檻，且五組均未通過原 optimize 的另一套工程門檻，不能宣稱全面進步。Stage9 特徵圖分析仍待指定。
