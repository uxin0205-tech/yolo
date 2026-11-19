# 2026-10-21：HOG 第一輪精度停止、剝除重載驗證與交付

> 本頁記錄完成驗證當時的保留狀態。使用者隨後要求釋放 HOG 資源，未採用候選、E0 副本與 E1 resume 已刪；小型證據、原主線與 MASF 正式結果仍保留。最新見 [資源清理紀錄](2026-10-21-hog-resource-cleanup.md)。

## 範圍與結果

使用者詢問目前狀況；延續已授權第五階段，讀取實際 queue、日誌、GPU、完整 metrics 與 resume。HOG 校準、GPU smoke 均通過；第一輪完整訓練後於 12:14:26（Asia/Taipei）因精度門檻停止，並非 GPU／OOM／AMP 錯誤。沒有自動繼續原定最多十輪，也沒有進入第六階段。

整體官方 AP50–95 0.514620791 → 0.509686298（−0.4934 個百分點）；person 0.643206570 → 0.641292874；球 0.513224007 → 0.512737434；球棒 0.487594089 → 0.462289209（−2.5305 個百分點）。唯一觸發事前 −0.02 AP 停止條件的是球棒 AP50–95。完整八項 AP、差值及資料見 [正式結果](../../yolo_hog/artifacts/stage5/formal-20261021-v1/RESULTS.md)。

本輪不採用 HOG 適應權重，交接保留原 parent；未採用的實際 E1 候選單獨標明。第四階段 MASF 正式結果、候選與必要 resume 全部保留。

## 事前參照與診斷

沿用事前固定的 reference-review，重新核對 11 份 optimize 來源 SHA、parent／handoff／EMA source SHA；再次唯讀查閱 optimize 的 2026-09-09 HOG 負結果與 2026-09-10 歸因紀錄。沒有改方法、改停止條件或追加新配方。

以已保存 metrics 呼叫實際 HOGStudy.guard_metrics：E0 通過，E1 穩定重現 `HOG AP safety stop; completed epoch retained`，只觸發 baseball_bat_AP50_95。此為事前研究停止條件正常生效，不當作訓練程式故障修掉。resume 保存了完整 E0／E1 history、925 次更新與 E1 EMA，最佳 epoch 仍為 E0。沒有 `training/result.json` 是 guard 在 persist 後、一般結果寫入前拋出例外；正式證據沒有遺失。

## 變更與驗證

新增 `yolo_hog/scripts/close_safety_stop.py`，只針對此已保存的一輪負結果做獨立覆核，不修改原訓練程式、source.json 或原 queue 狀態。先核對 resume SHA、manifest、history，再從 E1 EMA 還原實際已更新的 Neck／head／HOG。live 與 EMA 的所有固定 tensor，包括 Attention 與全部 BN buffers，和 parent 逐位相同。

CPU FP32 160×160 的剝除前後完整輸出逐位一致，只移除 HOG 的兩個參數 tensor；訓練後部署權重 strict reload 逐位一致。啟動前已說明短時間驗證，確認 GPU 無其他 compute process 才使用；完整 COCO val5000 的八項 AP 與全部 official 欄位和 E1 完全相等。沒有額外 optimizer 更新。驗證不足十分鐘，過程有 progress 記錄；GPU 程序結束後釋放資源。

既有 16 項 CPU 測試與 GPU smoke 通過。本輪新增腳本 Ruff 檢查及格式檢查通過，實際停止條件重現與完整剝除重載驗證通過；未重跑無關訓練或擴大測試。另目視 smoke 的訓練標註預覽，沒有明顯全域框錯位；這不能替代全資料標註稽核。

在 `result/stage5/` 提供 selected.pt／selected.json 連到既有主線、rejected-epoch01.pt 實體、handoff、CSV、JSON 與 SVG。後者由覆核資料夾搬入 result，原位置保留相容連結；沒有複製整套 final。handoff 明確區分 Stage3 主線 loader 與 HOG 候選 loader、負結果 resume 與選定 parent 的 source resume。同步根 README、研究計畫、HOG README、工作紀錄索引及 cleanup 入口。

## 清理

先保存兩份 smoke 權重的完整路徑、SHA、大小與原因，核對正式 E1 resume、正式剝除權重及原始 smoke 小型證據均可用後，依既有直接清理授權移除 smoke/epoch-01.pt 與 smoke/resume.pt，共 355,280,756 bytes（約 338.82 MiB）。保留原 smoke metrics、resume SHA、strip proof、訓練圖片及測試結果，沒有 archive／backup。詳細執行情形以 run 的 smoke-cleanup.json 為準。

## 困難與解法

背景程序正確停止訓練，但原待 GPU 的交接快照未同步，queue 的 verification／selection 因上游 failed 被阻擋。本次讀取實際 resume 與 metrics，補獨立停止後覆核與交接，保留原 failed 事件作證據，更新 stale README 與 working-handoff。首次狀態查詢誤用上層 working-handoff 路徑，改查正式 run 的實際路徑後恢復；沒有改動實驗內容。

## 未解事項與風險

單 seed、只有一輪、沒有本輪 matched control；無法分離 HOG loss、Neck／head 適應及交互作用的因果貢獻，亦未證明延長一定無效。校準初始 raw P3 梯度範數約 5% 不能保證後續 AP 改善。小球零 cell 覆蓋盲區支持 H4 的局部研究方向，但球棒 prefix 沒有同樣問題，不能拿它解釋這次球棒回退。沒有立即啟動 H4、KD 或恢復訓練。

部署不增加 HOG 參數／運算；QKV／P·V／projection／Conv 仍浮點，全模型 qSiLU 依第七階段處理。沒有硬體實測加速宣稱。第五階段完成負結果交付並停止，第六階段等待使用者指定。

## 最終交接檢查

原 source guard、覆核腳本 SHA、selected／candidate／resume SHA、交接符號連結與新增文件連結均通過；CSV 包含八項正式 AP。關閉 CUDA 可見裝置後，僅使用小型 JSON 重畫 SVG 通過，plot 路徑不讀 checkpoint。新增腳本 Ruff 與格式檢查通過，範圍內 git diff --check 通過。獨立 val5000 前向耗時 43.43 秒，官方統計另計；不當作端到端硬體 latency。最後 nvidia-smi 無 compute process、GPU 利用率 0%，約 618 MiB 為顯示等既有占用。兩份 smoke 權重確已移除，正式 E1 resume 與 MASF 保留資料未刪。
