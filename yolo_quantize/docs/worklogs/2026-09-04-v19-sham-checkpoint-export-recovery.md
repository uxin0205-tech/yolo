# 2026-09-04：V19 sham checkpoint 匯出修復與乾淨重啟

## 變更內容與原因

- V19 poly_shift + LSQ+ A8 / W8 matched sham 在 epoch 0 訓練、search validation 與雙 gate 完成後，於 materialized inference checkpoint 寫入階段退出。
- 根因是 deployment materializer 依 end-to-end 契約呼叫 head.fuse()，合法移除 training-only cv2/cv3；但 Full35 contract() 仍遍歷 detect_head.cv2，因此遇到 None。
- qat_runtime.py 新增單一 checkpoint seam：架構 contract 從 fuse 前的同一 EMA graph 深拷貝凍結，state dict 則仍取 fuse 後、移除 fake-quant 與 training-only branch 的 deployment graph。沒有把 cv2 暫時接回，也沒有把 training-only weight 寫入部署檔。
- 不完整 run 沒有刪除，完整移至 artifacts/runs/qat/v19-poly-shift-all-w8-qat-pilot-v1-sham-seed1-failed-inference-contract-v1/，並新增 failure.json。其中 full-resume checkpoints 保留供稽核；舊 inference files 明標不具權威性。
- 為避免 selector 與部署檔血緣混用，沒有從 epoch 0 checkpoint 續跑；改由乾淨正式目錄重新開始 matched sham。
- 新增 fail-closed 低頻佇列：每 300 秒只檢查 sham PID 與完成 manifest；15 epochs、plan/candidate/run identity、best_joint 路徑及 SHA-256 全部通過，且 GPU 沒有其他 compute process 時，才自動啟動同一 V19 的 QAT arm。任何缺證、非零退出或既有不完整 QAT 目錄都會停止佇列。

## 驗證方式與結果

- 最小 red 測試可在約 2 秒穩定重現正式 traceback：fused fixture 的 cv2=None 使舊儲存路徑拋出 TypeError: 'NoneType' object is not iterable。
- 修正後同一測試轉 green，並驗證 checkpoint contract 等於 fuse 前來源、state keys 等於 fuse 後 deployment。
- QAT runtime、validation、graph 相鄰測試共 14 passed in 13.25s，Ruff 通過。
- 真實 Full35 integration 測試確認 Detect 與 Pose deployment head 的 cv2 都是 None，仍可用正式 inference schema 寫入；結果 1 passed in 2.44s。
- 失敗 run 的 epoch 0 gate 本身通過：八項 total mAP50 最差 -0.00993784，八項 total mAP50–95 最差 -0.01256375。這只證明該 epoch 的 search gate，不把失敗 run 升格成完成實驗。
- 乾淨重啟確認讀取 COCO train 118,287 張與 canonical BBAT5 search-train 5,364 張；logical Detect batch 128、physical microbatch 16、資料 assignment、optimizer 與 augmentation 均未改。
- 佇列 identity、事件去重與 QAT completion checkpoint hash 三項單元測試全數通過；Ruff 與 git diff --check 通過。啟動後 status.json 為 observing_sham，精確掛接主程序 PID 757875。
- matched sham 完成15 epochs並產生hash-pinned manifest；best_joint來自通過全部gate的epoch 0。epoch 12–14因matched-drift gate失敗，不取代best checkpoint。
- Queue依契約啟動V19 QAT。QAT epoch 0完成後，在epoch 1套用0.2 blend ratio時退出；最小測試證實FP32 buffer讀回約0.20000000298，而controller使用嚴格不等判定。改用abs_tol=1e-7後，schedule/weight測試14項與Queue/runtime整合測試16項均通過。
- QAT epoch 0只有COCO box mAP50未過嚴格門檻：最差mAP50 -0.01744815、mAP50–95 -0.02224200。這是progressive QAT第一輪，不當成final結果；從尚未開始epoch 1 macro的正式last.pt精確續跑。
- 原始完整情境已重驗：Queue以resumed=true載入last.pt，epoch 1控制檔記錄blend ratio 0.2、396個quantizer parameters維持scale-only，並已產生macro 15；先前比例誤判不再重現。

## 困難與解法

- 困難：訓練與 validation 都成功，錯誤只在 epoch-end 的 deployment metadata 路徑出現，GPU smoke 未覆蓋實際 inference checkpoint serializer。
- 解法：先抽出 exact call-site seam 建立 red 測試，再補真實 Full35 fused-head integration 測試；後續不再只以 forward materialization 代表 checkpoint 可交付。
- 困難：父類別已先留下四個未 materialize 的 inference files，直接 resume 可能讓未再次入選的 selector 永久沿用錯誤檔案。
- 解法：保留失敗目錄供稽核，但不用它續跑；乾淨重跑確保 best_detect、best_pose、best_joint、last 都由修正後路徑建立。
- 困難：原schedule測試只覆蓋0、0.5與1，都是FP32可精確表示值，因而未捕捉0.2／0.4等實際五輪linear schedule比例。
- 解法：新增非二進位分數回歸案例並用明確FP32容差驗證read-back；Queue另增加需顯式旗標、且只接受run內checkpoints/last.pt的resume路徑。

## 未解事項或風險

- matched sham已完成；V19 QAT由epoch 0 last.pt續跑中，尚未形成完成manifest或gate-feasible best_joint。
- epoch 0 的良好數值不能外推至 15 epochs；仍須檢查每輪 mAP50／mAP50–95、absolute sham drift、checkpoint SHA-256 與最終 selector。
- 此次修正處理 checkpoint contract，不改變 PTQ、QAT 數學、權重、資料或 gate，因此不產生新的精度結論。
- formal validation、multi-seed、LS-SD4／ternary QAT 與最終 Pareto 鎖定尚未完成。
