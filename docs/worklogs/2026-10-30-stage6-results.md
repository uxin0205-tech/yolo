# 2026-10-30：第六階段結果、optimize 同設定重驗與交付

## 任務與範圍

回應「目前狀況」，接續已授權的最佳融合研究及階段完成後交付。最後 Pose head 恢復已於 2026-10-29 23:47 完成，共 5 輪，因最佳 E1 後連續 4 輪未改善而停止。本次完成結果核對、optimize 同設定驗證、CPU 成本／路由檢查、result 交接與文件。第 7 階段未啟動，沒有新增訓練。

使用 `finish-work` 技能整理本機成果；遵循既有直接清理授權及單一主代理規則，沒有呼叫子代理、修改 optimize、commit 或 push。正式入口為 [第六階段報告](../../yolo_combine/result/stage6/README.md)。

## 變更內容與原因

選定 **KD E2 → Pose head 恢復 E1**，SHA `e9b9fd7ccc88f10bbd170835138ebde5c7810db0726424496e2cac2aa462ffe7`。E1 完整重載 AP 與訓練評估逐值相同；head 恢復五輪均保持官方 COCO 四項不變，BBAT 六項在選定 E1 均高於 KD E2。最終官方 COCO 0.512676、BBAT box 0.609202、Pose 0.899850、球框 0.493883、球關鍵點 0.863474、球棒框 0.724521、球棒關鍵點 0.936227。

之前 optimize 比較有 validation batch 差異。本次先唯讀核對其 CURRENT、最終 Rep17＋20 E5 報告、自包含 loader、runtime、manifest 與 validation，將來源 SHA 寫入 [事前 protocol](../../yolo_combine/artifacts/stage6/final-review-20261029/protocol.json)，SHA `af4b2b828f6cb00769c1d57f2166e1fa9c9e04544f7652b2c435d5ed0720d269`，再實作 `validate_optimize_reference.py`。

optimize 模型的 SHA 是 `9d9306f3574eb584c5adbe43c303afb2865109234ddf6dcb4c866b63901fc161`。使用其原自包含 runtime，避免當前同名 Attention 類別取代原實作；在獨立程序套用當前原生 evaluator，完整 COCO val 5,000／BBAT val 683、batch 8、FP32、rect stride 32、conf 0.001、IoU 0.7、max_det 300。圖片順序及 batch padding 的 SHA 檢查通過，GPU 評估耗時 68.21 秒。

同設定下，本輪相對 optimize：官方 COCO +0.000610、內部 COCO −0.000501、BBAT 框 −0.000554、Pose +0.009160、球關鍵點 +0.005185、球棒關鍵點 +0.013134。兩個 COCO 口徑都保留，沒有混用；逐類完整比較見 [comparison.csv](../../yolo_combine/result/stage6/comparison.csv)。架構仍不同，不能做單一元件因果歸因。

`finalize_stage6.py` 執行 CPU 檢查，將已停止的唯一 Pose head 成果資料夾移至 `yolo_combine/result/stage6/pose-head-recovery/`，原 artifacts 入口改成相容連結；`selected.pt`／`.json` 亦採相對連結。搬移後 SHA、strict load、全部 state tensor 與 manifest 一致，沒有複製第二份模型。保存的 resume 為 E5，不是 E1 的 exact resume；handoff 明確記錄差異。

新增最終 README、handoff、manifest、完整比較 JSON／CSV、118 輪分段 history、成本明細與 `redraw.py`。圖表只讀 CSV，無 GPU／checkpoint 依賴；P 副本診斷與其餘階段分開，不畫成一條累積改善線。同步根層 README、整合計畫、清理入口、專案 README 與工作紀錄索引。

## 成本與硬體狀態

640×640、batch 1、FP32 eval 前向的 Conv2d hook 記錄真實執行形狀，再計算 MAC。兩個獨立模型合計 78.122906 G，融合雙任務 47.325696 G，卷積小計減少 39.42%；參數由 45,428,208 降為 26,453,936，減少 41.77%。權重檔 106,576,996 bytes。未包含 Attention P·V、Hadamard、normalization、BN、activation、decode 與資料搬移，不能宣稱整網減少相同比例或實測加速。

保留 B2 sign＋Hadamard、16 個 S1 Q10 scale slots、分解式 signed16／1024 bias、D1 整數 PWL `[-10,0]`／20 段與 LUT256 reciprocal；UQ1.15 exp、uint26 分母、Q1.16 reciprocal、UQ1.24 probability、max_tokens 1024。沒有浮點除法重新正規化。QKV、P·V、projection、Conv 與目前 SiLU 仍有浮點路徑。qSiLU 屬第 7 階段，RepConv 屬第 8 階段；各 Rep 候選仍須完整繼承硬體契約。

報告另外列出 head sigmoid、Pose26 座標乘加、E2E top-k 與座標還原的剩餘成本；reg_max=1，DFL 為 Identity。初始化插值、KD、task weights、RealNVP loss 屬訓練／初始化，不當作推論新增成本。未做 RTL／HLS、實板、延遲、能耗或新形狀研究。

## 驗證方式與結果

- 沿用本輪 trainer 已完成的相關 CPU 測試 **23 passed**，沒有在交付時重跑未變更的訓練測試。兩個 Pose macro GPU smoke 通過，258 個參數 tensor 更新，固定 live／EMA tensor 不變。
- 選定 E1 完整 COCO／BBAT AP 重載完全重現；optimize 原包 CPU preflight 通過後，完成同設定 GPU 驗證。
- CPU 640 輸入下，Detect／Pose 單任務與 `both` 對應輸出逐值相同；搬移前後權重與設定相同。
- 三個交付／重驗／重畫腳本的 Ruff 與格式檢查通過；26 個文件連結、118 筆歷程、12 項比較與 12 個參考來源 SHA 核對通過。圖表已重畫並目視檢查，最終紀錄見 [verification.json](../../yolo_combine/artifacts/stage6/final-review-20261029/verification.json)。
- GPU 評估完畢後無 compute process；最後一段背景 supervisor 已正常完成。沒有停掉其他使用者工作。

正式 BBAT 資料為 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`，registry 為 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。新工作未改影像、標註或 split；runtime View 只保留正式 assignment。COCO train／val 118,287／5,000，BBAT train／val 5,964／683。

## 清理與保留

已核對先前清理：KD E1／E3／E4／E5 未採用 inference 與已停止 E5 resume 共 10 檔、**742.60 MiB** 已刪，E2、小型結果與來源鎖保留，見 [逐檔清單](../../yolo_combine/artifacts/stage6/recovery-kd-20261029/cleanup-20261029.json)。更早舊 Stage6 清理釋放約 5.562 GiB，見 [重建紀錄](2026-10-29-stage6-clean-rebuild.md)。本次交接以搬移／連結完成，不建立 archive、backup 或另一套 final；必要 parent、原獨立 Pose、MASF 保留結果、正式資料與 optimize 外部依賴未刪除。

歷史 run 的 result、來源鎖及原始精度保持原樣；最終研究交付狀態由 `result/stage6/handoff.json` 表達。既有未通過精度門檻的欄位不改寫成成功。

## 困難與解法

資料設定快照初次 JSON 寫出遇到 PosixPath 不可序列化，在任何新 GPU 工作前加入明確 Path→字串轉換後完成。optimize 與當前程式存在同名但版本不同的 Attention 類別，因此採獨立程序與 runtime 路徑斷言載入原模型，再對齊 evaluator，避免錯誤重建。

交付腳本的初次 lint 指出迴圈 callback 應明確綁定目前模型，以及重畫腳本不需要的 noqa；已修正並重新核對。callback 原本即同步執行，修正後另外以小型 Conv 前向確認 MAC hook 行為，不重跑 GPU AP 或搬移成果。

成本交付檢查初次把每處 QKV 視為單一 Conv，實際紀錄是 Q／K／V 三個 Conv；改為逐一檢查三者的 20×20 輸出，確認每處 400 tokens，原 MAC 統計不受影響。

完整 Pose 副本曾將 BBAT box／Pose 提升至 0.631795／0.918548，但 COCO 降至 0.465314；J1 回退診斷未找到能同時修復兩任務的單一元件。KD 恢復部分 Detect，固定共享層的 head 適應再改善 Pose。後續相同 head 配方已停滯，依事前預算停止並交付問題，不再只因未達原目標就無限延長訓練。

## 未解事項或風險

原驗收門檻仍有四項未達：COCO 球棒、BBAT 整體關鍵點、球框、球關鍵點，超出原容許下降後還差 0.011753／0.000859／0.015212／0.013625 AP。研究工作已完成本輪交付，但 `original_accuracy_gate_passed=false`、`automatically_promoted=false`；是否接續使用此模型由使用者覆核。第 7 階段尚未開始。

單 seed、沒有統計顯著性或硬體實測；本輪選定是已評估候選中的綜合選擇，不代表所有融合方法的最優結果。尚未重試 MASF、完整硬體化全部非線性或變更輸入形狀，保留原計畫的後續階段責任。
