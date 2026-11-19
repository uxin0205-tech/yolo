# 2026-11-17：全輪研究整理、統一交接與清理

## 任務與範圍

使用者要求「結束後幫我把整個東西都整理起來」。Stage9 含 768×512 已完成 120E；本次整理第 1～9 階段的正式選擇、指標、真實血緣、可用權重與目錄責任，直接處理既有授權內確認無用的產物，不新增實驗、訓練或 GPU 工作。

沿用使用者歷次「錯誤／無用內容直接刪除，不封存」授權。`finish-work` 的收尾流程由主代理執行；清理已具前述明確授權，不重複索取確認。未啟動子代理，未 commit／push。

## 動手前參考與保護邊界

唯讀核對 optimize 的 `experiments/src/yolo_optimize/runtime.py`、`experiments/combine/bridge_v1/safe_source.py`、`reports/current-model/CURRENT.json`、`model/environment.json` 與 `experiments/benchmark/manifest.json`。它們證明舊 `full35`、Pose 來源、`achitechure_1`、`yolo_p2` 等仍被讀取；不能因資料夾名稱舊就整個清除。上述來源 SHA 已記入清理保護清單。

本次不改研究方法／run／配方／parent；各階段原 reference review 與 source lock 仍是實驗依據。正式資料仍為 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`，registry／Pose／Detect YAML 保持；未改任何 split、影像或標註。`yolo_optimize`、原始資料、已訓 Pose 與未指定重建的 `yolo_quantize` 保持。

## 變更內容與原因

1. 新增根層 [result/](../../result/README.md)，以相對連結提供正式模型包、多分支 EMA 與小型數據，沒有複製另一套大型 final。manifest 明列正式 SHA、算子、原 rect640、十項 AP、來源、限制與沒有精確 optimizer resume 的事實。
2. 新增 [研究總報告](../research/yolo-research-summary-2026-11-17.md)，整合九階段採用／未採用原因、融合方法、全模型簡化邊界、120E 形狀研究、長方形與二次冪分析，以及剩餘硬體工作。
3. 新增 7 個血緣節點、16 列形狀指標、Stage8 小型比較副本、來源 SHA，以及只用 CSV 的兩張 PNG／SVG 重畫程式。COCO／BBAT 分開，分析 MAC 與 GPU 實測時間分開；沒有把每欄最高 AP 拼成不存在的模型。
4. 重整根層 README、研究索引及計畫狀態，修正 Stage3 KD 未採用、Stage7／9 待開始、總輪數 100E 等過時現行描述。原實驗決策不改寫；Stage8 自動 SiLU 推薦與使用者最終 qSiLU 選擇明確區分。
5. 為 Stage1／2 新增 `availability.json`，保留不可變的歷史 handoff，另外標示已刪 optimizer resume；最佳模型仍可載入，不能再宣稱可精確續跑。

## 實際清理

刪除 6 份已完成的 optimizer resume：

- `yolo_binaryqk/result/stage1/resume.pt`
- `yolo_binaryqk/artifacts/stage1/formal-20261018-v2/B1/resume.pt`
- `yolo_binaryqk/artifacts/stage1/formal-20261018-v2/B2/resume.pt`
- `yolo_binaryqk/artifacts/stage1/formal-20261018-v2/B3/resume.pt`
- `yolo_binaryqk/result/stage2/resume.pt`
- `yolo_binaryqk/artifacts/stage2/formal-20261019-v1/S2/resume.pt`

另清理 92 個可重建 Python／測試快取，共 **98 檔，1,296,444,995 bytes（1.207408 GiB）**；移除檔案配置區塊共 **1,296,707,584 bytes**，空快取目錄另移除。本數字不重複計入先前 Stage9 的 3.57 GiB 與直向 0.63 GiB。

逐檔 SHA、大小、理由、刪除狀態與保護檔案見 [cleanup-20261117.json](../../result/cleanup-20261117.json)。刪除前檢查沒有本研究 active process，所有路徑重新解析且不是符號連結，nlink=1，大小與 SHA 相同。494 個受保護檔案在刪除前後 SHA 一致，包含選定權重、候選最佳、來源鎖依賴與資料入口。

保留 MASF 正式候選與必要 resume，因使用者先前指定等融合後再決定；Stage3 KD resume 因 HOG source guard 仍核對其 SHA 也保留。optimize 仍用的外部目錄、activation 外部量化權重、虛擬環境、使用者簡報及無法證明無用的檔案未刪。

## 驗證方式與結果

- 494 個受保護檔案、466 項來源鎖與既有同配方 guard 通過；正式包 372 個檔案 SHA 相符，folded／多分支 EMA SHA 與使用者選擇一致。
- 從新入口執行 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B result/model/load_shape.py`：CPU 160×320／640×640 的完整輸出逐位一致，錯誤 SiLU 被拒絕，GPU 未初始化。
- 16 列形狀數據與兩份原始 CSV 一致，Stage8 兩份小型副本逐位一致；7 個主線節點可追溯，未量測的任務欄位留空。
- 重畫兩張 PNG／SVG 並逐張目視確認標籤與座標軸。程式只讀 CSV，不匯入 PyTorch、不讀 checkpoint。
- 新增／重寫的現行文件共 162 個本地連結完成核對；先發現研究索引兩個已不存在的舊融合紀錄，已改連到實際 2026-10-29 重建及 10-30 完成紀錄。後續只重查文件及資料，未重跑模型測試。
- 98 個已清理路徑均不存在；新 result 的三個相對連結正常，Stage1／2 availability 與實體檔案一致。

詳細檢查記於 [validation.json](../../result/validation.json)。以上只用 CPU 與檔案檢查；既有 GPU AP／folding／replay 數據沿用原正式證據，不重跑已通過的 GPU 工作。

## 困難與解法

歷史 README、handoff 與自動選型報告存在不同時點的結論：保留小型原始證據與其 SHA，重寫現行入口、加上歷史註記及 availability overlay，避免修改已被下游鎖定的 manifest。研究索引另有兩個既已不存在的舊融合工作紀錄，改連實際重建／完成紀錄。Stage5 球／棒欄名與其他階段不同，整合表依相同 checkpoint 的 Stage3 指標對齊，沒有補造未量測的 BBAT AP。

部分資料夾看似舊內容仍有程式或量化外部依賴；先讀實際引用，僅清除能確認已完成且不用的 resume 和快取。未遇到 GPU 衝突，本次沒有使用 GPU；其他專案的 GPU 程序未干預。沒有自動審批拒絕。

## 未解事項或風險

正式選定不代表所有指標最佳或所有原門檻通過。仍有單 seed、validation 選型、缺少同預算 control、融合精度缺口、剩餘浮點／非線性／除法，以及未做整網定點、RTL／HLS、板端量測的限制。固定輸入下獨立改內部 stride 的架構變體尚未實作。新電腦安裝、ONNX／TensorRT 匯出未驗證。

已刪 optimizer 狀態不可精確恢復；保留最佳 checkpoint 可開新訓練段。正式包若移到另一台電腦，須複製連結指向的完整實體內容與準備第三方依賴。本輪完成後停止，未啟動下一研究。
