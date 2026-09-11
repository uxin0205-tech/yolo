# 2026-09-09：另開融合前 B100 方向 1

## 變更與原因

依使用者要求建立 `studies/pre-fusion-full35-b100/`，以架構專案 final 的 `full35-b-f10` 為明確 parent。保留原融合後的 README、計畫、權重與全部 artifacts，只新增研究分流入口。B100 舊 gate=rollback 已明記，並未擅自換成 A2。

新增方向 1 適配計畫、唯讀来源與 runtime View、CPU preflight、完整 COCO 驗證入口及 600 秒 blocking supervisor。單獨 Detect 不套 joint score、Pose loss 或跨任務梯度投影。

## 驗證與結果

CPU preflight 已通過：Float SHA256 `81dc8629fc6db2915242ba950c522c249d89ba1aa6d53ab39389c9b5a4ed0047`；Bit-True SHA256 `a8780ae988e61905fa78134b16bf362d2392d2a71f43941f770769bd42265cae`。兩者 21,973,037 parameters、P3/P4/P5=(16,19,22)、MASF alpha=0.1031494140625。state 差異僅兩個 normalization 的 knots／values 與 Bit-True endpoint buffers，沒有宣稱完整 forward 等價。

兩份 checkpoint 均 epoch=-1、optimizer=None、EMA=None；後續是 fresh optimizer／EMA 微調，不是 exact resume。完整 COCO 清單保留 train118287／val5000，快取只在新 runtime View。

首個 GPU baseline 完成推論但保存 summary 時存取了不存在的 `YOLO.validator` 而失敗；這是新入口的 API 錯誤，不是模型精度失敗。保留原 log／predictions，修正為捕捉 validator instance，另開 `baseline-bittrue-v2` 完整重驗，exit0。v2 固定 internal evaluator，避免 canonical AP 混入主分數。未在原始 final 寫入。

## 使用者補充：先 BinaryQK，後 MASF

已核對 `achitechure_1/inputs/parent/provenance/final-results.json`：正式 attention winner SHA256 `c989aeed09de7663ad093d32d098e5fc889cf04924fa1162efaf886869de0123`，與 final/models.json 的 A0 完全一致，證明 MASF 研究 A0 已是 BinaryQK attention parent，不是官方未改造 FP。

`inputs/parent/final-selection.json` 記錄 B26-FP AP=0.517998、正式 parent=0.5067368995935831、缺口=0.011261100406416813。與舊研究另一階段的 0.011641 不同，報告不得混成同一差值。歷史 final 同機重驗 A0=0.506754、A2=0.506391、B100=0.503503；能比較該表的改造／續訓變化，但不是配對重訓對二值化順序的因果證明。

## 困難與解法

- 既有 joint runner 不相容單 Detect，改用獨立來源 API，不把 Pose 假裝關掉後沿用 joint gate。
- 原生 verifier 可能寫 cache／修復圖片，沿用已稽核 readonly guard 並建立隔離 View。
- sandbox 的 bwrap helper 失敗；使用授權的 scoped escalated command／apply_patch，原始 final 保持唯讀。
- 驗證 summary API 問題已修正並於新 run 重驗；失敗產物保留可追溯。

## 未解事項與風險

尚未完成 HOG、RepConv、MASF bridge 或 BinaryQK recovery 訓練，不能宣稱方向 1 已完成或 AP 已回升。尚須檢查新 parent 的真實 Q/K score 梯度、EMA／criterion 起點與最小 smoke，再開始正式訓練。舊研究結論僅參考，不當作 B100 新結果。硬體 latency／energy 未測。
