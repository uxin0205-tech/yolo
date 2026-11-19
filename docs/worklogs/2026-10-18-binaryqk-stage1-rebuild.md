# 2026-10-18：Binary 第 1 階段重建、清理、驗證與背景工作交接

> 後續更新：使用者要求錯誤產物直接刪除、不封存；舊 v1 權重／resume／排程已移除，原產物連結改指向刪除清單與小型精度證據。現行方法以第 1 階段 v2 protocol 為準。

> 後續更正：B1 於 05:07 完成；B2 因 GPU 占用在 preflight 退出，B3 未啟動。核對 optimize 後停止延續舊 Q/K-only 配方，目前沒有背景長訓；下文原啟動 ETA 已失效。此文保留當時交接歷史，最新狀態及原始 parent 0.5180／0.5247 重驗見 [配方稽核紀錄](2026-10-18-binaryqk-status-and-protocol-audit.md)。

## 任務與範圍

接續 2026-10-17 的硬體友善與形狀分析補充，依使用者「可以先行開始」及 2026-10-18「繼續」啟動第 1 階段。只處理 `yolo_binaryqk` 及根層文件；`yolo_optimize` 唯讀，未指定資料夾不清理。採單一主代理。

使用者另指定：每階段完成後先停止並交付結果，等指定下一階段；長時間訓練確認穩定後退出 working，說明大約何時結束。GPU 使用前說明，確認其他工作是否會受影響。背景每 600 秒監測，異常或完成立即記錄，不會推進第 2 階段。

## 變更內容與原因

- 重建 `binaryqk/`，只提供 B1／B2／B3。精確 XNOR／popcount 前向與 signed-dot／clipped STE 反向參照 optimize，B3 共用 sign，合併 score 後一次 normalization、一次 P·V。
- 固定共同 YOLO26m parent、兩處 4 heads、key_dim 32。嚴格核對 SHA，拆分 native per-head `[Q,K,V]` channel layout，保留 V、PE、projection 與其他權重。
- 以共同 parent 的 128 張 COCO train 影像校準固定 FP32 scale；B1／B2 為 16 slots，B3 為 24 slots。第 2 階段才研究 learned Q10／二次冪。
- 新 checkpoint 攜帶 parent SHA、architecture、bases／sites、scale、bias、normalization／reciprocal、activation call-sites 及 Rep 狀態，使用 `weights_only=True`、嚴格載入及 sidecar SHA。缺契約或實際 normalizer 被替換即失敗。
- 新增完整 COCO runtime View 與唯讀資料 guard；完整 train 118,287／val 5,000。快取只寫在本階段 `artifacts/datasets/`。沒有抽換資料、修改標註或重切 split；BBAT5 v1 與 Pose 完全不動。
- 建立 E2E FP32 evaluator，分別保存 internal 與官方 COCO bbox，固定 overall／person／ball／bat 各兩項 AP。原始不同口徑歷史結果只供參考，不加訓 control。
- 建立 Q/K-only 訓練、固定 tensor 稽核、僅更新可訓練權重的 EMA、FP32 checkpoint 與 epoch 邊界 resume。三組共同 10 epochs／seed 0／batch 8／logical batch 128；理由與完整 recipe 見 [protocol](../../yolo_binaryqk/docs/stage1-protocol.md)。
- 新增背景 supervisor，600 秒監測、5 秒程序結束輪詢、STOP 機制、失敗停隊列；完成 B1～B3 自動輸出 comparison JSON／CSV／中文結果入口，不啟動後續階段。
- 同步整合計畫、根層 README、`yolo_cleanup.md`、專項 README 與本索引。

## 清理內容與保護資產

先抽取 169 個舊 run 的方法、設定、指標、訓練曲線、錯誤摘要及資料 assignment，保存約 7.1 MB 的小型歷史證據。26 份 canonical 權重逐一核對 SHA 與 strict reload 後，按逐檔 hash 清單移除 2,722 個檔案、555 個空目錄。被移除檔案的獨立配置區塊合計 3,050,942,464 bytes，約 2.84 GiB；這不是整個共用檔案系統的 free-space 差值。

清除舊 CLI／trainer／KD／矩陣腳本／services／舊 tests、被替代的設定、重複 checkpoint、已抽取證據的大型 logs 與重畫產物。未建立另一整套 legacy／archive。保留：

- 26 份正式唯一權重及必要 metadata。
- 32 份尚未驗證替代方式的唯一 checkpoint 與其重建設定，沒有憑「舊」或「失敗」猜測刪除。
- `binary_attention/` 的 9 個最小歷史載入相容檔案；移除 package import 時建立 cache 的副作用。
- `artifacts/reports/coco_area_metrics.csv`，因 optimize 的 `audit_accuracy_regressions.py` 仍讀取此確切路徑。
- `yolo_p2`：目前安裝的 Ultralytics 8.4.90 實際從此資料夾載入，尚不能刪除。

完整逐檔清單與結果見 [cleanup-plan](../../yolo_binaryqk/artifacts/migration-20261017/cleanup-plan.json)、[cleanup-result](../../yolo_binaryqk/artifacts/migration-20261017/cleanup-result.json)。歷史文件引用已刪 run 的路徑仍代表當時來源，必要數據改由整併證據查閱。

## 驗證方式與結果

- 最終 CPU 聚焦檢查共 33 項通過，Ruff 全部通過；97 個本次文件入口連結有效。新增 EMA／resume、silent normalization 防護已包含。
- B1～B3 皆以實際 native Detect loss 在 CPU 驗證：兩處 Q/K 有非零梯度且實際更新；全部固定 tensor 逐位一致；Softmax／P·V 各呼叫一次；FP32 checkpoint 重載輸出最大誤差 0。
- 128 張 640 × 640 真實影像校準完成，兩處 Attention 皆為 20 × 20。這是目前模型實測；計畫中的 40 × 40 是條件算例，最後形狀分析須依實際 site／stride 推導。
- 完整 loader 掃描：train 117,266 個有標註影像＋1,021 個 background，共 118,287；val 4,952＋48，共 5,000；0 corrupt，會員集合與 canonical list 一致。
- RTX 5090、FP32、batch 8：B1／B2／B3 median 約 0.084／0.092／0.098 秒，peak allocated 約 3.29／3.28／3.30 GiB。每組 2 個 warmup＋8 個量測 batch，含 backward／optimizer；這是 PyTorch 參考程式資源量測，不是硬體加速證明。
- 實際 B3 訓練拼圖已目視檢查，框與目標對應合理；繪圖產物由獨立 labels 副本產生。
- B3 完整 trainer／supervisor 短測通過：256 張訓練觀測、2 次累積 optimizer step、EMA、固定 tensor、best 重載、optimizer resume 及完成狀態。不是完整 epoch，短測中的 selection placeholder 不是正式 AP。
- 清理後再驗證 26 份歷史正式權重，全數仍可 strict reload；受保護 CSV／唯一 checkpoint／metadata 與唯讀來源 SHA 不變。正式 5,000 張 AP、10 epochs 結果與 winner 仍待背景工作產出；不把短測結果當作階段完成。

## 困難與解法

1. 一般 shell sandbox 的 `bwrap` 回報 loopback 權限錯誤。改用工具核准的工作區指令執行，範圍仍限制於本專案；未修改系統網路或權限。
2. 本機與舊設定的資料路徑不同。使用現有 `/home/uxin/yolo/coco2017`，核對原始 train／val 清單 SHA 與數量，沒有另建 split。
3. 盤點中一度出現其他 GPU Python 工作，持有約 2.7 GiB；當時未啟動 GPU。後續重新確認沒有 compute process 才執行短測。
4. 原生 `plot_images` 會改寫傳入的 labels 為 NumPy，首次 GPU 短測在 loss 的 tensor `.view()` 失敗。改傳獨立 CPU tensor／metadata 副本，三組重跑全部通過。
5. Native EMA 對固定浮點 tensor 也會做平均，可能造成微小漂移。只平均明確可訓練 Q/K，CPU 與 GPU 短測驗證固定 tensor 逐位不變。

## 未解事項或風險

正式訓練仍是單 seed、Q/K-only 的同預算比較；結果不保證優於 parent。FP32 scale、Softmax、SiLU、QKV／P·V 仍有浮點運算，Q10／完整 PWL reciprocal／qSiLU 依後續指定階段處理。沒有 RTL／HLS 或實板測量。

唯讀 guard 與 evaluator 依目前本機 Ultralytics 8.4.90 核對；升級或遷移 `yolo_p2` 必須重新驗證。正式訓練中途停止時可從最後完整 epoch 重跑，不宣稱 mid-batch 的完全連續恢復。

GPU 短測推估三組 10 epochs 加評估約 12～15 小時，正式進入訓練後依實際進度更新。背景監測只能寫入檔案，退出 working 後不宣稱會自行在聊天推播；可從 supervisor／各 run status 檢視。其他階段尚未啟動，未提交或發布 Git 變更。

## 正式背景啟動

於台灣時間 2026-10-18 01:22:44 啟動第 1 階段隊列，supervisor PID `634140`。B1→B2→B3，三者均由共同 parent 獨立初始化。啟動前確認 GPU 沒有其他 compute process。初估全部完成於 2026-10-18 13:22～16:22（Asia/Taipei），實際時間依資料讀取與 GPU 共用情況調整。

- [啟動命令與程式 SHA](../../yolo_binaryqk/artifacts/stage1/cleanup-invalid-v1.json)
- [隊列狀態](../../yolo_binaryqk/artifacts/stage1/cleanup-invalid-v1.json)
- [背景監測](../../yolo_binaryqk/artifacts/stage1/cleanup-invalid-v1.json)
- [B1 即時狀態](../../yolo_binaryqk/artifacts/stage1/cleanup-invalid-v1.json)

程序完成後在相同隊列目錄產生 `RESULTS.md`、`comparison.json`、`comparison.csv`；未完成前不建立假結果。退出 working 後由已啟動的背景 supervisor 繼續監測；需要完整方法分析與下一階段採用決策時，再由使用者檢視階段結果。

退出 working 前，B1 已完成 5,000 張正式零訓練評估：官方 COCO bbox AP50–95 為 0.401422、AP50 為 0.574796。這是 B1 替換後的起始精度，不是訓練完成結果或最終 winner。完整八項 AP 在 B1 的 `metrics-00.json`。
