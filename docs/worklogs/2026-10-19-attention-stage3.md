# 2026-10-19：第 3 階段 normalization 與完整 PWL／reciprocal

**最終狀態：第 3 階段已於 11:57:24 完成並停止，選 D1 AP 0.514189349，無加訓。** 交付、KD 判斷與實際清理驗證見 [完成紀錄](2026-10-19-attention-stage3-results.md)。以下進度敘述是當時的執行紀錄。

## 授權與資料夾

使用者指定接續第 3 階段，並要求換到該階段資料夾工作。已將 Stage 2 選定 S1 實體搬到 `yolo_binaryqk/result/stage2/`，15 個檔案 SHA 全部相同，舊 run 僅保留相容符號連結。新程式與結果位於 `yolo_attention`；沒有重訓 Stage 2、啟動 Stage 4 或改動 BBAT5。

## 實作前來源核對

已先讀取 optimize `attention_recovery_v1/modeling.py`、正式 `pwl_contract.py`、封裝 runtime normalization；追查 Attention PWL／BDCN、final 的 strict checkpoint、materialize-before-evaluate、選型與歷史結果。來源與 SHA 先保存於 [reference-review.json](../../yolo_attention/artifacts/stage3/reference-review.json)。採用既有 PWL 範圍／endpoint 定義，D1／D2／D3 的完整整數 reciprocal 為本專案延伸。

[指定論文](https://www.mdpi.com/2079-9292/14/12/2337) 的 4.2／5.3 已核對：Newton 用於 LayerNorm 平方根，架構仍保留共用 divider；不把本輪 reciprocal PWL／Newton 當成原文原樣復現。原出版者頁面直接開啟回覆 429，改由原出版者搜尋快取取得內文。

## 進度與驗證

先建立會失敗的整數單元測試，再新增 reciprocal 設定、整數 leading-one／LUT256／PWL16／Newton64，以及共用 BitTrue PWL 整數 interpolation。輸出用 UQ1.24，沒有最後浮點強制正規化；訓練 proxy 與部署前向分開。新候選由既有 `HardwareFriendlyAttention` 承接 S1，不複製另一套 Attention 類別。

CPU 完整測試 **193 passed／1 skipped，25.77 秒**；跳過的是既有 CUDA BDCN 回歸測試。Ruff 與新 CLI help 通過。5 個實際模型（N2／N3／D1／D2／D3）嚴格存讀與長方形輸入等價、浮點 fallback 拒絕測試通過。三種 reciprocal 各檢查 **1,638,400** 個分母／邊界，最大相對誤差為 D1 **0.001953066**、D2 **0.000942201**、D3 **0.000071318**，均小於事前門檻。詳細數據見 [cpu-reference.json](../../yolo_attention/artifacts/stage3/cpu-reference.json) 與 [cpu-verification.json](../../yolo_attention/artifacts/stage3/cpu-verification.json)。

GPU 啟動前已向使用者說明內容與預估時間；11:47 核對 RTX 5090 無 compute process、顯示用量 611 MiB。**11:48:22（Asia/Taipei）**在 `yolo_attention` 啟動背景 Stage 3 queue，監測間隔 600 秒；沒有訓練。程式、上游 loader／evaluator、reference 的 SHA 已固定，每個工作先核對。見 [啟動紀錄](../../yolo_attention/artifacts/stage3/formal-launch.json) 與 [評估資料夾](../../yolo_attention/artifacts/stage3/formal-20261019-v1/README.md)。

前 16 張真實影像診斷已完成，兩處各四個 heads 都有資料；D1／D2／D3 最差 head 的 PV relative L2 分別約 **0.003240／0.003067／0.003135**。N7／N8／N9（Multimax Top-1／3／5）約 **1.382／1.023／0.993**，超過事前 0.5 門檻，只保留診斷負結果，不寫成 AP=0。其餘 9 個候選接續完整 val5000，正式結果尚待完成。

首組 N1 的完整官方 AP 為 **0.514165814**，接近 S1 的 0.514151658。依實際評估速度，預估本日 **12:00 前後（約 ±5 分鐘）**完成；此為估計，非已完成宣告。依使用者長工作退出 working 的偏好，確認來源未變、監測及 worker 存活後，保留背景 queue 執行，自動選型、產生報告、釋放 GPU 並停止。當下快照見 [working-handoff.json](../../yolo_attention/artifacts/stage3/working-handoff.json)，正式結果以 queue、comparison／handoff 為準。

## 清理、困難與未解事項

final 的算法本體多為相同程式的中文 docstring 版本，但訓練與 recipe API 存在實質差異。optimize 仍引用其外部路徑，因此先移除 **59 份、502,664 bytes** 重複 source，改為共用來源連結，並補上新 reciprocal 依賴連結。必要入口、歷史訓練 API、唯一權重與證據保留；沒有新增 archive／backup，也沒有再維護另一份 Attention 數學實作。詳見 [清理紀錄](../../yolo_attention/artifacts/stage3/cleanup-core-duplicates.json)。

相容驗證除了新版入口，另外以 `yolo_attention_final/src` 及 `final/` 兩條實際 Python import 路徑載入歷史 `pwl-final-best.pt`，CPU 64×96 輸入輸出有限值、兩處 Attention 正常。舊 README 已改正單獨可攜宣稱，說明需保留共用來源目錄關係。

已解決的困難：原論文網站 429 已以上述方式處理；首次測試指令引用不存在的舊測試檔名，已核對實際檔名。checkpoint 的 JSON sidecar 會把 class-name 字典的整數 key 轉成字串，原本直接比較產生假失敗，已改用 JSON canonical 比較，並在重建時明確還原 parent class names。相容整併後的新 reciprocal import 亦已補齊並由兩條舊入口實測。

未解事項：正式 AP 與選型仍在執行；沒有以數值誤差通過取代任務指標。若完整路徑精度不合門檻，保留推薦而不產生 handoff，再處理少數候選適應。仍有浮點 QKV／P·V／projection／Conv／SiLU；未做 RTL／HLS／板端量測。第 4 階段不自動啟動。
