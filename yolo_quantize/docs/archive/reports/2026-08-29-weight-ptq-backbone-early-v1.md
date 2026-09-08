# Full35 backbone_early PTQ bit sensitivity 第一階段結果

日期：2026-08-29

## 結論

`backbone_early` 的第一階段已完成三條 A8 activation parent × W8/W4，共 6 格無訓練 PTQ；6 格皆維持 graph structure 且輸出有限值，但這只代表執行成功，不代表精度 gate 通過。

- W8 是目前唯一可送完整 validation 的 PTQ 候選。
- W4 在三條 activation parent 都出現 TopK 候選集合 collapse，因此只淘汰「backbone_early 直接 W4 PTQ」，不等於永久淘汰 W4 QAT 或更細粒度 mixed policy。
- W6 是合理的 conditional rescue，但依使用者要求暫停 GPU，尚未執行。
- 尚未計算 COCO box、BBAT box、BBAT pose mAP，所以不能套用 0.01 incremental gate 或宣稱 W8 已安全。

## 實驗契約

- activation：qSiLU＋A8、poly_quality＋A8、poly_shift＋A8，各自載入並驗證 short-recovery parent checkpoint。
- weight region：`backbone_early`，21 個 Conv、1,218,240 個 weights。
- weight PTQ：signed uniform、per-output-channel、nearest-even、MSE clipping scale。
- matched reference：同一 activation parent、同一 A8 observer、FP32 weights。
- calibration：每個 task 兩張 canonical train exemplar；probe：每個 task 一張 canonical val exemplar。
- BBAT5：沿用不可變 `bbat5-v1`，未建立新 split、未抽樣或改動 assignment。
- training／QAT：皆未執行。

## 六格 scorecard

| Activation parent | Weight | Weight NRMSE | Worst raw NRMSE | Minimum TopK overlap | 約保留候選 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| qSiLU＋A8 | W8 | **0.011977** | 0.106610 | 0.4800 | 144／300 | 保留，待 validation |
| poly_quality＋A8 | W8 | 0.011978 | **0.100267** | 0.5500 | 165／300 | raw 最佳，待 validation |
| poly_shift＋A8 | W8 | 0.011978 | 0.101581 | **0.6000** | 180／300 | TopK 最佳，待 validation |
| qSiLU＋A8 | W4 | 0.161622 | 0.261582 | 0.0233 | 7／300 | PTQ 淘汰 |
| poly_quality＋A8 | W4 | 0.161623 | **0.257891** | **0.0300** | 9／300 | PTQ 淘汰 |
| poly_shift＋A8 | W4 | 0.161623 | 0.263882 | 0.0167 | 5／300 | PTQ 淘汰 |

表中的 raw NRMSE 與 TopK overlap 都是相對各自 matched activation parent 的 proxy，不是 mAP。三個 parent 的起始絕對 mAP 不同，因此不能因 qSiLU 的相對 proxy 較弱就直接刪除它；qSiLU 仍保留 accuracy-headroom 角色。

## Weight數值與任務輸出為何不能混為一談

W8 的 weight NRMSE 只有約 0.012、SQNR 約 38.43 dB，但較差任務的 raw NRMSE 仍約 0.10，Minimum TopK overlap只有 0.48 至 0.60。這證明小的 weight reconstruction error 經 backbone 與後續非線性傳播後，仍可能改變候選排序。

W4 的 weight NRMSE升到約 0.162、SQNR降到約 15.83 dB；TopK overlap只剩 0.0167 至 0.03。這不是輕微排序變動，而是較差任務只保留約 5 至 9 個相同 Top-300 class–anchor pairs。

其他數值範圍：

- W8 clipping rate約 0.037% 至 0.038%，zero-code ratio約 2.78%。
- W4 clipping rate約 0.570% 至 0.571%，zero-code ratio約 29.11%。
- 三個 parent 的 weight NRMSE幾乎相同，但 forward proxy不同，正好顯示 activation–weight coupling 不能拆開選 winner。

## 容量收益

下表只計 `backbone_early` weight codes與每輸出channel FP32 scale，不含 bias、requant metadata或實際封裝 padding。

| 格式 | Code bytes | Scale bytes | 合計 | 相對 FP32 |
|---|---:|---:|---:|---:|
| FP32 | 4,872,960 | 0 | 4,872,960 | 1.00× |
| W8 | 1,218,240 | 9,472 | 1,227,712 | 約 3.97× |
| W6（推估，未跑） | 913,680 | 9,472 | 923,152 | 約 5.28× |
| W4 | 609,120 | 9,472 | 618,592 | 約 7.88× |

W4雖然容量最好，但本次 PTQ proxy 已 collapse；不能只依壓縮率晉級。

## 建議選擇

GPU重新允許後，建議順序是：

1. 先補三條 parent 的 W6 rescue，共 3 格；它能檢查 W8與W4之間是否存在較好的非2冪位寬折衷。
2. 若 W6沒有 collapse，將 W8／W6 的非劣 policy送完整 COCO／BBAT5 validation。
3. W4不再做同區域純 PTQ；未來若仍要追求 W4，只能改成低 learning-rate QAT、將第一層留 W8，或拆出更細的 layer sensitivity。
4. 完成這個停點並由使用者選擇後，才進 `backbone_deep`，不自動往後跑。

若希望最快取得 mAP 結果，也可以跳過 W6，直接對三條 W8 policy做完整 validation；代價是暫時失去 6-bit 折衷資訊。

## GPU暫停狀態

使用者指示 GPU 需讓其他人優先使用，因此下列工作尚未啟動：

- backbone_early W6 三格。
- backbone_early W8／W6完整 validation。
- backbone_deep及後續region PTQ。
- 任何 QAT 或正式訓練。

## 可追溯產物

- 完整 JSON：`artifacts/reports/weight-ptq-backbone-early-v1.json`，SHA-256 `27ce8fe399cba8ab4e89e4ced955eeadd2a2206e0c74c2ac0e4a54eb7ea90c5e`。
- CSV scorecard：`artifacts/reports/weight-ptq-backbone-early-v1-summary.csv`，SHA-256 `83a5ae24ee507733ee1a0584886fe22075a1b089bf2c7c780fd4bdd4af8945a2`。
- 初始 tracer JSON：`artifacts/reports/weight-ptq-smoke-v1.json`，SHA-256 `88cdfa9e2ac888593d39445f728817393885bd907cbeaa567f24d99bc8dd2599`。
- Frozen source plan：`configs/experiments/weight-region-sensitivity-plan-v1.yaml`，SHA-256 `070dc5d08729f5d938d7667d560372d10c36ed5d1c6629261344a0fbe8f39253`。
- Result manifest：`configs/experiments/weight-ptq-backbone-early-result-v1.yaml`。
