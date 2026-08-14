# BBT5 Clean 完整公平實驗計畫

更新日期：2026-08-14。這是目前唯一有效的正式實驗計畫；尚未啟動 GPU。

## 1. 固定輸入與資料規則

- Dataset：完整保留的 `bbt5-detect-baseline/dataset/`。
- Locked evidence：`artifacts/locked-bbt5-dataset/`，dataset hash `6e16c975941dbae2af174a2eb4b5424bffd4736c74aad56d424805da019b8fbc`。
- Clean initializer：`../original/weight/yolo11m.pt`，與 Ultralytics v8.3.0 release SHA-256 完全一致。
- `train` 只用於更新權重；`validation` 只用於 checkpoint、選模與 calibration。
- 現有 `test` 標為 historical，禁止訓練與選模；trainer YAML 實體移除 `test` key。
- `final_holdout` 尚未建立，因此目前禁止 unseen claim。

## 2. 嚴格公平組

以下 12 個模型全部使用相同 direct full-model 100 epochs、imgsz 640、batch 16、SGD、momentum 0.937、cosine LR、`lr0=0.001`、AMP、Ultralytics 8.4.90 與 seeds 42/43/44。

| ID | 實驗 | 唯一架構差異 |
|---:|---|---|
| C00 | B0-Clean | 原始 P3/P4/P5 三尺度 Detect |
| C01 | P2-Direct-Clean | 標準 P2/P3/P4/P5 四尺度 Detect |
| C02 | P2-PaperFormula-Clean | P2 全 channels，DW3/DW5/F7/F9，雙 1×1 paper fusion |
| C03 | P2-Lite35-Clean | P2 全 channels，DW3+DW5 |
| C04 | P2-Lite35-F7-Clean | P2 全 channels，DW3+DW5+factorized 7 |
| C05 | P2-Partial50-Clean | P2 前 50% channels 使用 DW3+DW5 |
| C06 | P2-Partial25-Clean | P2 前 25% channels 使用 DW3+DW5 |
| C07 | P3-M7-Clean | P3 DW3+DW5+factorized 7，舊單 1×1 fusion |
| C08 | P3-Lite35-Clean | P3 全 channels，DW3+DW5，雙 1×1 fusion |
| C09 | P3-Lite35-F7-Clean | P3 DW3+DW5+factorized 7，雙 1×1 fusion |
| C10 | P3-Partial50-Clean | P3 前 50% channels 使用 DW3+DW5 |
| C11 | P3-Partial25-Clean | P3 前 25% channels 使用 DW3+DW5 |

P3 全部禁止 9×9；只有忠實測試完整公式的 P2 C02 保留 F9。

## 3. 分開排名的最佳化控制

- C12-A `P2-Control-Clean-Head`：只訓練新增 P2 slot/head，20 epochs、`lr0=0.01`。
- C12-B `P2-Control-Clean-Full`：承接同 seed 的 A-best，全模型 80 epochs、`lr0=0.001`。
- 總預算同為 100 epochs，但 schedule 不同，因此只用來診斷 P2 最佳化，不混入 strict-fair 架構勝負。

## 4. Selective P2 延伸組

- C13 `SP2-Clean`：Ball-only hidden=32 P2 head，Bat 保持 P3/P4/P5。
- C14 `SP2P-Clean`：只使用 validation mean 選出的 P2 partial 配方。
- 只有主矩陣 selection freeze 後才建立 C14；不得看 historical test 決定 partial ratio。
- Selective 組同樣跑 seeds 42/43/44，但單獨報告，不混入 strict-fair 排名。

## 5. 執行順序

1. CPU：config/hash、12 模型 build、tensor transfer、forward/backward、stride、P3 無 F9、train/val-only YAML。
2. GPU acceptance：每個 unique graph 做 640 AMP forward/backward、optimizer step、peak memory 與 strict reload。
3. Smoke：12 個嚴格公平模型各用 seed 42 跑 3 epochs；只判斷工程穩定，不列結果。
4. Formal strict：12 模型 × 3 seeds，共 36 個 100-epoch runs。
5. P2 control：Head/Full × 3 seeds，共 6 個 stages。
6. 只用 validation 產生 mean±std、排名與 selection freeze。
7. Selective P2：若 gate 通過，再跑 SP2/SP2P × 3 seeds，共最多 6 runs。
8. Selection 全部凍結後，才允許產生 historical-test report。
9. 取得新影片／比賽／攝影機後，建立 final holdout 並只評估一次。

主流程為 12 smoke + 36 strict formal + 6 control stages = 54 個 GPU jobs；Selective 延伸最多再加 6 個。所有工作由單一 GPU queue 線性執行。

## 6. 統一指標

- COCO mAP50–95、mAP50、mAP75。
- AP_S、AP_M、AP_L。
- Ball/Bat AP、Ball AP_S、precision、recall、FP、missed。
- Params、GFLOPs、peak memory、FP16 batch-1 latency mean/P50/P95。
- 每個正式最終模型報告三 seeds mean ± std，另保留逐 seed 原值。
- 保存 initializer/config/dataset/checkpoint SHA-256、resolved profile、transfer report 與 lineage。

## 7. Selection 與宣稱規則

- 只用 validation mean 排名；同時考慮 overall、AP_S、Ball AP/recall、FP 與硬體成本。
- historical test 不可改變模型、threshold、seed 或選模結果。
- 沒有 final holdout 前，只能宣稱「clean initializer 下的 locked-BBT5 架構消融」，不能宣稱 unseen generalization。
- 舊 data-exposed 結果不回填、不作勝負基準。

## 8. 目前進度

- [x] 舊不準確結果、metrics、報告與約 1 GB 權重副本自目前版本移除。
- [x] `bbt5-detect-baseline/` 完整保留。
- [x] Locked dataset evidence 搬至中立路徑。
- [x] 官方 clean initializer provenance/hash 驗證。
- [x] Clean contract、builder、worker、資料可見性與 CPU feasibility。
- [ ] 擴充矩陣的最終 CPU inspect 與 GPU acceptance。
- [ ] 建立單一可續跑 GPU queue。
- [ ] Smoke、formal、selection、historical report。
- [ ] 新 final holdout。
