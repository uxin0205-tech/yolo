# MASF-YOLO 執行計畫與完成狀態

更新日期：2026-08-10

## 範圍與固定輸入

主要實作位於 `masf_yolo/`。資料固定使用 `bbt5-detect-baseline/dataset/`，初始化固定使用 `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`（SHA-256：`9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d`）。該權重已看過 BBT5，所有結果均標示 `data_exposed=true`。

Phase 1 不包含 S0、Temporal Motion，也不包含正式 ONNX 匯出工作。ONNX 僅有早期 CPU 技術驗證，不列入本次 34-stage pipeline 與交付。

## 已完成項目

- [x] 資料稽核與 leakage-aware 固定切分：2,578 unique frames；train/val/test 1,987/300/291；group/hash overlap 為空。
- [x] B0 來源權重 hash、task、classes、stride 與 CPU forward 驗證。
- [x] B1：P2 baseline，A 階段凍結 backbone 0–10 共 10 epochs，B 階段全解凍 90 epochs。
- [x] M7：P2 的 DW3×3 + DW5×5 + DW(1×7→7×1)。
- [x] M0/M1/M2/M3：完整、輕量與 1/2、1/4 partial-channel 消融。
- [x] P3M：P2 Identity，只在 P3 使用 DW3、DW5、factorized DW7，沒有 9×9。
- [x] SP2：Ball-only hidden=32 輕量 P2 prediction head；P3–P5 保留 Ball/Bat。
- [x] SP2P：SP2 加 validation 選出的 M2，採 A/B 10+90 訓練。
- [x] 十個模型的 validation、test、faster-coco-eval 指標與 RTX 5090 FP16 profile。
- [x] M2/M3 只用 validation 選模，`selection.json` 在 test 前凍結。
- [x] 34 個 pipeline stages 全部 completed；`final_audit.json` 為 `ok=true`、errors 空。
- [x] 中文結果與分析：`EXPERIMENT_RESULTS_ZH.md`。
- [x] Repo 分層中文 README 與 GitHub Issues/domain 文件入口。
- [x] 白名單清理 smoke/preflight/gate checkpoint、cache 與預覽圖；正式 checkpoint 經 SHA-256 前後驗證保留，細節見 `artifacts/static-phase1/cleanup_manifest.json`。

## 已執行實驗矩陣

| ID | 結構／訓練 | 正式權重 |
|---|---|---|
| B0 | 指定三尺度來源權重直接評估 | 原始 initializer（不重訓） |
| B1 | P2；freeze 10 + unfreeze 90 | `training/b1_b/canonical.pt` |
| M7 | P2 3/5/factorized-7 | `training/formal_m7/canonical.pt` |
| M0 | P2 3/5/7/9 | `training/formal_m0/canonical.pt` |
| M1 | P2 3/5 | `training/formal_m1/canonical.pt` |
| M2 | 1/2 channels 使用 M1 | `training/formal_m2/canonical.pt` |
| M3 | 1/4 channels 使用 M1 | `training/formal_m3/canonical.pt` |
| P3M | 只在 P3 使用 3/5/factorized-7 | `training/formal_p3m/canonical.pt` |
| SP2 | Ball-only P2；freeze 10 + unfreeze 90 | `training/sp2_b/canonical.pt` |
| SP2P | SP2 + M2；freeze 10 + unfreeze 90 | `training/sp2p_b/canonical.pt` |

以上相對路徑的根目錄皆為 `artifacts/static-phase1/`。Native best/last 在 `runs/<run>/weights/`，全數保留。

## 結果摘要

公平 test mAP50–95：M7 0.7386、P3M 0.7340、B1 0.7303；其餘 M0/M1/M2/M3 皆未超過 B1。SP2 0.7170、SP2P 0.7213，雖減少理論 GFLOPs，但實測 latency 增加且 Ball false positives 大幅升高。完整 per-class、Precision/Recall、FP/FN 與硬體表見 `EXPERIMENT_RESULTS_ZH.md`。

## 後續工作（需另開 GitHub Issue）

- [ ] B1、M7、P3M 至少重跑 3 seeds，報告平均值與變異。
- [ ] 使用未接觸 BBT5 的乾淨 initializer，建立 leak-free 泛化實驗。
- [ ] 對 M7/P3M 做固定 confidence calibration 與 false-positive 類型分析。
- [ ] 若續做 SP2，先降低 auxiliary loss、加入 P2 hard negatives 與 Ball confidence gate；在誤報及實測 latency 改善前不擴增結構。
- [ ] S0 與 Temporal Motion 留待下一階段；不得直接混入本次 immutable artifacts。

新的工作項目與驗收條件依 `docs/agents/issue-tracker.md` 建立 GitHub Issue；domain 名詞與架構決策依 `CONTEXT.md`、`docs/adr/` 維護。
