# masf_yolo 套件

`masf_yolo` 是 BBT5 Clean 公平實驗的實作核心。新研究入口集中在 `clean/`；其餘模組提供共用的資料稽核、模型、訓練、評估與 artifact 契約。

## 目錄

| 路徑 | 用途 |
|---|---|
| `clean/` | Clean initializer、固定矩陣、資料可見性、CPU inspect、單 job worker |
| `models/` | B0、P2、P3、MFAM、Partial MFAM、Selective P2 與權重轉移 |
| `data/` | 標註解析、group/hash leakage audit、固定 split 與 COCO export |
| `training/` | profile、preflight、trainer、checkpoint completion 與續跑 |
| `evaluation/` | COCO 指標、大小物件 AP、profiling、selection freeze |
| `artifacts/` | canonical checkpoint、hash、strict reload 與原子狀態管理 |
| `retest/` | 舊 data-exposed 第二輪流程的歷史實作；不可產生新公平結論 |

## Clean 資料流

```text
bbt5-detect-baseline/dataset（只讀、完整保留）
  → artifacts/locked-bbt5-dataset（固定 train/val/historical test）
  → 官方 COCO80 yolo11m.pt clean initializer
  → 12 個 strict-fair 架構 × 2 seeds
  → validation-only selection freeze
  → historical test 報告
  → 未來新 final holdout（只評估一次）
```

P3 family 一律不含 9×9/F9。完整 PaperFormula 的 F9 只存在於 P2 對照。所有 strict-fair 模型均使用相同 direct full-model 最多 100 epochs 與 `patience=30` early stopping；P2 head20+full80 只列為最佳化控制，不與 strict tier 混排。

## 主要入口

- 研究計畫：[`../codex_plan.md`](../codex_plan.md)
- Clean 設定：[`../configs/clean/README.md`](../configs/clean/README.md)
- Clean module：[`clean/README.md`](clean/README.md)
- 鎖定資料：[`../artifacts/locked-bbt5-dataset/README.md`](../artifacts/locked-bbt5-dataset/README.md)
- 測試：`../tests/clean/` 與 `../tests/models/`

目前尚未啟動 GPU，也沒有 Clean 正式數值。舊結果已從目前版本移除，但來源 BBT5 資料、舊 initializer 與工具完整保留。
