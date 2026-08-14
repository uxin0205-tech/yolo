# MASF-YOLO：BBT5 Clean 公平實驗

本 repo 現在只把官方 Ultralytics COCO80 `yolo11m.pt` 初始化的 Clean 實驗視為有效研究線。先前使用已接觸 BBT5 initializer 的訓練結果、權重副本、metrics 與報告已從目前版本移除，不再提供數值結論。

> `bbt5-detect-baseline/` 依使用者要求完整保留。它是資料、舊 initializer 與轉換工具的來源資料夾；Clean 正式實驗不使用其中 data-exposed initializer。

## 目前狀態

- 官方 clean initializer 已與 Ultralytics v8.3.0 release 比對，SHA-256 為 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`。
- Locked BBT5 dataset audit 已搬到 [`artifacts/locked-bbt5-dataset/`](artifacts/locked-bbt5-dataset/)。
- Clean 模型、transfer、資料可見性與 multi-seed plan 已通過 CPU 驗證。
- 尚未啟動 GPU、尚未產生任何 Clean 正式結果。
- 現有 test 已被過去研究反覆查看，只能作 historical report；真正 unseen claim 仍需新 final holdout。

## 從這裡開始

| 內容 | 入口 |
|---|---|
| 完整實驗矩陣、順序與驗收 | [`codex_plan.md`](codex_plan.md) |
| Clean 設定與資料可見規則 | [`configs/clean/README.md`](configs/clean/README.md) |
| CPU feasibility 證據 | [`configs/clean/FEASIBILITY.json`](configs/clean/FEASIBILITY.json) |
| Locked split 與 leakage audit | [`artifacts/locked-bbt5-dataset/README.md`](artifacts/locked-bbt5-dataset/README.md) |
| BBT5 baseline 資料夾 | [`bbt5-detect-baseline/README.md`](bbt5-detect-baseline/README.md) |
| Clean module 結構 | [`masf_yolo/clean/README.md`](masf_yolo/clean/README.md) |
| GPU 啟動前清單 | [`GPU_TEST_PLAN.md`](GPU_TEST_PLAN.md) |

## 主要結構

```text
yolo_masf/
├── bbt5-detect-baseline/          # 完整保留的 BBT5 detection view 與舊權重
├── artifacts/locked-bbt5-dataset # 固定 split、COCO JSON、manifest、audit
├── configs/clean/                 # Clean 唯一有效正式設定與 feasibility
├── masf_yolo/clean/               # Clean builder、contract、plan、worker
├── masf_yolo/models/              # P2/P3/MFAM/SP2 架構實作
├── clean_bbt5_study/              # 新實驗的文件與未來結果入口
└── tests/                         # CPU 契約與模型測試
```

舊結果仍可從 Git 歷史恢復，但不得重新放回首頁當成公平或 unseen 結論。
