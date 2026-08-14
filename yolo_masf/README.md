# MASF-YOLO：BBT5 球棒／球偵測實驗

這個資料夾包含 BBT5 上的 YOLO11m、P2 detection head、P3-only MFAM、Partial MFAM 與 Selective P2 實驗。首頁只連到 GitHub 中實際存在的檔案；dataset 不隨 repo 發布。

> **結果口徑**：初始化權重 `yolo11m_bat_detect_init.pt` 已接觸 BBT5，因此目前結果是 data-exposed operational ablation，不是無洩漏泛化估計。全部正式結果只有單一 seed=42。

## 從這裡開始

| 想看什麼 | GitHub 入口 |
|---|---|
| Clean initializer、公平比較與資料可見規則 | [Clean 實驗設計與 CPU feasibility](configs/clean/README.md) |
| 每一個實驗在做什麼、為何要做、如何比較 | [完整實驗目錄](EXPERIMENT_CATALOG.md) |
| 第二輪完整做法、表格與分析 | [B1R / P2 / P3 完整實驗報告](b1r_p2_p3_study/results/REPORT.md) |
| 從資料檢查到 GPU queue、評估與發布的過程 | [完整實驗流程](b1r_p2_p3_study/EXPERIMENT_PROCESS.md) |
| 每一個模型自己的做法與精確指標 | [逐模型實驗索引](b1r_p2_p3_study/results/experiments/README.md) |
| 第二輪 CSV / JSON 總表 | [comparison.csv](b1r_p2_p3_study/results/comparison.csv) / [summary.json](b1r_p2_p3_study/results/summary.json) |
| 第二輪權重是否存在、原 SHA-256 | [Checkpoint 狀態](b1r_p2_p3_study/results/weights/CHECKPOINT_STATUS.md) |
| 第一輪 M0–M7、P3M、SP2/SP2P 報告 | [Phase 1 完整中文報告](EXPERIMENT_RESULTS_ZH.md) |
| 第一輪逐模型正式 artifacts 與權重 | [Static Phase 1 索引](artifacts/static-phase1/README.md) |
| 資料來源、split 與限制 | [資料說明](b1r_p2_p3_study/data/README.md) |
| MASF 論文與現有實作差異 | [論文實作稽核](docs/research/2026-08-11-masf-paper-implementation-audit.md) |

## 第二輪最重要結果

| 模型 | Test mAP50–95 | AP_S | AP_M | AP_L | Ball AP | Bat AP | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Original 3Scale | **0.770812** | 0.595169 | 0.879066 | 0.853723 | 0.678444 | 0.863179 | 67.646 |
| P2 Control-Full | 0.747240 | 0.562834 | 0.856095 | 0.831980 | 0.645597 | 0.848883 | 88.962 |
| P2 PaperFormula-Full | 0.743312 | 0.535394 | 0.868195 | 0.831479 | 0.637197 | 0.849428 | 91.072 |
| P3 PaperFormula-Full | 0.750617 | 0.557967 | 0.871743 | 0.827096 | 0.643119 | 0.858115 | 69.540 |
| P3 Partial25-35 | **0.754951** | **0.576821** | 0.877733 | 0.845235 | **0.651076** | **0.858826** | **67.779** |

結論是：在本次固定 split 與單一 seed 下，加入完整 P2 沒有自動提升結果；P3 Partial25-35 是新增模型中最接近 B0、且成本最低的折衷方案。但 B0 使用不同訓練歷史，只能當 operational reference，不能稱為公平消融勝者。

## 兩輪實驗如何區分

| 輪次 | 模型 | 目的 | 結果位置 |
|---|---|---|---|
| Phase 1 | B0/B1、M0–M3、M7、P3M、SP2/SP2P | 初版 P2 MFAM、P3-only 與 Selective P2 探索 | [報告](EXPERIMENT_RESULTS_ZH.md)、[artifacts](artifacts/static-phase1/README.md) |
| B1R/P2/P3 Retest | Direct、Control、P2/P3 各五種 MFAM | 修正 baseline、公平比較 placement 與 partial channels | [study](b1r_p2_p3_study/README.md)、[報告](b1r_p2_p3_study/results/REPORT.md) |

兩輪使用同一 BBT5 來源，但模型矩陣與訓練程序不同；不把兩輪數值混成同一個公平排名。

## 第二輪 checkpoint 狀態

- B0 initializer 已在 [`bbt5-detect-baseline/weights/`](bbt5-detect-baseline/weights/) 並由 Git LFS 管理。
- 第二輪 13 個 formal best 的原始路徑與 SHA-256 已保存，但整理發布時 runtime `.pt` 已不存在，因此無法上傳；詳見 [狀態表](b1r_p2_p3_study/results/weights/CHECKPOINT_STATUS.md)。
- 第一輪 Phase 1 的 native best/last 與 canonical checkpoint 已在 `artifacts/static-phase1/`，仍可下載。

## 資料夾導覽

```text
yolo_masf/
├── README.md                         # 本首頁
├── CONTEXT.md                        # 專案共同術語
├── MFAM_plan.md / codex_plan.md      # 原始研究規劃
├── EXPERIMENT_RESULTS_ZH.md          # 第一輪完整報告
├── bbt5-detect-baseline/             # 模型設定、來源權重；dataset 不發布
├── b1r_p2_p3_study/
│   ├── README.md                     # 第二輪研究入口
│   ├── EXPERIMENT_PROCESS.md         # 完整執行流程
│   ├── data/README.md                # split、hash、資料限制
│   └── results/
│       ├── REPORT.md                 # 第二輪完整報告
│       ├── experiments/<model>/      # 每模型 README、metrics、profile、lineage
│       ├── comparison.csv / summary.json
│       ├── profiles/                 # 機器可讀硬體成本
│       ├── metadata/                 # queue、request、worker、audit
│       └── weights/                  # checkpoint 狀態與 SHA-256
├── artifacts/static-phase1/          # 第一輪正式 artifacts 與 Git LFS 權重
├── configs/                          # 模型與訓練設定
├── masf_yolo/                        # 實作程式
├── scripts/                          # 發布包重建工具
└── tests/                            # 自動測試
```

## 重建第二輪發布包

Runtime artifacts 還存在時，可執行：

```bash
python scripts/build_retest_publication.py \
  --source artifacts/b1r-p2-p3-retest \
  --repo .
```

正式訓練、queue 與後處理命令見 [完整實驗流程](b1r_p2_p3_study/EXPERIMENT_PROCESS.md)。
