# Repository 文件整理與實驗產物清理設計

日期：2026-08-10

## 目標

將 `yolo_masf` 整理成可由根目錄快速理解、可由各重要子目錄就地查詢用途的研究 repo，同時刪除已完成 Phase 1 後不再需要的 smoke、preflight、gate checkpoint 與可重建 cache。清理不得修改或刪除任何正式訓練權重、正式評估、資料集、來源權重或最終研究證據。

## 已確認決策

1. 使用分層 README，不在每個生成型 leaf directory 重複放置文件。
2. 正式 `best.pt`、`last.pt`、canonical checkpoints 全部保留。
3. smoke、preflight 與 M7 acceptance gate 的 checkpoints 可以刪除。
4. val/test metrics、predictions、false-positive 圖表、profiles、selection、final audit 與報告全部保留。
5. 清理前產生 dry-run；實際刪除留下 `cleanup_manifest.json`。
6. 不啟動 GPU、不重新訓練。
7. `AGENTS.md` 只保留一份不重複的 Agent skills 設定。

## 文件架構

### 根目錄入口

重寫 `README.md`，使其成為 repo 的唯一總入口，至少包含：

- 研究目的與 Phase 1 範圍。
- B0、B1、M7、M0、M1、M2、M3、P3M、SP2、SP2P 的架構差異。
- BBT5 dataset 與 `yolo11m_bat_detect_init.pt` 的來源、類別、切分與 data-exposure 警告。
- B1、SP2、SP2P 的 A/B 訓練方式，以及其他變體的共同正式預算。
- pipeline 執行方式、stage 順序與 GPU 排程原則。
- 最終結果摘要，以及 `EXPERIMENT_RESULTS_ZH.md`、原始 metrics、weights、profiles、selection、final audit 的路徑。
- 清理後哪些 artifacts 仍可直接使用，哪些 acceptance artifacts 需要重建。
- 完整但不展開每張影像與每個 checkpoint 的資料夾樹。

### 重要子目錄 README

建立或整理以下文件：

- `configs/README.md`：正式設定、模型 YAML、鎖定 hash 與各 config 用途。
- `bbt5-detect-baseline/README.md`：來源 dataset、weights、準備腳本及唯讀原則。
- `masf_yolo/README.md`：Python package 總覽；取代大小寫不標準的 `masf_yolo/Readme.md`。
- `masf_yolo/artifacts/README.md`：checkpoint、原子 I/O、state 與 strict reload。
- `masf_yolo/data/README.md`：資料稽核、group split、COCO export。
- `masf_yolo/evaluation/README.md`：COCO metrics、選模、profiling、false positives。
- `masf_yolo/models/README.md`：模型建構、MFAM、Selective P2、權重轉移。
- `masf_yolo/training/README.md`：profiles、preflight、worker、resume 與 completion。
- `tests/README.md`：測試分層、CPU/GPU 邊界與執行方式。
- `artifacts/README.md`：所有 runtime artifact 類別與保留政策。
- `artifacts/static-phase1/README.md`：Phase 1 產物地圖與模型到 weights/metrics/profile 的對照。
- `docs/README.md`：agent、spec、plan、ADR 文件用途。
- `field_check/README.md`：獨立 receptive-field/context 研究及其與正式 pipeline 的界線。

生成型目錄如 `runs/m2/weights/`、`evaluation/test/m2/` 不放 README；由最近的上層 README 統一解釋。

### Agent 設定

`AGENTS.md` 只包含一個 `## Agent skills` block，指向：

- `docs/agents/issue-tracker.md`：GitHub Issues，repo 為 `uxin0205-tech/yolo`。
- `docs/agents/domain.md`：single-context，未來按需建立 `CONTEXT.md` 與 `docs/adr/`。

未安裝 triage skill，因此不建立 triage label 文件。

## 清理設計

### 可刪除白名單

清理工具只允許刪除下列 workspace-relative patterns：

```text
.pytest_cache/**
**/__pycache__/**
**/*.pyc
**/*.pyo
artifacts/static-phase1/smoke_runs/*/weights/*.pt
artifacts/static-phase1/training/smoke_*/canonical.pt
artifacts/static-phase1/preflight/*.pt
artifacts/static-phase1/m7_gate/m7.pt
artifacts/static-phase1/runs/*/train_batch*.jpg
artifacts/static-phase1/runs/*/labels.jpg
artifacts/static-phase1/smoke_runs/*/train_batch*.jpg
artifacts/static-phase1/smoke_runs/*/labels.jpg
yolo26n.pt
```

`yolo26n.pt` 只有在全 repo reference scan 為零時才加入實際刪除清單。空的 cache/weights directory 可在其內容清空後移除；不得以寬泛 glob 或遞迴根目錄刪除取代逐項 resolved path。

### 永久保留清單

- `bbt5-detect-baseline/dataset/**`
- `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`
- `artifacts/static-phase1/runs/*/weights/best.pt`
- `artifacts/static-phase1/runs/*/weights/last.pt`
- `artifacts/static-phase1/training/{b1_a,b1_b,formal_*,sp2_a,sp2_b,sp2p_a,sp2p_b}/canonical.pt`
- 正式 `run.json`、`worker_request.json`、`worker_result.json`、`results.csv`、`results.png`。
- `artifacts/static-phase1/evaluation/**`
- `artifacts/static-phase1/profiles/**`
- `artifacts/static-phase1/dataset/**`
- `artifacts/static-phase1/references/**`
- `artifacts/static-phase1/stages/**`、`state.json`、`pipeline.json`、`selection.json`、`final_audit.json`、`report.md`。
- `artifacts/official_baseline_cpu/**`、`artifacts/official_pose_cpu/**`、`artifacts/official_weight_metrics/**`。
- `field_check/**`。

正式 weights 在清理前後都必須有相同 SHA-256。任何白名單與永久保留清單衝突時，以永久保留為優先並中止清理。

### 清理工具

新增 repository-owned cleanup utility，具備：

- 預設 dry-run；只有明確 `--apply` 才刪除。
- 確認 `masf-yolo-phase1.service` 不是 active。
- 確認 `stages/report.json` 為 completed 且 `final_audit.json` 為 `ok: true`、`errors: []`。
- 將每個 resolved target 限制在 repo root 內，拒絕 symlink escape。
- 在刪除前計算路徑、類別、原因、byte size 與 SHA-256。
- dry-run 顯示分類與總容量，不寫 artifact。
- apply 先原子寫入 `artifacts/static-phase1/cleanup_manifest.json`，再逐檔刪除。
- manifest 記錄執行時間、pipeline ID、保留 checkpoint hashes、刪除項目與總節省 bytes。
- 若刪除中途失敗，manifest 保留每個項目的 planned/deleted 狀態，讓操作者能判斷實際範圍。

清理後，正式 native `last.pt` 仍可直接用於訓練續跑；但 smoke/preflight/gate checkpoints 不再是完整封存的一部分，需要時必須重新產生。README 與 manifest 必須明確揭露此限制。

## 驗證與失敗處理

### 自動測試

以 temporary directory 建立代表性的 artifact tree，驗證：

1. dry-run 不刪除或寫入任何檔案。
2. apply 只刪除白名單項目。
3. 正式 best/last/canonical、metrics 與 reports 保持存在且 hash 不變。
4. active service、未完成 report、final audit failure、symlink escape、保留清單衝突都會 fail closed。
5. manifest 路徑、大小、hash、原因與狀態正確。
6. 未被引用的 `yolo26n.pt` 可刪；一旦有 repo reference 就保留。

### Repo 驗證

實作後依序執行：

- cleanup dry-run 並人工核對清單。
- 記錄正式 checkpoint SHA-256。
- 執行 cleanup apply。
- 重新計算正式 checkpoint SHA-256 並比較。
- 驗證所有 README 路徑與 Markdown relative links。
- 執行完整 `pytest` 與 `compileall`。
- 驗證 pipeline 最終狀態仍為 report completed，final audit 原始結果仍為 PASS。
- 回報清理前後容量、實際刪除項數與節省容量。

任何正式 checkpoint、evaluation、profile、selection 或 report 缺失都視為失敗；停止後續刪除，不以重新訓練掩蓋問題。

## 完成標準

- 根 README 能回答「這是什麼、資料在哪、怎麼訓練、怎麼評估、結果在哪、哪個模型最好、有哪些限制」。
- 每個確認的重要目錄都有就地 README，且不在生成型 leaf 重複文件。
- `AGENTS.md` 不重複，GitHub Issues 與 single-context 規則可被工程 skills 找到。
- cleanup manifest 可重現實際刪除範圍。
- 所有正式實驗權重與正式研究證據完整且 hash 不變。
- smoke/preflight/gate checkpoints、cache 與指定預覽檔已移除。
- CPU test suite、compileall、link check 與 preserved-artifact check 全部通過。
