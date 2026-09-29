# YOLO26 Detect–Pose 融合實作計畫

- 日期：2026-08-22
- 狀態：P1正式完成；P2/P3依空卡memory gate使用64×2／32×4並已恢復GPU pipeline
- 主線：Full35-A2 Float；Partial75-A2 備案

## Module 與 seam

```text
Application / future main.py adapter
              │
              ▼
  RoutedDualModel or SharedDualHeadModel
              │
       ┌──────┴──────┐
       ▼             ▼
   detect output   pose output

SourceBundle adapter ── final bundle + Ultralytics 8.4.90
TaskLossRouter      ── Detect/Pose criteria + missing-task masking
Checkpoint module  ── state_dict + provenance; no pickled combined model
VariantWorkspace   ── Full35/Partial75 sources, evidence, configs, isolated runs
```

外部 interface 保持小：tensor 加 `tasks`，回傳 task-keyed outputs。來源 bundle 的絕對路徑、pickle module resolution、attention conversion、MASF graft 與 tensor transfer 全部藏在 SourceBundle adapter 的 implementation。

## 檔案配置

```text
src/yolo_combine/
├── contracts.py    # task vocabulary and output contract
├── source.py       # bundle verification, loading, Pose graft, tensor transfer
├── models.py       # RoutedDualModel and SharedDualHeadModel
├── training.py     # task-aware loss routing and 1:1/2:1 loss composition
├── loaders.py      # task schema checks and separate COCO/BBT5 loaders
├── joint.py        # explicit step-based development orchestration
├── freezing.py     # inherited attention/MASF freeze and drift guard
├── materialize.py  # official Detect/Pose graphs for validators/export
├── baselines.py    # materialized Full35 Pose26 baseline trainer
├── checkpoint.py   # state_dict-only save/load and provenance
├── data.py         # BBT5 lineage audit + workspace-local BBT5/COCO views
├── variants.py     # fail-closed isolated variant workspace adapter
├── variant_cli.py  # folder-locked audit/cpu-check/joint/Pose interface
└── cpu_validation.py # Float/Bit-True CPU acceptance

configs/data/       # local-path adapters；source datasets remain immutable
variants/           # Full35/Partial75 folder-locked experiments and artifacts
tests/              # tests only through the modules' interfaces
```

## 實作進度

- [x] 建立 package、設定與 SourceBundle manifest validator。
- [x] 建立 Full35 Pose26 factory，要求 layers 0–22 完整 tensor mapping。
- [x] 實作 F0.5 `RoutedDualModel` 與統一 task interface。
- [x] 實作 F1 `SharedDualHeadModel`；P3/P4/P5 只抽取一次。
- [x] 實作 task-aware `TaskLossRouter` 與 1:1/2:1 composition。
- [x] 實作 state-dict-only checkpoint 與 source/checkpoint provenance。
- [x] 加入 interface、真實 checkpoint、GPU forward/backward、freeze 與 reload tests。
- [x] 建立 basic BBT5 read-only derived view，修正 4 個極小負 keypoint 座標。
- [x] 跑 P0 trainer smoke、F1 1:1、F1 2:1 與 P0-head 初始化 smoke。
- [x] 實作 F1 → 官方 Detect/Pose graph materialization，逐值輸出一致。
- [x] 跑過舊 Full35 P1（batch16、head-only 10 epochs）；只保留為 provisional 歷史 run。
- [x] 建立 Full35/Partial75 對稱工作區與原始 BBT5 Pose／COCO Detect 不融合報告。
- [x] 將兩工作區的 cache、report、checkpoint 與 run root 完全移入各自資料夾。
- [x] 完成兩架構 Float/Bit-True independent→shared CPU 逐值驗證。
- [x] 完成兩架構 synthetic 與 workspace-local 真實 COCO/BBT5 2:1 CPU step。
- [ ] 取得授權後復原被第一次 direct smoke 重建的 COCO source cache。
- [x] 完成正式 Full35 P1（physical128、head-only 17 epochs）。
- [ ] 完成正式 Full35 P2（physical64、accumulate2、neck+head 22 epochs）。
- [ ] 完成正式 Full35 P3（physical32、accumulate4、最多100 epochs、patience20）。
- [ ] 鎖定正式 F1 loop 的 AMP、EMA、scheduler、定期雙任務 validation。
- [ ] 跑 F1 1:1 與 2:1 正式比較。
- [ ] 視時間與候選結果決定第二 seed。

## 已驗證的工程證據

- bundle manifest：121/121。
- Detect → Pose shared layers：587/587 tensors。
- P0 Pose26 head → F1 head：411/411 tensors。
- F1 shared state → 官方 Pose：998 tensors。
- 雙模型參數：45,580,762。
- F1 參數：26,529,701（約 -41.8%）。
- Partial75 雙模型/F1：45,442,522 / 26,460,581（約 -41.8%）。
- 自動測試：38 collected；本輪 CUDA 隱藏回歸 35 passed、3 GPU tests skipped。
- 真實資料 smoke：COCO + BBT5 的 1:1、2:1 均完成 loss、backward、optimizer step。
- 本輪 folder-locked smoke：Full35/Partial75 的 Float、Bit-True 與 2:1 均通過；
  詳見 [CPU validation audit](../audits/2026-08-22-cpu-variant-validation.md)。
- 舊 P1-b16-e10 seed 0：box mAP50-95 0.521、pose mAP50-95 0.816；
  ball/bat pose 0.796/0.836，僅為 provisional 歷史證據。
- P0/P1 `best.pt` SHA256：
  `bf8f1f902fed64231447718c6eca3f2bd659ae9bdf56f4cd6acf934e918b280b`。
- P0/P1 Pose head 嚴格轉入 F1：411/411 tensors；P1 → P2 transition smoke 通過。

## 第一階段完成條件

- bundle manifest 與 model graph fail closed。
- Full35 Detect/Pose forward 可重現。
- F0.5/F1 使用相同 task-keyed interface。
- F1 trunk parameters只有一份，P3/P4/P5 只抽取一次。
- Detect/Pose loss可個別 backward，未知標註不成為另一任務背景。
- checkpoint 可 save、rebuild、reload，且不依賴 pickle combined model。
- 自動測試與 smoke 指令記錄於 README。

## 第二階段實驗順序

1. D0：既有 Full35-A2 Detect baseline 重驗。
2. P0：依128×1、64×2、32×4的logical128 staged profile訓練 Full35 Pose26。
3. F0.5：D0 + P0 routed bundle。
4. F1-80 R1：full COCO Detect : BBT5 Pose = 1:1。
5. F1-80 R2：2:1，Detect microbatch loss先平均。
6. 依逐項 `baseline - 0.08` gate選擇，必要時才啟動 task-specific BN/partial neck/adapters。
7. F1-80 合格後才建立 person-only F1-1 與後續量化實驗。

## 執行界線

- Full35 是唯一主線；Partial75 不與主線平行耗用 GPU，只有 Full35 gate 失敗或效率
  需求改變時才啟動。
- basic BBT5 split 可用於目前開發與第一輪 provisional baseline，但其 10 個
  source-group overlap 必須寫入報告。
- 正式長訓練開始前固定 run name、完整超參數與 seed，禁止覆寫既有 artifacts。
- 開發先用 seed 0；只有入選者才視時程跑第二 seed。
- 正式與 smoke 命令優先使用 `variants/<architecture>/run.py`，不得以自由
  `--architecture`/`--project` 組合交叉寫入。
- 來源 COCO cache 復原前，不以 full source YAML 執行 Ultralytics fraction subset。
