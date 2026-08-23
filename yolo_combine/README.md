# YOLO26 Detect–Pose Fusion

YOLO26m Detect–Pose 研究工程：一次抽取 Backbone + Neck 特徵，再分別送入
COCO80 Detect head 與 ball/bat Pose26 head。

## 已固定的架構決策

- 主線：**Full35-A2 Float**。
- 備案：Partial75-A2；不與 Full35 混用，也不自動切換。
- Detect：第一版保留 COCO80 head；應用層只消費 person。
- Pose：ball/bat，`kpt_shape=[2,3]`。
- F1：一份真正共享的 layers 0–22，加兩個獨立 head。
- 共享是建立新的 F1 graph 並移除重複 trunk，不是平均、剪枝或覆寫原始 Full35；
  「逐值相同」只在兩個 task trunk 權重尚一致、且尚未 joint training 時成立。詳見
  [ADR 0001](docs/adr/0001-share-full35-trunk-with-independent-heads.md)。
- 精度 gate：person box、ball box、bat box、ball pose、bat pose 各自不得比對應
  獨立 baseline 低超過 0.08 mAP50-95。

權威來源維持唯讀：

```text
/home/uxin/yolo/yolo_achitechure/achitechure_1/final
```

## 唯一 BBAT5 資料契約

所有新實驗的 BBAT5 部分只讀取：

```text
/home/uxin/yolo/configs/datasets/bbat5-v1.yaml
```

registry 是同時管理 Pose 與二類 Detect Task View 的版本入口，不會取代 COCO80：

| 模型 route | 資料入口 |
| --- | --- |
| COCO80 Detect 主線 | `/home/uxin/yolo/coco2017.yaml` |
| BBAT5 Pose 主線 | registry 的 `tasks.pose` |
| BBAT5 ball/bat Detect 診斷 | registry 的 `tasks.detect_2class` |

因此融合 run 會記錄 COCO 與 BBAT5 兩種資料契約；不能把 `detect_2class` YAML 當成
COCO80 head 的訓練資料。

registry 鎖定 6,647 張影像、formal train/val 5,964/683、零同源群組 leakage、四筆
座標修補，以及 task YAML／lineage manifest 的 SHA256。各 workspace 只建立
`artifacts/cache-views/bbat5-v1/` symlink-only View 來隔離 cache，不再複製 labels 或建立資料入口。
舊 `bbt5_pose_basic` 的 6,080/567 split 有 10 個 overlap groups，只用來解釋既有
checkpoint；完整核對見[資料統一紀錄](docs/data/2026-08-22-bbat5-unification.md)。

## 已完成

- `SourceBundle` 驗證 121-file manifest 並載入 Full35/Partial75。
- Full35 Detect → Pose26 layers 0–22 完整移植：587/587 tensors。
- P0 checkpoint → F1 Pose head 嚴格移植：411/411 tensors。
- `RoutedDualModel`：兩個完整模型的統一 task interface。
- `SharedDualHeadModel`：一份共享 trunk，Detect/Pose 兩個 head。
- 共享模型參數 26,529,701；雙完整模型合計 45,580,762，減少約 41.8%。
- task-aware loss routing；未標註的另一任務不會被當成背景。
- 1:1 與 2:1 joint step；2:1 的兩個 Detect loss 先平均。
- inherited attention/PWL 與 Full35 MASF 凍結、eval-mode 與 drift guard。
- state-dict-only checkpoint，以及轉回官方 Detect/Pose graph 的 validation/export seam。
- 建立 Full35/Partial75 完全隔離的 folder-locked 工作區；設定、cache、report、
  checkpoint 與 run outputs 不會交叉。
- Full35/Partial75 Float 與 Bit-True 的獨立模型 → 共享模型 CPU forward 均逐值相同。
- 兩架構的 synthetic 及真實 COCO/BBT5 2:1 CPU optimizer step 均通過。
- 現有 41 項測試；本輪 CUDA 隱藏回歸為 38 passed、3 GPU tests skipped。

## 目前正式進度

- 舊 Full35 P1（seed 0、batch 16、10 epochs）已完成，但只保留為 provisional 歷史 run。
- 最佳完整驗證：box mAP50-95 0.521；pose mAP50-95 0.816。
- 分類 pose mAP50-95：ball 0.796、bat 0.836。
- P1 `best.pt` 已通過 schema 檢查與 411/411 Pose-head tensor transfer。
- 舊 P1 → P2 transition smoke 已通過；先前 batch16 P2 嘗試 OOM，沒有接受其 checkpoint。
- 新正式排程已鎖定 batch128、P1 17、P2 22、P3 最多100；等待共用 GPU 空閒。

這些既有數字仍是 provisional，因為產生它們的舊 run 使用有已知 source-group overlap
的 BBT5 basic split；它們不能冒充後續 `bbat5-v1` 正式結果。所有新 run 已切換到
leakage-safe `bbat5-v1`，歷史數字不回溯改寫。

### P1 batch 與 epoch 的精確口徑

P1 確實重新訓練 10 epochs，但 layers 0–22 全凍結，只更新 fresh Pose26 head。
每個 forward 的 physical batch 是 16；`nbs=64` 使穩態 gradient accumulation 約為
4，因此每次 optimizer update 約涵蓋 64 張。官方 YOLO26m COCO pretraining recipe
列出的 batch 128、80 epochs 是上游參考，不是這次 ball/bat P1 已重現的設定。
正式替代 run 將直接使用 batch128：P1、P2 分別固定跑滿 17、22 epochs；P3 最多
100 epochs，patience 20。舊 batch16 P1 不會被覆寫或冒充新結果。

兩個 workspace 的介面、輸出與 CPU 報告已完全分開，詳見
[variants](variants/README.md)。CPU 驗證期間發現的 COCO cache 來源副作用與待授權
復原事項記錄於 [CPU validation audit](docs/audits/2026-08-22-cpu-variant-validation.md)。

## Environment

專案已有依照權威 bundle 建立的 `.venv`。若需重建：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r /home/uxin/yolo/yolo_achitechure/achitechure_1/final/requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

## 回歸測試

```bash
.venv/bin/python -m pytest
```

本輪以 `CUDA_VISIBLE_DEVICES=""` 執行：`38 passed, 3 skipped`；三項 GPU test
刻意略過，沒有干擾正在執行的 GPU 任務。

## Folder-locked CPU acceptance

每個入口從所在資料夾鎖定架構，且把 view/cache/report 寫回自己的 `artifacts/`：

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/full35/run.py audit --verify-hashes
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/full35/run.py cpu-check
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/full35/run.py joint-smoke --ratio 2:1

CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/partial75/run.py audit --verify-hashes
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/partial75/run.py cpu-check
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/partial75/run.py joint-smoke --ratio 2:1
```

需要測試已訓練的 Pose26 head 時：

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python variants/full35/run.py joint-smoke \
  --ratio 2:1 \
  --pose-head-checkpoint variants/full35/artifacts/pose/NAME/weights/best.pt
```

CPU smoke 不是正式精度結果。正式比較仍須使用 640 輸入、完整 `bbat5-v1`、完整
validation 與獨立 baseline；只有歷史 basic split 結果維持 provisional。

## Pose baseline stages

正式 profile 固定為 batch128：head-only 17 → neck+head 22 → full 最多100。
P1/P2 關閉 early stopping、必須跑滿；P3 patience 20。P2/P3 都從前一階段
`best.pt` 建立新的 optimizer，不沿用舊 optimizer state。

```bash
# P1：固定 17 epochs，只訓練 Pose26 head
.venv/bin/python variants/full35/run.py pose \
  --stage p1 --batch 128 --epochs 17 \
  --name p0-full35-p1-b128-e17-seed0

# P2：固定 22 epochs，解凍 neck + head
.venv/bin/python variants/full35/run.py pose \
  --stage p2 --batch 128 --epochs 22 \
  --name p0-full35-p2-b128-e22-seed0 \
  --initial-checkpoint variants/full35/artifacts/pose/p0-full35-p1-b128-e17-seed0/weights/best.pt

# P3：最多 100 epochs，patience 20
.venv/bin/python variants/full35/run.py pose \
  --stage p3 --batch 128 --epochs 100 \
  --name p0-full35-p3-b128-e100max-seed0 \
  --initial-checkpoint variants/full35/artifacts/pose/p0-full35-p2-b128-e22-seed0/weights/best.pt
```

CLI 會保存 registry、dataset ID、runtime View manifest、resolved training 參數與輸入
checkpoint SHA256。正式 run 使用 `bbat5-v1`；profile smoke 的 override 結果仍不能當成
正式 baseline。

## 下一個實驗

依 `bbat5-v1` 從頭執行 Full35 P1 17 → P2 22 → P3 最多100，再以最佳 Pose head初始化
F1，比較 1:1 與 2:1。五個逐項 gate 都通過後，才進入第二 seed、Partial75 或量化。

See [the implementation plan](docs/plans/2026-08-22-yolo26-fusion-implementation.md) and [source acceptance audit](docs/audits/2026-08-22-source-acceptance.md).
