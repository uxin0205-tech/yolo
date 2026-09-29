# YOLO26m Detect–Pose Shared-Trunk

本 repository 實作 Full35 YOLO26m 的一份 shared backbone/neck 與兩個任務 head：

```text
image
  └─ shared Full35 YOLO26m layers 0–22
       ├─ COCO80 Detect head（應用層取 person）
       └─ Pose26 head（ball、bat，kpt_shape=[2,3]）
```

這不是在 forward 內跑兩個完整 YOLO，也不是平均權重。Ultralytics 原 graph executor、
skip connection、Concat 與 `from` indices 全部保留；只把最後 prediction module 換成
dual-head wrapper。初始化時兩個獨立模型合計 45,580,762 parameters，共享 graph 為
26,529,701，少 19,051,061（41.796%）。這是參數／weight storage 減少，不等同於
精度、latency、能耗或 FPGA resource 已改善。

## 目前完成狀態

- 完整 pre-head graph compatibility audit，以及 shared/detect-head/pose-head 逐 tensor
  loading report。
- `task=detect`、`task=pose`、`task=both`；`both` 只抽一次 shared features。
- current Ultralytics 8.4.90 原生 Detect／Pose26 E2E loss、progressive loss、RLE 與
  visibility mask，沒有自行簡化。
- 兩個 native dataloader、2 Detect : 1 Pose macro、依 actual batch size 正規化、三次
  sequential backward 後一次 optimizer step。
- J0 Pose-head適應、J1保守neck融合、J2後段backbone/MASF融合；J3保留為明確啟用的
  attention低學習率refinement。AdamW是正式主線，MuSGD是後續同設定challenger。
- Float training，Float 與 Bit-True 每 epoch 各自驗證；Bit-True 才是 checkpoint
  selection 與精度 gate 的正式口徑。
- exact token-tiled XNOR、shared BN policy、EMA、gradient norm/cosine、八項 hard gate、
  JSONL/CSV/PNG/TensorBoard capability、四種完整 resume checkpoint。
- Full35 真實 COCO/BBAT5 CPU macro與完整測試已通過；CPU驗收明確隱藏CUDA，不會占用
  正在執行的GPU工作。
- Partial75 有完全隔離且預設 `enabled: false` 的同型設定，未執行。

詳細架構見[正式設計](docs/design/2026-08-24-yolo26-joint-training-spec.md)，實際操作見
[Full35 runbook](docs/runbooks/2026-08-24-full35-joint-runbook.md)，本輪全部變更與驗證見
[工作紀錄](docs/worklogs/2026-08-24-full35-joint-implementation.md)。
新版保守重訓的原因、stage政策、測試與舊run保留方式見
[v2工作紀錄](docs/worklogs/2026-08-26-full35-joint-v2-retraining.md)。
J2完成後的獨立低學習率attention/full-backbone微調見
[J3 challenger工作紀錄](docs/worklogs/2026-08-27-full35-j3-challenger.md)。

## Final 交付

目前可執行的Full35交付包位於[final/full35](final/full35/README.md)。它封裝J2 accepted
權重、完整resume／inference權重、獨立回退權重、程式碼快照、Float／Bit-True AP輸出、
訓練logs與全檔案SHA256。J3仍是獨立challenger，完成前不會覆寫final；第一個J3
validation雖通過八項gate，但joint score `0.7096216737087447`尚未超過J2的
`0.7100382159275817`。

final不複製COCO或BBAT5資料本體，只保存registry／YAML snapshot；詳細封裝邊界、驗證、
困難與風險見[工作紀錄](docs/worklogs/2026-08-27-full35-final-package.md)。

## 唯一資料契約

新 BBAT5 實驗只使用 machine-readable registry：

```text
/home/uxin/yolo/configs/datasets/bbat5-v1.yaml
```

| route | 正式入口 |
| --- | --- |
| COCO80 Detect | `/home/uxin/yolo/coco2017.yaml` |
| ball/bat Pose | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml` |
| ball/bat Detect 診斷 | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml` |

BBAT5 共 6,647 張，formal train/val 為 5,964/683。runtime View 只保留既有 assignment
與 labels，讓 cache 留在 variant 的 `artifacts/`；舊 6,080/567 basic split 只用於歷史
checkpoint 稽核，不能成為新 run 輸入。

### 30% 診斷抽樣

前期 Pose 流程／loss／收斂方向檢查可明確傳入 `--fraction 0.3`。這不是 Ultralytics
原生的排序前綴抽樣，而是固定 seed、以 `.rf.` 前的 source group 為單位隨機抽樣；完整
683 張 canonical validation 仍全部使用。Full35 seed0 已固定為 696 groups、1,777 train
images、991 ball 與1,457 bat instances，manifest 位於
`variants/full35/artifacts/datasets/diagnostic/pose/f0p3-seed0/manifest.json`。

所有此模式輸出的 run 都寫入 `artifacts/pose/diagnostic/` 並帶有
`diagnostic-run.json`；它可產生 AP/mAP 供趨勢診斷，但不能作正式 P3、八項 baseline、
0.08 gate 或解除 JOINT preflight。正式 P1→P3 與 validation 一律使用100%。完整設計、
真實抽樣統計與驗證見[工作紀錄](docs/worklogs/2026-08-25-fraction-0p3-diagnostic-sampling.md)。

## 正式設定

Full35 的唯一 joint 設定是
[joint.yaml](variants/full35/configs/joint.yaml)：

- Detect logical train batch固定為128；空卡實測後，每個logical batch由physical
  microbatch64×2執行。Pose physical batch為16；驗證固定Detect32、Pose16。
- 每macro仍是2個Detect logical batch128，再1個Pose batch16；實際依序執行4×64與1×16
  forward/backward，只step一次。每次update的Detect exposure仍為256張。
- `reference_batch_size=64` 只控制 backward 尺度，不是 physical 或 effective batch。
- Joint stage的Detect/Pose task weights為1.0/0.25；J0只有Pose是active task，因此不會
  再被0.25額外縮小。
- Detect 與 Pose `mosaic=0`；Detect `fliplr=0.5`，Pose `fliplr=0`；Pose visibility 0/2
  原樣交給 native augmentation/loss。
- shared BN running statistics 固定；affine 預設可訓練且可由 YAML 關閉；head BN 正常
  train。
- XNOR 使用逐值相同的 `token_tile=32`；Q/K STE challenger 關閉。
- 獨立 Pose P1/P2/P3 的 patience 分別為10/12/20。
- 獨立 Pose stage實測設定為P1 physical128×accumulate1、P2 64×2、P3 32×4；
  `nbs=128`，三者logical images/update與effective weight decay維持一致。
- J0固定8 epochs且只訓練Pose head；J1最多20、patience8，只開neck與兩heads；
  J2最多80、patience17，才開backbone layers9+與MASF；J3最多20、patience5且必須
  明確`--enable-j3`，才開attention可微部分。
- J0/J1/J2 warmup各1 epoch；J3保留3 epochs。
- J2以 Bit-True joint score 判斷；連續8個 epoch 無至少1e-4改善時只降一次 LR ×0.5，
  momentum/beta保持不變。

COCO 與 BBAT5 train image ratio 約19.83:1；2:1 是 batch-call ratio，不是 image ratio。
以2×logical128／1×physical16計算，每個完整macro的Detect:Pose exposure是256:16，即16:1。

## 正式 preflight

執行：

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python variants/full35/joint.py preflight
```

目前已回傳 `ready=true`，兩項正式前置均已完成：

- canonical `bbat5-v1` 的 P1→P2→P3 Pose26 best checkpoint。
- 同一 Float/Bit-True validator 產生的八項獨立 baseline。

正式JOINT v2已完成J0 8、J1 20與J2 25 epochs；J2因patience17正常early stop，最佳
Bit-True joint score為`0.7100382159275817`，八項hard gate全部通過。

## GPU 空閒串接佇列

Full35 seed0已排定持久佇列。GPU 0必須沒有compute process、free memory至少30,000 MiB、
utilization不超過10%，並連續2次poll都成立才開始。順序不是只有短測，而是：30% P1
健全性檢查 → 正式P1→P2→P3 → Float/Bit-True獨立baseline → preflight → 舊版JOINT J1→J2
AdamW → best_joint Float/Bit-True final validation。任何OOM、NaN、缺檔或preflight失敗都會
停止，不會自動改batch或繞過gate。

狀態與log位於`variants/full35/artifacts/queue/full35-seed0-pipeline/`；完整設計與停止／查詢
方式見[佇列工作紀錄](docs/worklogs/2026-08-26-full35-seed0-gpu-queue.md)。

新版v2 service與狀態：

```bash
systemctl --user status yolo-full35-v2-seed0.service
cat variants/full35/artifacts/queue/full35-v2-seed0/queue-status.json
journalctl --user -u yolo-full35-v2-seed0.service -f
```

v2使用獨立run
`full35-joint-adamw-v2-j0e8-j1e20-j2e80-seed0`；舊run不會被覆寫。J3不在v2 queue
自動啟用，但已由使用者另行授權以`yolo-full35-v2-j3-b32-seed0.service`執行，結果仍須
和J2做八項gate及joint score決選。

2026-08-26執行快照：30%診斷已完成；正式 P1 原先在 epoch7 validation因原生
`train batch128 → val batch256`自動加倍而OOM。P1已用train128/val16完成17 epochs；
P2/P3 memory gate另證實需使用64×2／32×4來維持logical128，並由持久service接續其餘
pipeline。P3與八項baseline已完成；JOINT logical batch128的單次physical forward在空卡
實測OOM，因此保留logical128，但改由microbatch64×2執行；每macro兩個logical batch，
仍保留256張Detect exposure。修正、strict checkpoint驗證、測試與
目前指標見[P1續跑工作紀錄](docs/worklogs/2026-08-26-full35-p1-resume-and-val-batch-cap.md)；
JOINT 啟動、logical128／physical64×2 的區分、AMP overflow 復原與現行服務則見
[JOINT啟動工作紀錄](docs/worklogs/2026-08-26-full35-joint-startup-batch128-and-amp-recovery.md)。

## 下一步命令

完整命令與路徑在 runbook。順序為：

```bash
# 1. GPU 空閒時跑 Full35 P1/P2/P3
.venv/bin/python variants/full35/run.py pose --stage p1 --device 0 \
  --name p0-full35-p1-b128-e17-seed0

# 2. P3 完成後，同口徑產生獨立 baseline
.venv/bin/python variants/full35/baseline.py --device 0 \
  --pose-checkpoint PATH_TO_P3_BEST

# 3. 將 P3 checkpoint 與 formal-gate JSON 填入 joint.yaml，再檢查
.venv/bin/python variants/full35/joint.py preflight

# 4. 新版 J0→J1→J2（J3不自動啟用）
.venv/bin/python variants/full35/joint.py train --device 0 \
  --name full35-joint-adamw-v2-j0e8-j1e20-j2e80-seed0
```

## CPU 驗收

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python variants/full35/cpu_joint_smoke.py \
  --steps 2 --imgsz 64
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest -q
```

真實資料 smoke 證據在
[formal-joint-cpu-smoke.json](variants/full35/artifacts/formal-joint-cpu-smoke.json)。它只驗
graph/loss/gradient/BN/EMA contract，不是640精度結果。

## 工作紀錄

所有新變更、驗證、困難與風險記錄於 [docs/worklogs](docs/worklogs/README.md)。
