# Full35 JOINT 執行手冊

## 已排定的持久佇列

目前seed0完整流程由`yolo-full35-seed0-pipeline.service`等待GPU空閒後執行。查詢：

```bash
systemctl --user status yolo-full35-seed0-pipeline.service
cat variants/full35/artifacts/queue/full35-seed0-pipeline/queue-status.json
cat variants/full35/artifacts/queue/full35-seed0-pipeline/pipeline-status.json
tail -f variants/full35/artifacts/queue/full35-seed0-pipeline/pipeline.log
```

它會串接本手冊第1至第4節，再對`best_joint.pt`做Float＋Bit-True final validation。任何
stage失敗即停止；不自動降batch、不自動啟用J3/MuSGD/seed1/Partial75。

## 0. CPU／唯讀檢查

```bash
cd /home/uxin/yolo/yolo_combine
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python variants/full35/run.py audit --verify-hashes
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python variants/full35/joint.py preflight
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python variants/full35/cpu_joint_smoke.py --steps 2 --imgsz 64
```

目前 preflight exit 1 是預期結果，直到下列 P3 checkpoint與 baseline完成。

### 可選：30% 非正式 Pose 診斷

GPU 空閒後，可先用固定 seed 的 canonical train subset 驗證資料、loss、checkpoint與短期
收斂方向：

```bash
# 建議先跑2 epochs；仍使用指定 physical batch，故不等於VRAM memory gate
.venv/bin/python variants/full35/run.py pose --stage p1 --device 0 \
  --fraction 0.3 --epochs 2
```

此命令不使用 Ultralytics 的 sorted-prefix `fraction`。repository 先將 canonical train 的
2,320個source groups用seed0固定抽出696組，得到1,777張（29.795%），再以
`trainer_fraction=1.0`載入固定清單；validation仍是683張全量。產物只在：

```text
variants/full35/artifacts/datasets/diagnostic/pose/f0p3-seed0/
variants/full35/artifacts/pose/diagnostic/
```

若要串接診斷 P2/P3，後續命令也必須保留 `--fraction 0.3`。診斷 run 會寫入
`diagnostic-run.json`，正式 fraction=1.0 transition與JOINT preflight會fail-fast拒絕它。
診斷 AP/mAP只看方向，不得填入 formal gate。

## 1. canonical Pose26 P1→P2→P3

GPU空閒後才執行。所有命令自動使用 `bbat5-v1` runtime View與 exact tiled XNOR。

```bash
# P1：最多17 epochs，Pose26 head only，patience10
.venv/bin/python variants/full35/run.py pose --stage p1 --device 0 \
  --name p0-full35-p1-b128-e17-seed0

# P2：最多22 epochs，從 P1 best，neck/head/MASF/attention可微部分，patience12
.venv/bin/python variants/full35/run.py pose --stage p2 --device 0 \
  --name p0-full35-p2-b64a2-e22-seed0 \
  --initial-checkpoint variants/full35/artifacts/pose/p0-full35-p1-b128-e17-seed0/weights/best.pt

# P3：最多100，從 P2 best，patience20
.venv/bin/python variants/full35/run.py pose --stage p3 --device 0 \
  --name p0-full35-p3-b32a4-e100max-seed0 \
  --initial-checkpoint variants/full35/artifacts/pose/p0-full35-p2-b64a2-e22-seed0/weights/best.pt
```

P1–P3沿用 Full35 source：MuSGD、lr0 3.8e-4、lrf .5、momentum .948、wd
2.7e-4、cosine；mosaic0、pose fliplr0。P1/P2只使用原生 cosine與early stop，不在
plateau時額外改 LR或momentum；P3 patience20。空卡實測結果為P1 physical128可行、P2
physical128 OOM但64可行、P3 physical64 OOM但32可行。因此P2使用64×accumulate2、P3
使用32×accumulate4，兩者`nbs=128`，維持每次optimizer update的logical exposure128與
effective weight decay 2.7e-4；這不宣稱與physical128逐值等價，BN仍按64/32更新。

## 2. 產生同口徑 standalone baseline

```bash
.venv/bin/python variants/full35/baseline.py --device 0 \
  --name p3-seed0 \
  --pose-checkpoint variants/full35/artifacts/pose/p0-full35-p3-b32a4-e100max-seed0/weights/best.pt
```

這會分別驗證獨立 Full35 Detect trunk與獨立 Pose P3 trunk，並產生：

```text
variants/full35/baselines/formal-gate-p3-seed0.json
```

不會先把 Pose head接到 Detect trunk後再冒充獨立 Pose baseline。

## 3. 啟用正式 joint input

在 [joint.yaml](../../variants/full35/configs/joint.yaml) 填入：

```yaml
source:
  pose_checkpoint: /absolute/path/to/P3/best.pt
  baseline_metrics: /absolute/path/to/formal-gate-p3-seed0.json
```

再執行：

```bash
.venv/bin/python variants/full35/joint.py preflight
```

必須 `ready=true` 才能開始。

## 4. J1→J2 訓練

```bash
.venv/bin/python variants/full35/joint.py train --device 0 \
  --name full35-joint-adamw-seed0
```

輸出根：

```text
variants/full35/artifacts/fusion/formal/full35-joint-adamw-seed0/
  checkpoints/{best_detect,best_pose,best_joint,last}.pt
  inference/{best_detect,best_pose,best_joint,last}.pt
  validation/epoch-XXXX/{float,bittrue}/
  logs/{events.jsonl,macro.csv,epoch.csv,validation.csv,gate.csv,plateau.csv,plots/}
  resolved-config.json
  factory-report.json
```

J1固定跑滿5 epochs。J2最多40 epochs，以 Bit-True unconditional joint score監控：
超過歷史最佳至少1e-4才重設 stale counter；連續8個 epoch未改善時，下一 epoch起將
所有 parameter group 的 cosine LR乘0.5，整個 J2只做一次。momentum/AdamW beta固定，
stale到17即早停。每次 decision都寫入 `plateau.csv`/JSONL並繪圖；controller state
在停止判斷前存入完整 checkpoint，所以 resume不會重設 stale counter或重複降 LR。

中斷後只可用完整 `checkpoints/last.pt` exact resume：

```bash
.venv/bin/python variants/full35/joint.py train --device 0 \
  --name full35-joint-adamw-seed0 \
  --resume variants/full35/artifacts/fusion/formal/full35-joint-adamw-seed0/checkpoints/last.pt
```

J3不會自動跑。J2通過八項 gate、無BN/gradient/resume問題且有underfit證據時才執行
同一命令加 `--enable-j3`。

## 5. 手動 validation 與 inference

```bash
.venv/bin/python variants/full35/joint.py validate --device 0 \
  --checkpoint PATH_TO_COMBINED_CHECKPOINT --backend both --name manual-check

.venv/bin/python variants/full35/joint.py infer --device 0 --task both \
  --checkpoint PATH_TO_COMBINED_CHECKPOINT \
  --source /path/to/frame.jpg --output-json /tmp/full35-both.json
```

Python整合可直接使用 `yolo_combine.inference.SharedDualPredictor`；`task="both"` 回傳
`{"detect": list[Results], "pose": list[Results]}`，同一呼叫只跑一次 shared graph。
