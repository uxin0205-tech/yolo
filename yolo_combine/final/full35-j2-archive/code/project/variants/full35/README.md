# Full35 正式主線

Full35 是目前唯一啟用的融合架構。設定、資料 runtime View、logs、validation、
checkpoint 與 inference weights 全部留在本資料夾，不會寫入 Partial75。

## 入口

| 檔案 | 用途 |
| --- | --- |
| [configs/joint.yaml](configs/joint.yaml) | J0/J1/J2與opt-in J3、batch、optimizer、BN、XNOR、validation gate |
| [joint.py](joint.py) | preflight、train、validate、infer |
| [run.py](run.py) | P1/P2/P3 獨立 Pose baseline |
| [baseline.py](baseline.py) | 產生同口徑 Float/Bit-True standalone baseline |
| [cpu_joint_smoke.py](cpu_joint_smoke.py) | 真實 COCO/BBAT5 兩步 CPU smoke |

## 已驗證

- Detect/Pose pre-head graph compatible，587 shared tensors 完整。
- Pose26 head 411 tensors 完整移植。
- 45,580,762 → 26,529,701 parameters，減少41.796%。
- token_tile32 exact XNOR 已取代 full broadcast；Bit-True endpoint tables 不漂移。
- 真實資料兩個 macro-step：三個 gradient group 都存在；EMA updates=2；95個 shared BN
  都不變；84個 head BN 都能更新。
- CPU 證據：[formal-joint-cpu-smoke.json](artifacts/formal-joint-cpu-smoke.json)。
- canonical P1已完成17 epochs；空卡memory gate鎖定P1=128×1、P2=64×2、P3=32×4，
  三者logical exposure皆128，validation固定16。
- canonical P3與Float／Bit-True八項獨立baseline已完成，`joint.py preflight`為
  `ready=true`。
- JOINT Detect batch維持logical128；單次physical128空卡實測OOM，故每個logical batch
  由physical microbatch64×2執行。每macro仍為2×logical128加1×Pose16。

## 正式狀態

`joint.py preflight`目前為`ready=true`；canonical P3 checkpoint與八項baseline均已填入。
舊`full35-joint-adamw-seed0`已產生完整epoch/checkpoint/validation並永久保留，作為
1:1 task weight與較高LR的診斷證據，不會被新版覆寫。新版run名稱固定為
`full35-joint-adamw-v2-j0e8-j1e20-j2e80-seed0`。

新版共用政策是J0只訓練Pose head 8 epochs；J1最多20且patience8，只開neck與兩heads；
J2最多80且patience17，才開backbone layers9+與MASF；J3保留opt-in，才調attention。
Joint task weights改為Detect1.0/Pose0.25，J0單task會重新正規化。

## 30% 診斷入口

GPU 空閒後若要先驗證 trainer、loss 與收斂方向，可執行：

```bash
.venv/bin/python variants/full35/run.py pose --stage p1 --device 0 \
  --fraction 0.3 --epochs 2
```

它使用固定 seed 的 source-group random View（1,777 train／683 full val），輸出隔離在
`artifacts/pose/diagnostic/`。診斷 checkpoint 會被正式 P2/P3 transition 與 JOINT
preflight 拒絕；正式鏈仍須移除 `--fraction`，以5,964張 train重跑。`fraction` 不會降低
單一 physical batch 的 VRAM 峰值，因此不能取代正式 memory gate。

## 已排定的seed0 GPU pipeline

舊版user-level systemd unit曾依序執行
30% P1 memory/stability gate、正式P1/P2/P3、獨立baseline、preflight、JOINT J1/J2與final
Float/Bit-True validation。狀態：

```bash
systemctl --user status yolo-full35-seed0-pipeline-resume7.service
cat artifacts/queue/full35-seed0-pipeline/queue-status.json
cat artifacts/queue/full35-seed0-pipeline/pipeline-status.json
tail -f artifacts/queue/full35-seed0-pipeline/pipeline.log
```

若需人工停止：

```bash
systemctl --user stop yolo-full35-seed0-pipeline-resume7.service
```

停止不會刪除checkpoint或log；partial output會讓自動resume fail-fast，必須先人工稽核。

新版queue入口是：

```bash
.venv/bin/python scripts/run_full35_v2_seed0.py --plan
.venv/bin/python scripts/run_full35_v2_seed0.py
```

它只執行新版preflight、J0→J1→J2與best_joint最終Float/Bit-True驗證；使用獨立
queue state、run與evaluation目錄，J3不會自動執行。

正式啟動後查詢：

```bash
systemctl --user status yolo-full35-v2-seed0.service
cat artifacts/queue/full35-v2-seed0/queue-status.json
journalctl --user -u yolo-full35-v2-seed0.service -f
```

操作順序見[runbook](../../docs/runbooks/2026-08-24-full35-joint-runbook.md)。

2026-08-26 現行正式 J1 保留 Detect logical batch 128，但因 Full35 單次 physical128
在空卡 RTX 5090 實測 OOM，執行時明確拆成 physical64×2；每個 macro 仍包含
2×logical128 Detect 與 1×physical16 Pose。這是顯存執行策略，不是把實驗 batch
規格改為64。完整證據見[JOINT 啟動工作紀錄](../../docs/worklogs/2026-08-26-full35-joint-startup-batch128-and-amp-recovery.md)。

## J3 challenger

J2 已在第25個epoch因patience17正常結束；最佳Bit-True joint score為`0.7100382`。
J3 使用獨立run，從父J2 `last.pt`的`stage_complete=true`狀態exact-resume，不覆寫父run；
父`best_joint.pt`是必須超越的正式基準。規劃與啟動入口：

```bash
.venv/bin/python scripts/run_full35_v2_j3_seed0.py --plan
.venv/bin/python scripts/run_full35_v2_j3_seed0.py
```

監看入口：

```bash
systemctl --user status yolo-full35-v2-j3-b32-seed0.service
cat artifacts/queue/full35-v2-j3-b32-seed0/queue-status.json
journalctl --user -u yolo-full35-v2-j3-b32-seed0.service -f
```

J3最多20 epochs、patience5；只有自己的`best_joint.pt`存在且八項gate全通過時才代表
超越父run。完整理由、設定與風險見[J3工作紀錄](../../docs/worklogs/2026-08-27-full35-j3-challenger.md)。
第一次沿用physical64的啟動在full-unfreeze首個forward OOM，已保留作診斷證據；正式
retry保持logical128並改用physical32×4，每macro共8個Detect microbatches。
