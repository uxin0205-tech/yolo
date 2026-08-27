# Full35 final 交付包

這個資料夾目前封裝已通過八項 hard gate 的 **J2 best_joint**。J3 仍是 challenger，沒有被自動升格；只有 J3 的八項 mAP50-95 全部下降不超過 0.08，且 joint score 高於 `0.710038215928`，才可另行更新 final。

## 最直接的檔案

- 部署／推論：`weights/combined/inference/best_joint.pt`
- 從最佳點續訓：`weights/combined/full-resume/best_joint.pt`
- 從 J2 結束點續訓：`weights/combined/full-resume/last.pt`
- 完整程式碼快照：`code/project/`
- Full35／XNOR 必要來源：`source_bundle/`
- Float 與 Bit-True 正式 AP：`outputs/validation/accepted-best-joint/`
- 完整訓練 CSV／JSONL／圖：`outputs/training/logs/`
- 機器可讀狀態：`RELEASE_STATUS.json`
- 全檔案 SHA256：`MANIFEST.json`、`CHECKSUMS.sha256`
- CPU 重建／strict-load／shared-forward 證據：`provenance/cpu-shared-forward-smoke.json`
- canonical BBAT5 單張 CPU `task=both` JSON：`outputs/smoke/cpu-bbat5-task-both.json`

## 環境

必須使用 Python 3.12、PyTorch 2.11.0+cu128、Ultralytics 8.4.90；不要自行升級。`source_bundle` 是從原 721MB 研究 bundle 裁出的 Full35 必要子集，保留 Float／Bit-True A2 權重與完整 Python 模組，不含 Partial75 或其他淘汰候選。

## 驗證完整性

```bash
python verify.py
```

建立後已從本資料夾的 source bundle 重建模型、嚴格載入 1,238 個 EMA tensors，並以
CPU 64×64 輸入完成一次 `task=both`；輸出同時包含 `detect`、`pose`，參數仍為
26,529,701。

另以一張 canonical BBAT5 val 影像完成640×640 CLI端到端測試，Detect回傳1筆、Pose
回傳1筆且包含2個keypoints。28.56秒包含factory重建與checkpoint載入，只是CPU smoke，
不是GPU／FPGA latency benchmark。

## 推論

```bash
python run.py infer \
  --checkpoint weights/combined/inference/best_joint.pt \
  --source /path/to/image.jpg \
  --task both --device 0 \
  --output-json outputs/inference.json
```

`--task` 可用 `detect`、`pose`、`both`。`both` 只抽取一次 shared backbone/neck feature，再分別執行 Detect 與 Pose26 head。

## 正式驗證

```bash
python run.py validate \
  --checkpoint weights/combined/inference/best_joint.pt \
  --backend both --device 0 --name final-recheck
```

驗證依賴本機 canonical COCO2017 與 BBAT5 v1；資料本體刻意不複製進 final。固定路徑與 YAML snapshot 在 `configs/`，正式 BBAT5 仍只能是 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`。

## 續訓／J3

完整 J2 run 已結束；若要以 J2 last 作 J3 challenger：

```bash
python run.py train --device 0 \
  --resume weights/combined/full-resume/last.pt \
  --enable-j3 --j3-detect-microbatch 32 \
  --name full35-final-j3-challenger
```

這是 physical microbatch 32 × accumulation 4，logical Detect batch 仍為 128；不要誤稱 physical batch 128。J3 不會因完成就自動覆寫 final。

## 權重角色

- `combined/inference/`：只有 EMA/live state 與 model contract，不可 exact resume。
- `combined/full-resume/`：含 live/EMA、optimizer、scheduler、AMP scaler、RNG、loader、criterion progressive state。
- `standalone/`：Detect A2 與 Pose P3 回退／比較來源，不是融合推論入口。

## 目前正式結果

- shared 參數：26,529,701；兩個獨立模型合計 45,580,762；減少 41.796%。
- Bit-True COCO overall mAP50-95：0.497346
- Bit-True COCO person AP50-95：0.618511
- Bit-True BBAT box mAP50-95：0.630566
- Bit-True BBAT pose mAP50-95：0.901884
- 八項 gate：全部通過；最差 delta 為 ball pose -0.015465，仍高於 -0.08 下限。

## 邊界與風險

- 目前是 seed0 正式結果；第二 seed 尚未執行，因此跨 seed 結論仍屬 provisional。
- J3 challenger 尚未決選，J2 是當前 accepted candidate。
- 尚未做 FPGA、HLS、真實 latency／energy 驗收。
- Ultralytics checkpoint 及衍生部署的 AGPL／商用授權需在非研究交付前另行確認。
