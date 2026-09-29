# Full35 Activation 正式實驗配置

本目錄把 `/home/uxin/yolo/yolo_combine/final/full35` 的 accepted J3 `best_joint` 接到
`activation_lab`。所有正式 run 固定使用完整 COCO2017 train／val 與 Canonical BBAT5 v1 formal
train／val；`fraction=1.0`、`resampling=false`，不建立 30% 或其他抽樣資料夾。

## 基線與資料

| 項目 | 固定值 |
| --- | --- |
| 模型 | YOLO26m Full35 shared trunk + COCO80 Detect head + BBAT5 Pose26 head |
| 正式 joint config | `/home/uxin/yolo/yolo_combine/final/full35/configs/joint.yaml`；直接讀取，不複製改寫 |
| 初始 checkpoint | `combine/final/full35/weights/combined/inference/best_joint.pt` |
| checkpoint SHA-256 | `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` |
| COCO | `/home/uxin/yolo/coco2017.yaml`，完整 118,287 train／5,000 val |
| BBAT5 | registry `bbat5-v1`；完整 5,964 formal train／683 formal val |
| 影像尺寸 | 640 |
| backend | Float 與 Bit-True 都驗證；selector 使用 Bit-True |
| XNOR | `tiled_exact`／runtime `bool_tiled`，token tile 32，Q/K STE 關閉 |

Full35 的 BBAT5 box 指標來自 Pose26 head 的 box branch；COCO80 指標來自 Detect head。兩者是同一個
shared-trunk run，不把 BBAT5 二類 YAML 錯接到 COCO80 head。原 COMBINE 的 `0.08` 是相對 standalone fusion baseline 的舊 safety gate。SiLU control 完成後，
activation 專用 gate 已凍結為：以 delivered `best_joint` 八項 BitTrue mAP50-95 為基準，每項最大
下降 `0.015`。此 gate 是淘汰線；候選仍須和同 seed、同 budget SiLU control 比較，通過不等於勝出。

## 已接手的原 Full35 recipe

原模型的正式訓練血緣如下；這些 epochs 用來說明 accepted checkpoint 的來源，不會為每個 activation
候選全部重跑。

| Stage | Epoch 上限 | Patience | Warm-up | Trainable scope／LR |
| --- | ---: | ---: | ---: | --- |
| J0 | 8 | 0 | 1 | Pose head `2e-4`；其餘凍結 |
| J1 | 20 | 8 | 1 | Detect/Pose head `2e-4`、neck `7.5e-5` |
| J2 | 80 | 17 | 1 | heads `2e-4`、MASF `1.5e-4`、neck `7.5e-5`、backbone `1.5e-5` |
| J3 | 20 | 5 | 3 | heads `5e-5`、MASF `3.8e-5`、neck `1.9e-5`、backbone `3.8e-6`、attention `5e-7` |

Accepted J3 實際在 global epoch 63 早停，`best_joint` 是 global epoch 58。若最後需要 from-source
confirmatory run，SiLU 與相同 finalists 才一起重跑 `8+20+80+20=128` epoch 上限；目前預設關閉。

## Activation 各部分訓練配置

| 部分 | Epoch | Seed | Patience／warm-up | 用途與啟動條件 |
| --- | ---: | ---: | --- | --- |
| Baseline reproduction | 0 | 0 | 不適用 | 完整 COCO val + BBAT5 formal val，Float／Bit-True |
| Activation profiling | 0 | 1 | 不適用 | 完整 training split，收集 range、tail、elements 與 region cost；不訓練 |
| Region zero-shot sensitivity | 0 | 1 | 不適用 | `poly_shift` 逐 region；完整 validation，先於任何 recovery |
| Uniform zero-shot | 0 | 1 | 不適用 | Hardswish、ReLU、`qsilu_pq`、`poly_shift`、`poly_quality` 全網替換 |
| Short recovery | 10 | 1 | patience 0／warm-up 1 | SiLU sham + Hardswish + qSiLU + Shift + Quality；固定跑滿、不得候選早停 |
| Region sensitivity | 10 | 1 | patience 0／warm-up 1 | `poly_shift` 每 region one-at-a-time，和同 budget SiLU 比 |
| BCSP policy search | 10 | 1 | patience 0／warm-up 1 | 每個 mixed policy；每資料集／共同 run 最多 8 個 trial |
| Finalist seed 1 | 20 | 1 | patience 5／warm-up 3 | 最多兩個 finalist + 同 seed SiLU |
| Finalist seed 2 | 20 | 2 | patience 5／warm-up 3 | 資源允許才執行，同一組 finalist + SiLU |
| Claim ablation | 20 | 1 | patience 5／warm-up 3 | 只對最終 proposed policy；matched-cost controls |
| PTQ | 0 | 1 | 不適用 | 完整 train calibration；先做 PTQ |
| QAT（必要才做） | 10 | 1 | patience 0／warm-up 1 | 只有 PTQ 超出凍結的 activation budget；J3 LR 全部乘 0.5 |
| Board validation | 0 | 1 | 不適用 | 無訓練；固定硬體 protocol 後量 latency／power／resources |

所有 recovery 共用：AdamW、`weight_decay=2.7e-4`、`beta1=0.948`、`beta2=0.999`、AMP、gradient
clip 10、cosine final factor 0.5。Short／region／search 的探索 LR 固定為原 J3 的 `0.1×`：backbone
`3.8e-7`、neck `1.9e-6`、MASF `3.8e-6`、attention `5e-8`、Detect/Pose heads `5e-6`。Finalists
才回到原 J3 LR、20 epochs；只有最佳點落在最後 3 epochs 且曲線仍改善時，SiLU 與 finalists 一起
追加 10 epochs，上限 30。

## QSILU-PQ 候選

`qsilu_pq` 是針對本模型原始 SiLU 的 C1 分段二次近似，不用 GELU 函數替代 SiLU。它在
`|x|=[0,1,2,4,8]` 使用四段全 dyadic 係數、共用一次平方，`|x|>=8` 為 exact ReLU tail；
Q16.10 與 ONNX gate 已通過。它先跑 uniform zero-shot／short recovery，暫不進 region search，避免
在同一輪同時改變 activation family 與 placement search。研究與 prior-art 邊界見
[qsilu-hardware-approximation.md](../../docs/research/qsilu-hardware-approximation.md)。

## Batch、augmentation 與 BN

- COMBINE base config 是 Detect logical batch 128、physical 64；J3 recovery 沿用 release 實跑 override
  physical 32。每 macro 2 個 logical Detect batches，因此 J3 共 8 次 physical forward、256 images。
- Pose physical batch 16、每 macro 1 batch；loss 權重 Detect `1.0`、Pose `0.25`。
- validation batch：Detect 32、Pose 16；workers 分別 4、8。
- Detect `mosaic=0`、`fliplr=0.5`；Pose `mosaic=0`、`fliplr=0`。
- Shared BN running statistics 凍結，affine trainable；兩個 head 的 BN 依 stage policy 訓練。
- 每個候選從同一 accepted checkpoint 乾淨載入，重建 optimizer、scheduler、EMA 和 RNG；不沿用前一
  候選的 mutable model 或 optimizer state。

## Manifest 與 11 個 regions

正式 manifest 包含 190 個 SiLU reference。Ultralytics 會讓多個 Conv path 共用同一個 stateless
`Conv.default_act`；因此盤點必須使用 `remove_duplicate=False`，否則只會看到 31 個 module objects。

| Region | Sites |
| --- | ---: |
| backbone early／deep／attention | 21／21／3 |
| neck／neck attention／MASF | 33／1／3 |
| Detect one-to-many／one-to-one | 18／18 |
| Pose one-to-many／one-to-one／flow | 24／24／24 |

## 指令

```bash
# 完整 hash、資料與環境 preflight
/home/uxin/yolo/.venv/bin/python scripts/full35_activation.py preflight

# 重新盤點時先產生 draft；只有人工核對後才加 --approve
/home/uxin/yolo/.venv/bin/python scripts/full35_activation.py manifest --approve

# Production graph smoke
/home/uxin/yolo/.venv/bin/python scripts/full35_activation.py \
  smoke --activation poly_shift --device cpu

# 完整 zero-shot validation
/home/uxin/yolo/.venv/bin/python scripts/full35_activation.py validate \
  --activation poly_shift --backend both --device 0 \
  --name zero-shot-uniform-poly-shift-seed1

# recovery 前：完整 training splits 的 190-site distribution profiling
/home/uxin/yolo/.venv/bin/python scripts/full35_profile.py --task both --device 0

# recovery 前：11 regions 的完整 COCO/BBAT5 zero-shot sensitivity
/home/uxin/yolo/.venv/bin/python scripts/full35_sensitivity.py \
  --activation poly_shift --backend both --device 0

# 10-epoch 全量 short recovery
/home/uxin/yolo/.venv/bin/python scripts/full35_activation.py train \
  --activation poly_shift --phase short_recovery --device 0 \
  --name short-recovery-uniform-poly-shift-seed1

# 低 token serial queue：先看單行狀態，再執行所有已註冊且 gate 允許的 jobs
/home/uxin/yolo/.venv/bin/python scripts/full35_queue.py status
/home/uxin/yolo/.venv/bin/python scripts/full35_queue.py run --device 0

# 靜態 queue 結束後，依 frozen selector 執行 matched SiLU + qSiLU seed-1 finalists
/home/uxin/yolo/.venv/bin/python scripts/full35_queue.py \
  --queue training/full35/finalist-seed1-queue.yaml status
/home/uxin/yolo/.venv/bin/python scripts/full35_queue.py \
  --queue training/full35/finalist-seed1-queue.yaml run --device 0
```

完整 machine-readable 數值在 [activation-recipe.yaml](activation-recipe.yaml)，正式 trainer 設定直接引用
`/home/uxin/yolo/yolo_combine/final/full35/configs/joint.yaml`，site 清單在
[contracts/activation-manifest.yaml](contracts/activation-manifest.yaml)，profile/sensitivity 凍結後的 bounded
搜尋政策在 [policy-search-plan.yaml](policy-search-plan.yaml)，可執行的 serial queue 在
[experiment-queue.yaml](experiment-queue.yaml)，通過 frozen selector 後的 seed-1 queue 在
[finalist-seed1-queue.yaml](finalist-seed1-queue.yaml)；verbose GPU 輸出只寫入 `artifacts/runs/full35/queue/`，
console 與監測只保留單行 JSON。

## 2026-08-30 最終狀態

- 靜態 queue：`5 completed / 0 pending / 14 blocked`。qSiLU、Hardswish、`poly_shift`、`poly_quality` 的 10-epoch recovery 均完成；只有 qSiLU 的八項 gate 全部通過。
- 14 個 blocked jobs 是 `poly_shift` uniform prerequisite 未通過後的預註冊結果，不是執行故障。
- Finalist queue：SiLU control 在第 6 個完成 epoch early-stop 並通過 gate；qSiLU 完成 epoch 1，進入 epoch 2 macro 106 後依使用者要求停止，沒有 completion marker。
- qSiLU physical batch 128、64、32 均在第一個 forward OOM；physical 16 可穩定訓練，logical batch 仍維持 128。
- Activation-only 最佳化已停止。下一步先用相同量化設定比較 accepted SiLU 與完成 10-epoch recovery 的 qSiLU 權重。

完整數值、權重、數學與後續 PTQ／QAT 順序，以[權威整合報告](../../reports/completed-activation-integrated-report.md)為準。
