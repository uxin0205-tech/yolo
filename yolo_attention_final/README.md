# YOLO26m PWL Attention 最終專案

本專案保存從 `v1-br` YOLO26m parent 到最終 Bit-True PWL 權重的完整訓練、驗證、queue 與交付流程。

固定模型規格為官方 `yolo26m.yaml`、`scale: m`、COCO 80 類，且下列兩個 Attention site 必須同時存在：

- `model.10.m.0.attn`
- `model.22.m.0.1.attn`

PWL 規格為 Q8.8 score、範圍 `[-10, 0]`、20 個寬度為 `0.5` 的 segments，以及 21 個 UQ1.15 endpoints（每個 site 共 336 bits）。分母目前仍採 exact float reference。

## 最終成果

LR sweep 與 staged recovery queue 共 19 個工作，已全部完成。最高單次 Bit-True 結果來自 block x1：mAP50-95 為 `0.5069387`；audited parent 為 `0.5067442`，改善只有 `+0.0001944`。由於未達預先設定的 `+0.001` 穩健門檻，而且依使用者指示只跑 seed 0，正式 winner 保留 zero-train fallback `0.5067369`。

- 最終可攜交付與唯一正式權重：[final/](final/README.md)
- 微調方法、演算法與重跑命令：[final/TRAINING.md](final/TRAINING.md)
- 完整實測結果與選擇理由：[final/RESULTS.md](final/RESULTS.md)
- 機器可讀摘要：[final/results.json](final/results.json)
- 方法依據與主要文獻：[docs/research/pwl-final-training-rationale.md](docs/research/pwl-final-training-rationale.md)

## 最快使用方式

```bash
cd final
python run.py check
python run.py predict path/to/images --device 0 --save
python run.py recipe
python run.py train                 # 只預覽，不啟動訓練
python run.py train --execute       # 明確啟動或續跑 GPU queue
python run.py status
```

## 本次怎麼微調

訓練端使用可微分 Float-PWL surrogate，候選選擇一律使用實際 materialize 的 Bit-True checkpoint 在完整 5,000 張 COCO2017 val 上驗證。流程先比較 block LR x1/x2/x4，再依序擴大到 Attention 所在 block、Neck/Detect、Backbone 最後 stage 與 full model。各層使用遞減 LR，BatchNorm running statistics 全程鎖定，並以 early stopping 控制回合數。每階段 child 若比直接 parent 低超過 `0.001` mAP50-95，就 rollback。

完整 queue 可用下列命令檢查：

```bash
PYTHONPATH=src python -m yolo_attention.cli queue validate --queue-root artifacts/lr-sweep-queue
PYTHONPATH=src python -m yolo_attention.cli queue status --queue-root artifacts/lr-sweep-queue
```
