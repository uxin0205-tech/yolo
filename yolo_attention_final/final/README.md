# YOLO26m Bit-True PWL 最終交付

這個目錄是可直接使用與重跑的最終版本，只保留一個正式模型權重和一個命令入口。

## 目錄內容

- `pwl-final-best.pt`：本目錄唯一的 `.pt`；正式 zero-train winner，mAP50-95 為 `0.5067369`。
- `run.py`：統一提供權重檢查、推論、配方顯示、queue 訓練／續跑與狀態查詢。
- `training_recipe.py`：可執行的 LR、分層解凍、early stopping、Bit-True gate 與 rollback 配方。
- `yolo_attention/`：checkpoint 載入與推論所需的最小 Attention 演算法程式。
- `TRAINING.md`：完整演算法、微調方法、實跑設定與命令。
- `RESULTS.md`、`results.json`、`metrics.csv`：實測結果與 provenance。

## 直接使用

```bash
python run.py check
python run.py predict path/to/images --device 0 --save
python run.py recipe
python run.py train                 # 預設只做 dry-run
python run.py train --execute       # 明確啟動或續跑 GPU queue
python run.py status
```

## 之前實際做了什麼微調

目標不是重新訓練一個一般 YOLO，而是讓從 Exact Softmax 改成固定硬體 PWL 後的模型恢復精度。Bit-True 路徑包含 rounding、integer cast 與 table indexing，無法提供有效的一般梯度，因此訓練時使用相同範圍與 endpoints 的可微分 Float-PWL surrogate；每個候選則重新轉成真正 Bit-True PWL，再跑完整 5,000 張 COCO2017 val。

實際流程如下：

1. 從同一個保留的 Phase-B parent，測試 block LR x1、x2、x4。Attention LR 分別為 `5e-6`、`1e-5`、`2e-5`，相鄰 block LR 為其五分之一。
2. 由較佳低 LR 路徑往外解凍 Attention 所在 block，接著解凍 Neck/Detect。
3. 再加入 Backbone 最後一個 stage，最後測試 full-model recovery。
4. 越遠離 Attention 的層使用越低 LR：Attention `5e-6`、相鄰 block `1e-6`、Neck/Detect `5e-7`、Backbone `1e-7`。
5. 全程使用 AdamW、AMP、deterministic seed 0、constant LR、無 warmup；所有 BatchNorm running mean、variance 與 counter 鎖定。
6. 各階段使用 early stopping；每個 `best.pt` 都做實際 Bit-True 驗證。child 若比直接 parent 低超過 `0.001` mAP50-95，就 rollback。

19/19 個 queue 工作已完成。block x1 得到最高單次結果 `0.5069387`，但只比 audited parent 高 `0.0001944`；這個差距小於正式 `+0.001` gate，而且依使用者指示沒有跑 seed 1/2。因此最終正式權重仍保留 zero-train Bit-True checkpoint，避免把單次 evaluator noise 誤認為可靠改善。更詳細的每階段結果見 [RESULTS.md](RESULTS.md)。

## 模型與硬體規格

唯一 checkpoint 是官方 YOLO26m、`scale: m`、80 classes，並同時包含兩個 Bit-True PWL Attention sites。Attention 使用 Hadamard Binary Q/K、固定 power-of-two scale、decomposed 2D bias；score 為 Q8.8、範圍 `[-10, 0]`、20 segments、21 個 UQ1.15 endpoints。分母仍是 exact float reference。
