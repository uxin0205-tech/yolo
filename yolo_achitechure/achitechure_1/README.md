# achitechure_1

這是隔離版本的 YOLO26m Attention + P3 MASF 實驗。模型建構、pretrained transfer、Float／Bit-True
轉換、MuSGD phase scope、Bit-True early stopping、斷電續跑、資源保護 queue、COCO／ball-bat 驗證皆已
實作並通過測試。

## 2026-08-24 最終結果

- `fraction=0.3` 與 `fraction=1.0` 的 Full35／Partial75 Phase B、C 均已完成；八個 child 全部由 gate rollback 到各自 A2。
- 完整資料 Phase C 使用 physical batch 8、`nbs=16`、workers 6、patience 7；Full35／Partial75 分別在 10／11 epochs early stopping，沒有 OOM、NaN 或 Inf。
- 同機 RTX 5060 Ti FP16 batch-1 p50 latency 已補齊：A0 25.848 ms、Full35 A2 28.821 ms、Partial75 A2 27.987 ms。依既定 tie-break，Partial75 是 Full35／Partial75 兩個 P3-MASF 架構中的工程 winner。
- Partial75 A2 的 COCO mAP50-95 與 A0 幾乎相同，且 A0 latency 較快；不能宣稱 P3-MASF 已帶來實質整體 accuracy 或效率提升。
- 使用者指定的 BBT5 `detect_dataset` 完整指標已列入最重要的局部分析：overall／bat 最高是 Full35 C-30%，ball 最高仍是 A0。

完整數字、ball／bat 分析、訓練圖、資源紀錄與限制請見
[`final/reports/RTX5060TI_FINAL_0824.md`](final/reports/RTX5060TI_FINAL_0824.md)。完整 AP 表位於
[`final/reports/FULL35_PARTIAL75_AP.md`](final/reports/FULL35_PARTIAL75_AP.md)，機器可讀版本列於
[`results/README.md`](results/README.md)。

可攜式交付包位於 [`final/`](final/README.md)，包含 Full35／Partial75 的確切程式碼、11 個已驗證
Bit-True 候選、10 個 Float best checkpoint、8 次訓練的完整過程圖，以及 COCO2017／BBT5 `detect_dataset` 的完整 AP 報告。

## 文件入口

- `plan.md`：研究目的、限制與目前執行狀態。
- `EXPERIMENT_SPEC.md`：固定架構、訓練與 gate 契約。
- `RUNBOOK.md`：環境、queue、驗證與接續指令。
- `scripts/README.md`：各操作腳本用途。
- `results/README.md`：結果索引與報告邊界。
- `final/README.md`：可複製至其他資料夾的程式碼、權重、設定與驗證入口。

正式環境固定 Python 3.12、PyTorch `2.11.0+cu128`、Ultralytics `8.4.90`。A1／A2／B 使用 physical
batch 16；C 使用 physical batch 8、`nbs=16`；validation 固定 batch 8。Peak VRAM 只作容量診斷，不參與
候選排名。
