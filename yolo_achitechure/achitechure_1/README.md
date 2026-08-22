# achitechure_1

這是隔離版本的 YOLO26m Attention + P3 MASF 實驗。模型建構、pretrained transfer、Float／Bit-True
轉換、MuSGD phase scope、Bit-True early stopping、斷電續跑、資源保護 queue、COCO／ball-bat 驗證皆已
實作並通過測試。

## 2026-08-22 階段性結果

- `fraction=0.3` 的 Full35／Partial75 Phase B、C 已完成，四個 child 均由 gate rollback 到各自 A2。
- `fraction=1.0` 的 Phase B 對照已完成；比 30% run 回升，但仍不足以取代 A2。
- `fraction=1.0` Phase C 使用 physical batch 8 × accumulate 2、workers 6、patience 7；該次 Full35
  訓練在第一個 epoch 尚未產生 checkpoint 前失敗，未完成結果不納入結論。
- 目前沒有證據宣稱整體 accuracy 改善。Partial75 是較合理的工程候選，但正式 winner 仍缺 same-GPU
  FP16 p50 latency。

完整數字、ball／bat 分析、資源紀錄與限制請見
[`results/rtx5060ti-half-0822.md`](results/rtx5060ti-half-0822.md)。機器可讀版本為
[`results/rtx5060ti-half-0822.json`](results/rtx5060ti-half-0822.json) 與
[`results/rtx5060ti-half-0822.csv`](results/rtx5060ti-half-0822.csv)。

可攜式交付包位於 [`final/`](final/README.md)，包含 Full35／Partial75 的確切程式碼、9 個已驗證
Bit-True 候選、8 個 Float checkpoint，以及 COCO2017／BBT5 `detect_dataset` 的完整 AP 報告。

## 文件入口

- `plan.md`：研究目的、限制與目前執行狀態。
- `EXPERIMENT_SPEC.md`：固定架構、訓練與 gate 契約。
- `RUNBOOK.md`：環境、queue、驗證與接續指令。
- `scripts/README.md`：各操作腳本用途。
- `results/README.md`：結果索引與報告邊界。
- `final/README.md`：可複製至其他資料夾的程式碼、權重、設定與驗證入口。

## 全域研究規範

- [BBAT5／BBT5 固定資料集政策](../../docs/agents/bbat5-datasets.md)：detect 與 pose 必須使用固定來源，不自行重切或替換。
- [全域工作紀錄](../../docs/worklogs/README.md)：所有變更、驗證、困難與解法均以中文留存。
- [根層 README](../../README.md)：共用規則與文件總入口。

正式環境固定 Python 3.12、PyTorch `2.11.0+cu128`、Ultralytics `8.4.90`。A1／A2／B 使用 physical
batch 16；C 使用 physical batch 8、`nbs=16`；validation 固定 batch 8。Peak VRAM 只作容量診斷，不參與
候選排名。
