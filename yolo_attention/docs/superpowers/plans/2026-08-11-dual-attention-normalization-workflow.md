# YOLO26m 雙 Attention 與 Normalization 工作流實作計畫

## 目標

建立 CPU-first、可維護的研究骨架，使正式實驗同時替換 YOLO26m 的兩個 Attention，並能以同一個 CLI 選擇、檢查及執行 normalization screening。此階段不啟動 GPU 訓練，也不把延後的 mixed-precision 量化併入主線。

## 實作順序

1. 在 `tests/` 先加入以下行為測試：
   - YOLO26m YAML 轉換必須找到兩個 Attention，且路徑與預期一致。
   - 各 normalization 候選輸出非負、有限且每列正規化。
   - Multimax 只保留指定的 top-k 機率。
   - N0/N1/A-FINAL 出現在工作流，且量化階段明確標成 optional/deferred。
   - CLI 能輸出完整工作流，不開始訓練。
2. 擴充 `VariantConfig` 與 `normalization.py`：所有候選共用 `scores -> probabilities` 介面，由 factory 建立；PMP 由 wrapper 處理，不放進 Attention 主體。
3. 修改 `convert_yolo26_model`：掃描所有官方 `Attention`，跳過已轉換模組；對 YOLO26m 正式模式驗證兩個預期路徑，避免靜默漏改。
4. 擴充 experiment registry：保留 I/H/T5 10-epoch screening、winner Direct/Progressive recovery，再加入 N0 zero-train、N1 5-epoch normalization recovery 與 A-FINAL。量化僅列 optional。
5. 以既有 `yolo26-attention` CLI 作唯一公開入口，加入 `workflow` 命令；scripts 只能是薄包裝，不複製研究邏輯。
6. 更新 `AGENTS.md`、README、architecture 與設定說明，規定模組責任、執行方式、結果結構、文件同步與測試要求。

## 驗收

- 全部 pytest 通過。
- Ruff 通過。
- `yolo26-attention workflow --json` 能列出主線與 optional 量化。
- CPU smoke 能回報兩個 converted paths。
- 所有新資料夾有 README。

## 不在本次範圍

- 不執行 COCO2017 GPU 訓練。
- 不宣稱 FPGA 實測速度或 INT5/INT6 原生硬體支援。
- 不自動選出 I/H/T5 或 normalization winner；winner 必須由實驗結果決定。
