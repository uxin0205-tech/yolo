# 整合研究小型數據

這裡是 2026-11-17 整理時，從已完成的正式結果抽出的資料。沒有重新訓練或重算 AP，也沒有複製 checkpoint。

| 檔案 | 內容 |
|---|---|
| [stages.csv](stages.csv) | 7 個主線交接節點的指標；空白表示該階段未評估該任務 |
| [lineage.json](lineage.json) | 各節點 checkpoint、SHA、原 manifest 及 MASF／形狀研究不換 parent 的決策 |
| [shapes.csv](shapes.csv) | 原尺寸、9 個固定尺寸 E0、6 個適應最佳，共 16 列；尺寸順序 H×W |
| [stage8-metrics.csv](stage8-metrics.csv) | 原 Stage8 比較表的逐位相同小型副本，包含原自動推薦 |
| [stage8-costs.csv](stage8-costs.csv) | 訓練／folded 參數、檔案大小與局部卷積成本 |
| [provenance.json](provenance.json) | 原始 CSV 路徑與 SHA256 |
| [redraw.py](redraw.py) | 只讀 CSV 的靜態繪圖程式 |

在有 matplotlib／numpy 的 Python 環境執行：

```bash
.venv/bin/python docs/research/data/integrated-20261117/redraw.py
```

可重畫 [主線精度](mainline-accuracy.png) 與 [尺寸取捨](shape-tradeoffs.png)，並輸出 SVG。不需 GPU、PyTorch、checkpoint 或原資料集。圖上以英文縮寫避免字型依賴，軸與圖例的意義由總報告說明。

AP 欄存 0～1，圖用百分比或百分點。Stage1～3 為 Detect 模型，Stage6 之後為融合模型，各節點是順序研究，不能當作從同一 parent 開始的公平消融。Stage5 與 Stage3 權重相同，圖不重複畫成新進步。形狀 MAC 是雙任務 Conv＋P·V 小計，參考是**固定 640×640**，不是正式動態 rect640 的平均實測成本；圖不把它稱為硬體加速。

資料來源與完整論證見 [總報告](../../yolo-research-summary-2026-11-17.md)。
