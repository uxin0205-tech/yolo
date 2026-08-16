# Clean BBT5 Study

此目錄是新一輪公平研究的人類可讀入口。實作在 `../masf_yolo/clean/`，鎖定設定在 `../configs/clean/`，資料證據在 `../artifacts/locked-bbt5-dataset/`。

| 路徑 | 用途 |
|---|---|
| `data/` | 資料來源、split 可見性與 lineage 說明 |
| `experiments/` | 每個 Clean 實驗的命名、唯一差異與共同訓練契約 |
| `results/` | 已完成的逐 seed 指標、mean/std、硬體資料、逐實驗 README 與正式中文報告 |

40/40 主流程 jobs 與完整後處理皆已完成。先讀 [`results/README.md`](results/README.md)，再由 [`results/REPORT.md`](results/REPORT.md) 查看方法、排名、硬體取捨、限制與證據位置。
