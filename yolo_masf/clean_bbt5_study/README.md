# Clean BBT5 Study

此目錄是新一輪公平研究的人類可讀入口。實作在 `../masf_yolo/clean/`，鎖定設定在 `../configs/clean/`，資料證據在 `../artifacts/locked-bbt5-dataset/`。

| 路徑 | 用途 |
|---|---|
| `data/` | 資料來源、split 可見性與 lineage 說明 |
| `experiments/` | 每個 Clean 實驗的命名、唯一差異與共同訓練契約 |
| `results/` | GPU 完成後才生成的逐 seed 指標與統整；目前為空 |

完整順序與 54 個主流程 GPU jobs 見 [`../codex_plan.md`](../codex_plan.md)。目前只完成 CPU 規劃與驗證，未啟動 GPU。
