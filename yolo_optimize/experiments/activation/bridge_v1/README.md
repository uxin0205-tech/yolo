# P3 bridge Activation 配對

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../../docs/OPERATIONS.md>)。

狀態：已完成。相同父模型下 SiLU／qSiLU 各 10 輪，qSiLU E2 為新主線起點；不是每項都優於 SiLU，也沒有通過原始所有融合 gate。

- [完整表格、超參數與限制](<RESULTS.md>)
- [權重導航](<../../../reports/checkpoints/README.md>)
- 原始指標與 runs（本機／歷史參照：`artifacts`；未隨本次報告發布）

重建入口為 `verify_selected.py::SelectedSource`，必須恢復 qSiLU 類型、PWL [-10,0]／20 段與 BinaryQK，不可直接以原 SiLU 架構載入 state dict。舊配對 queue 已完成，不應重跑；當時規劃保留於[歷史入口](<../../../docs/history/README.md>)。
