# Clean 實驗單一 GPU 執行清單

目前狀態：`prepared_not_queued`。本文件不代表已啟動訓練。

啟動前必須全部滿足：

1. 既有 GPU 任務與 predecessor systemd units 全部結束。
2. GPU 0 連續三次符合 free memory ≥24 GiB、utilization ≤10%，間隔 60 秒。
3. Clean CPU tests、initializer SHA、dataset hash、12-model feasibility 全部通過。
4. Worker 使用 `artifacts/clean-bbt5-ablation/data/train_val_only.yaml`，其中不得存在 `test` key。
5. Queue 一次只能啟動一個 worker；每個完成 stage 寫入原子 state/result manifest，重啟時不得重跑 completed stage。

順序固定為：12 個 seed42 smoke → 24 個 strict formal → 4 個 P2 control stages → validation selection freeze → 最多 4 個 Selective P2 runs。historical test 與 final holdout 不屬於訓練 queue。

完整模型、epochs、指標與宣稱規則見 [`codex_plan.md`](codex_plan.md)。
