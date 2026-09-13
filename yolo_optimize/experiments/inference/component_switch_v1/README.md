# 同權重 BinaryQK／MASF 開關診斷

狀態：兩組新增 GPU 完整驗證已完成，兩組既有完整結果重用；沒有重新訓練。

目的：確認目前 qSiLU E2 的模組位置、開關語意與精度依賴性。不能用本結果判定移除二值補償設計並重新訓練的效果。

`evaluate_switches.py` 以 SHA 核對目前權重，先做 CPU 架構／alpha 旁路／Pose 隔離檢查，再做 MASF off 的 BinaryQK 和 FP-QK 兩組。兩组均使用 COCO val5000 及 BBAT5 v1 val683，640、Detect batch32、Pose batch16、同一 BitTrue PWL [-10,0]／20 段。FP-QK 仍保留舊相對位置偏置，並非完全原生 Attention。

使用者更正目標後已加入 `STOP_AFTER_CURRENT` 防止未來誤啟動；當時的 queue 程式已載入記憶體，兩組在本次更新前均完成。未刪除、不覆寫有效結果。`run_queue.py` 目前啟動即停止，不會自動重跑。

產物：`artifacts/preflight.json`、`artifacts/binary_masf_off.json`、`artifacts/fp_masf_off.json`，各組完整原始評估保留於同名子目錄。兩組 MASF on 的來源是 `../../activation/bridge_v1/artifacts/selected-and-teacher-probe-v1/summary.json`。

[目前模型白話入口](<../../../reports/current-model/README.md>)／[新重訓計畫](<../../attention_recovery_v1/README.md>)。Git 可收錄程式及 JSON／報告，不含模型、資料集或 cache；本次未執行發布。

新增 `audit_binary.py`：CPU 驗證固定係數、無效 gamma、signed-dot 等價與 surrogate 梯度，產物 `artifacts/binary-audit-v1.json`。這不是 AP 或 scale 最優性測試，正式重訓仍未啟動。
