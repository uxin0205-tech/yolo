# P3 bridge 融合主線

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../../docs/OPERATIONS.md>)。

完整 Pose、J0、balanced J1/J2/J3、Pose head 恢復及 BN 校準已有結果；之後 activation 與 KD 在各自獨立目錄。沒有本研究待執行 queue。

- [融合／恢復詳細結果](<BBAT_RECOVERY_RESULTS.md>)
- [目前計畫狀態](<plan.json>)
- [全階段總報告](<../../../reports/final/README.md>)
- [權重與續訓檔導航](<../../../reports/checkpoints/README.md>)

`artifacts/fusion/` 保存每個完整 run，`inference/` 與 `checkpoints/` 用途不同；前者不含完整 optimizer 狀態。原 J3 恢復 E5 作為 activation 的父模型，非所有融合 gate 通過的 best_joint。

MASF 只接 Detect P3；Pose 使用 raw P3/P4/P5，因此推論只切 MASF α 不會直接改 Pose。原始超參數、分段理由與舊執行中訊息已移入[歷史入口](<../../../docs/history/README.md>)。本區程式位置不改，舊來源與所有失敗紀錄保留。
