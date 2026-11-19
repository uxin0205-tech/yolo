# 第 2 階段正式背景隊列

**已於 2026-10-19 04:58:48 完成並停止。** S1／S2 均完成五輪，選定 S1 Q10，官方 AP50–95 0.514152。見 [正式結果](RESULTS.md)、[八項 AP CSV](comparison.csv)、[曲線](curves.svg)、[handoff](handoff.json) 及 [交付驗證](handoff-verification.json)。GPU 已釋放，第 3 階段未啟動。

於 2026-10-19 00:35:26（Asia/Taipei）啟動。S1／S2 各自從 `yolo_binaryqk/result/stage1/epoch-00.pt` 獨立開始，五輪相同預算，完整 COCO train118287／val5000；方法、門檻與停止條件見 [protocol](../../../docs/stage2-protocol.md)。

`supervisor.json` 是背景狀態，`monitor.jsonl` 每 600 秒記錄 GPU／進度，`S1`／`S2` 保存各自的訓練資料。`source.json` 固定實作 SHA，進行中不可修改程式／recipe 或搬動此資料夾。`STOP` 在 batch 邊界要求停止。完整完成後才會有比較 JSON／CSV、SVG、`RESULTS.md`、部署常數及 `handoff.json`；然後停止，不啟動第 3 階段。

切換下一階段時將選定 S1 資料夾整理至 `result/stage2/`。小型數據與報告可提交 Git；checkpoint／resume 保留本機，不自動提交。下方資料不作自動加訓授權。
