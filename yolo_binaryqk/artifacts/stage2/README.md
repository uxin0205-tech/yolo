# 第 2 階段：可學習固定 scale 與 relative bias

S1 Q10 與 S2 二次冪各自從第 1 階段 B2 最佳權重開始。事前方法核對、來源 SHA、預算與停止條件見 [reference-review.json](reference-review.json)。本階段完成後停止，不接第 3 階段。

此目錄保存 CPU／GPU 短測、正式候選結果、部署常數與 handoff。小型 JSON、CSV、SVG 與文件可交付；checkpoint／resume 僅本機保存，不自動提交 Git。正式 run 不覆寫。

正式工作 [formal-20261019-v1](formal-20261019-v1/README.md) 已於 04:58 完成，S1／S2 官方 AP50–95 為 0.514152／0.513711，選 S1。[result/stage1](../../result/stage1/README.md) 是兩組共同 parent，之後切換階段時以 `result/` 交接；目前不啟動下一階段。詳見 [正式報告](formal-20261019-v1/RESULTS.md)、[交付驗證](formal-20261019-v1/handoff-verification.json)、[protocol](../../docs/stage2-protocol.md) 及 [完成工作紀錄](../../../docs/worklogs/2026-10-19-binaryqk-stage2-results.md)。
