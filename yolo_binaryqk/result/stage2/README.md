# 第 2 階段選定成果：S1 Q10

使用者於 2026-10-19 指定第 3 階段後，將完成的 S1 資料夾實體搬到此處。15 個既有檔案 SHA 均相同，沒有複製權重；原 run 的 S1 路徑為相容符號連結。見 [搬移紀錄](../stage2-relocation.json)。

下一階段入口為 [handoff.json](handoff.json)，選定 `epoch-05.pt`，SHA `9cacbd802f089a6e0aca87c2e0d1743d74458e7c030029be3076dd3090b48dea`，官方 AP50–95 0.514152。B2 sign＋Hadamard、16 個 Q10 scales、signed16／1024 分解式 bias、native Softmax、SiLU 均記錄於完整 manifest。部署常數見 [deployment-constants.json](deployment-constants.json)。

第 3 階段的實作與新結果由 `yolo_attention` 維護，不寫入此成果資料夾。checkpoint／resume 不提交 Git；小型設定、指標與報告可提交。完整比較與負結果仍見原 [第 2 階段報告](../../artifacts/stage2/formal-20261019-v1/RESULTS.md)。

2026-11-17 全輪整理已移除完成階段的 optimizer resume，選定與 S2 最佳權重保持。歷史 `handoff.json` 仍含當時路徑，現況以 [availability.json](availability.json) 為準；只能從權重開始新段，無精確 optimizer 續跑。全輪成果見 [統一交接](../../../result/README.md)。
