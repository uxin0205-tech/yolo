# 第 1 階段選定成果：B2

此資料夾由已完成的 `artifacts/stage1/formal-20261018-v2/recovery/` 實體搬移而來。舊路徑只保留相容符號連結；沒有第二份權重。搬移前後 14 個原始檔案 SHA 均相同。

下階段入口：[handoff.json](handoff.json)。選定 `epoch-00.pt`（SHA `db8b11ddec3c3263e9a79d815645f320dc74cb646f2ea638b8ded4c2efe268f9`），官方 AP50–95 0.509889，internal AP50–95 0.503328。B2 sign＋Hadamard，兩個 Attention sites、四個 heads、16 個 scales、native Softmax。完整模型 manifest 嵌於 checkpoint 與 `epoch-00.json`。

全模型恢復五輪沒有超過初篩最佳，因此交付 epoch 0；2026-11-17 已清理完成階段的 `resume.pt`，選定與候選最佳權重保持；目前無精確 optimizer resume。歷史 `handoff.json` 不改寫，現況見 [availability.json](availability.json)。完整比較、成本與負結果見 [正式研究報告](../../artifacts/stage1/formal-20261018-v2/RESULTS.md)。第 2 階段 S1／S2 都從此模型獨立開始。

權重及 resume 不提交 Git；小型指標、設定與紀錄可提交。
