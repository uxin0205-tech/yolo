# 實驗設定：現行入口與歷史契約

目前量化已延後，沒有待啟動的新 QAT。先看 [CURRENT_PLAN](../../docs/CURRENT_PLAN.md) 與[交接決策](../../artifacts/queues/full-model-cumulative-0908/phase-hold.json)。原 [full-model-continuous-0907.json](full-model-continuous-0907.json)、[六組 QAT 清單](../../artifacts/queues/full-model-continuous-0907/selected-qat-jobs-v2.json)及累積 PTQ plan 保留作已執行契約，不是新 GPU 授權。

其餘版本化 YAML 不是自動待執行的 queue，也不表示所有內容已完成。保留原因包括 parent 建圖、accepted 指標、外部 sham、舊結果重建與來源 SHA-256；未證明無依賴前，不更名、不搬移、不刪除。

查舊版本請用[歷史設定索引](../../docs/archive/config-index-before-2026-09-08.md)。不要因版本號大就直接執行；GPU 工作只由經確認的 queue 排入。
