# artifacts/logs

本目錄保存 queue worker 的程序層級 log。它可能包含已被 rewind 的歷史失敗，因此判定目前 run 是否有效時，必須以 `../queue/queue.json` 與 `../runs/<run-id>/metrics/queue-result.json` 為準，不能只搜尋本目錄的 traceback 或 NaN。

- 輸入：`queue run-next --execute`／`queue run --execute` 的 stdout、stderr。
- 輸出：供除錯與時間追蹤的 append-only log。
- Git：log 不提交；本 README 提交。
