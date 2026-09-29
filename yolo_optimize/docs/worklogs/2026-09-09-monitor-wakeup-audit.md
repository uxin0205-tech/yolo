# 600 秒等待與模型喚醒核對

## 內容與原因

使用者要求正常時由 shell 等待，僅錯誤、停滯或完成事件才讓模型處理，避免反覆空輪詢消耗 token。本次只核對執行介面與既有監測程式，沒有更動訓練、權重或 queue。

## 驗證方式與結果

- 唯讀檢查 `run_rep17_pair.py` 與 `run_scope_pair.py`：既有 `wait_job` 使用 `child.wait(timeout=600)`，逾時直接繼續等待；這本身不呼叫模型。Queue 可自動接續預定 job，但最後會進入 `awaiting_analysis`，不能據此聲稱模型會自動接手。
- 核對本回合可用工具：有執行與等待介面，未發現直接訂閱本機事件並喚醒目前會話的工具。工具等待與已結束回合的重新啟動是兩件事。
- 唯讀執行 `codex exec resume --help`：本機 CLI 支援指定會話 ID 接續；環境也提供一致的會話 ID。尚未驗證 CLI 是否能重新載入此介面的會話，亦未實際呼叫接續命令。
- 開啟官方 [Non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode) 與 [App Server](https://learn.chatgpt.com/docs/app-server)：官方提供指定 ID 的 `codex exec resume`，以及 `thread/resume`、`turn/start` 接續機制。這些是可串接機制，不代表本次已部署事件喚醒。
- 純外部程序等待不生成模型 token；模型發起工具、處理工具回傳、推理及回覆會有用量。未取得帳戶實際 token 或週用量，不能宣稱精確消耗或完全零用量。

## 困難與解法

缺少已驗證、連到目前會話的事件喚醒通道。將「shell 仍在執行」與「模型會自動繼續」明確區分，不用空輪詢假冒零成本喚醒，不啟動其他代理或變更帳戶設定。

## 未解事項與風險

尚未安裝或端到端驗證自動喚醒橋接；本次未讀訓練 log，未查 GPU 或訓練 process，故不據此更新實驗完成狀態。若後續實作橋接，需要同一會話定位、單一執行者、事件去重與既有權限保留，並防止反覆接續形成額外用量。
