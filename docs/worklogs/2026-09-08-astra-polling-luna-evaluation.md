# 2026-09-08：評估 Astra 輪詢消耗與直接使用 Luna MAX

## 任務與範圍

依使用者要求，只評估 Reddit 提出的等待參數及直接使用 Luna／MAX 的方案。保持單一主代理與 `agents.enabled = false`，未修改 Codex 設定、啟用子代理或啟動模型對照實驗。

## 評估結論

對目前已停用子代理的工作方式，優先考慮直接以 Luna／MAX 作為主代理，省去 Astra 協調與等待子代理的環節。建立 Git 分支本身不會切換模型；若使用分支對話，仍須確認該對話選用的模型與推理強度，且分支可能攜帶已有上下文。只帶必要背景的新對話可避免繼承整段長歷史。

貼文的等待調整是多代理流程的改善方向，不能視為全部 token 消耗的通用修復。對單一主代理的終端輪詢、工具呼叫與研究工作不會自動產生相同效果。

## 來源與證據

[Reddit 原文](https://www.reddit.com/r/codex/comments/1wa9c9d/i_investigated_why_gpt6_astra_burns_quota_so_fast/)作者回報 47 次無新狀態輪詢，累積 7,130,181 個輸入 token，其中 7,114,112 個為快取輸入；並表示套用 25 分鐘等待設定後，30 秒輪詢消失。這是作者自行觀測的結果，不是本站本機實測，也不能把原始輸入總數直接換成訂閱額度扣除量。作者同時承認仍在調查其他用量來源。

[OpenAI Codex GitHub issue #35259](https://github.com/openai/codex/issues/35259)有其他使用者提供長時間等待與終端狀態輪詢的紀錄，描述等待逾時後反覆重新呼叫模型並帶入大量上下文。該 issue 為使用者回報，不能當成官方已確認所有客戶端相同行為的保證。

[官方 Luna 模型頁](https://developers.openai.com/api/docs/models/gpt-5.6-luna)確認支援 `max`，標準 API 每百萬 token 價格為輸入 USD 0.20、快取輸入 USD 0.02、輸出 USD 1.20。[官方 Astra 模型頁](https://developers.openai.com/api/docs/models/gpt-6-astra)對應為 USD 10、USD 1、USD 50。標準短上下文單價的輸入／快取差距為 50 倍，輸出約 41.7 倍；這不保證同樣任務或 Codex 訂閱額度會節省相同比例，仍受推理量、返工、上下文與計費方式影響。

## 本機相容性與驗證結果

目前 CLI 為 `codex-cli 0.153.4`，主模型設定 `gpt-6-astra`／`medium`，子代理開關維持關閉。

以 `codex -c 'features.multi_agent_v2={enabled=true,min_wait_timeout_ms=1500000,default_wait_timeout_ms=1500000,max_wait_timeout_ms=1500000}' features list` 做單次命令的唯讀設定解析：exit code 0，`multi_agent_v2` 顯示啟用。此命令只列功能旗標，不啟動模型；未將覆寫寫回檔案。安裝的原生程式亦包含全部 3 個等待欄位名稱。

將 `min_wait_timeout_ms` 換成無效字串後，CLI 拒絕解析並回報 `FeatureToml` 型別錯誤，支持這組設定被實際解析，而非單純忽略未知文字。設定檔 SHA-256 比對未變。

[公開設定參考](https://learn.chatgpt.com/docs/config-file/config-reference)目前未找到上述 `multi_agent_v2` 與 timeout 欄位。本次未取得對應版本原始碼，也未做 25 分鐘運行測試，因此只確認本機辨識與型別解析，未宣稱全部等待行為已驗證。

25 分鐘是 1,500,000 毫秒。依作者的結果，工作事件可提早喚醒父代理；不宜把這組參數解讀成模型在任何情況下都必須沉默 25 分鐘。它也不能保證阻止父代理在等待前或被喚醒後另外研究、發送訊息。

## 本會話用量快照

唯讀統計當前會話 rollout 的工具名稱、等待參數及最新累計 token 欄位，未輸出私人對話內容。

- 直接工具紀錄：`exec` 32 次、`spawn_agent` 1 次、`followup_task` 2 次、`list_agents` 1 次。
- 沒有直接 `wait_agent` 呼叫，無法把本段對話的用量歸因於 Reddit 所述 30 秒父代理輪詢。
- 累計輸入 3,630,551；其中快取輸入 3,164,544。
- 輸出 31,773；其中推理輸出 7,943。推理輸出為輸出子項，不重複加總。
- 原始累計總數為 3,662,324。這是查核當下的會話快照，仍會隨後續操作增加，不是 OpenAI 內部計費明細，也不包括對其他會話的全面稽核。

從本對話流程觀察，多次工具操作與長上下文反覆帶入是更直接可見的因素；未做各操作計費歸因，不量化各原因占比。

## 困難與解法

公開設定參考未列等待欄位，改用本機 CLI 的解析與無效型別檢查確認支援程度。官方 Markdown 端點與對應版本原始碼讀取失敗，改讀官方 HTML；原始碼及實際等待語意的未驗證部分明確保留。

延續既有沙箱限制，以 `require_escalated` 完成唯讀本機檢查及中文工作紀錄寫入。其他困難：無。

## 未解事項或風險

尚未對同一任務做 Astra／Luna 的成本、成功率與耗時對照，不能保證 Luna MAX 完全取代 Astra 的複雜推理能力。

既有 GPU 每 600 秒監測原則繼續有效。子代理等待 timeout 與 GPU 程序監測是不同機制；若只由父代理負責檢查、又沒有提早事件，25 分鐘的等待可能延後 10 分鐘監測。具體監測實作仍由各專案決定。
