# 2026-09-08：整合 Codex 全域偏好與子代理規則

> 狀態更新：使用者已取消子代理使用；本篇的分工與 Luna 設定僅為歷史，現行設定見 [停用子代理工作紀錄](2026-09-08-disable-subagents.md)。

## 任務與範圍

依使用者本次明確要求，把主動平行委派、可讀訊息與空格、適量測試、自主完成任務，以及清楚簡潔的中文段落寫入主代理的全域記憶，並核對既有子代理設定。舊有行為偏好若有衝突，以本次要求為準。

本次範圍為本機 Codex 指令、既有 Luna 角色提示與工作區文件入口，未執行資料集、程式功能或模型訓練變更。

## 變更內容與原因

原先不存在使用者層級的 `/home/uxin/.codex/AGENTS.md`，本次建立此檔作為跨工作區的持續規則。內容明確涵蓋主代理與子代理的平行分工、共同檔案責任、中文可讀性、必要驗證與停止重複測試的條件，以及依上下文自主推進至完成。使用者最新明確偏好優先於相衝突的舊偏好，同時保留系統與開發者指令、工具權限的實際限制。

已確認既有自訂角色位於 `/home/uxin/.codex/agents/luna-worker.toml`，角色名稱為 `luna_worker`，模型為 `gpt-5.6-luna`，推理強度為 `max`。本次只修改其 `developer_instructions`：明確允許在原委派範圍內再分工，並把遇到未授權決策時的整體停止，改為回報受阻部分、繼續可獨立完成的已授權工作；同步適量驗證與中文表達要求。

主模型設定為 `gpt-6-astra`，推理強度為 `xhigh`；`/home/uxin/.codex/config.toml` 保持原樣。角色設定已被目前會話的協作工具列為可用角色，無須另外建立角色登錄或修改並行設定。

工作區根 `AGENTS.md` 新增全域規則入口與本次偏好說明；`README.md` 補上使用者層級規則入口，保留資料集規範與工作紀錄入口；`docs/worklogs/README.md` 新增本紀錄索引。

## 官方依據

本次使用 OpenAI Docs 技能，實際搜尋並閱讀官方文件，另委派代理平行核對載入規則與子代理設定，並覆核具體草稿。

- [GPT-6 Astra 官方模型提示指南](https://developers.openai.com/api/docs/guides/latest-model#prompting-best-practices)：使用者第 1–5 項與官方針對自主完成、指令衝突、文字風格、子代理委派及適量測試的建議一致。
- [AGENTS.md 官方文件](https://learn.chatgpt.com/docs/agent-configuration/agents-md)：預設從 `~/.codex` 讀取全域指令；非空的 `AGENTS.override.md` 可優先於 `AGENTS.md`；專案指令依目錄層級合併，指令鏈在新執行或 session 啟動時建立。
- [子代理官方文件](https://learn.chatgpt.com/docs/agent-configuration/subagents#custom-agents)：使用者角色可由 `~/.codex/agents/*.toml` 定義。角色明確設定的模型與推理欄位會覆蓋先前解析值，不能假設所有子代理都與主代理相同。

## 驗證方式與結果

已完成與本次文件及提示修改相符的檢查：

- 獨立代理核對第 1–5 項完整性、Luna 再委派與範圍限制、指令優先權及模型設定敘述：通過，未發現實質問題。
- 以 Python `tomllib` 解析全域設定與更新後 Luna 角色檔：通過。角色欄位集合保持相同，除 `developer_instructions` 外所有欄位值均與修改前一致。
- 寫入後內容與已核對草稿逐字比對：通過。`config.toml` 的 SHA-256 與修改前一致。
- 工作區 `AGENTS.md` 除新增前言外，原內容逐字保留，包括 BBAT5 與中文工作紀錄規範。
- 目前未設定 `CODEX_HOME`，不存在全域 `AGENTS.override.md`；全域與工作區根指令合計 7,276 bytes，小於官方預設的 32,768 bytes。此容量檢查只涵蓋目前工作目錄。
- 本次已成功使用協作工具，完成官方文件核對與草稿覆核，確認目前會話的委派工具可用。
- 最終文件檢查：6 份文件／設定無行尾空白，41 個本機 Markdown 連結皆存在；工作區 `git diff --check -- AGENTS.md` 與相對本次修改前副本的差異空白檢查均通過。

這是可逆的提示與文件調整，未新增只複述文件內容的測試，也未執行無關的 YOLO 訓練或測試套件。

## 困難與解法

預設 shell 與 `apply_patch` 都因 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` 無法啟動。改由執行工具的 `require_escalated` 路徑完成必要讀寫；全域路徑位於工作區外，寫入時一併依既有使用者授權取得工具層級核准，未變更權限設定。

原本只有工作區根 `AGENTS.md`，沒有 Codex 使用者層級全域檔案；依官方路徑建立後，在工作區補上入口。舊 Luna 提示可能造成整體提早停止，已縮小為僅暫停依賴未授權決策的部分。

工作區開始時已有許多修改及未追蹤檔案。本次先保存所涉文件的修改前內容並核對雜湊，只處理指定規則與文件入口，保留其他工作。臨時副本位於 `/tmp/codex-global-memory-2026-09-08/`，供本次比對使用。

其他困難：無。

## 未解事項或風險

沒有未解的內容或設定格式問題。本次完成靜態檢查與現有協作工具驗證，未另外啟動新的 Codex 會話驗證全域載入或更新後的 Luna 行為；依官方載入機制，重開 Codex 會話可重新建立完整指令鏈。更深層的專案指令、不同 `CODEX_HOME` 或未來新增的 override 仍會影響實際載入結果。

使用者層級指令與 Luna 角色存放於本機 `~/.codex`，不屬於此 Git 儲存庫；工作區文件保留變更來源及可追溯入口。本次未建立 commit 或推送。
