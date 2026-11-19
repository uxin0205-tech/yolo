# 2026-08-30：Codex session active writer 診斷

## 任務與範圍

診斷 Codex TUI 續接 thread `01a04328-049c-72c0-8aa7-e9987fce98b4` 時出現
`already has an active writer (code -32600)` 的原因；範圍只包含唯讀程序、檔案與官方指令文件檢查。

## 變更內容與原因

- 未修改 Codex session JSONL、專案程式、設定或資料集。
- 新增本工作紀錄，以符合根層 `AGENTS.md` 的中文追溯要求。
- 官方文件確認 `codex resume SESSION_ID` 用於續接原 thread，`codex fork SESSION_ID` 會建立新 chat。

## 驗證方式與結果

- `stat` 確認 session 檔存在、權限正常，大小 30,815,932 bytes，最後修改時間為
  2026-08-30 12:33:07 +0800。
- `lsof` 確認該 JSONL 目前由 `codex` PID `2359865` 以可寫方式開啟，因此確有 active writer。
- `ps` 確認程序位於 `pts/3`，自 2026-08-28 17:16:55 +0800 起仍存活，工作目錄為
  `/home/uxin/yolo`。
- `codex --version` 顯示 `codex-cli 0.148.0`。
- 結論：先在 `pts/3` 正常退出；若終端無法操作，確認無未完成輸出後才送 `SIGTERM`，待
  `lsof` 不再列出 writer 後再執行 `codex resume`。

## 困難與解法

- 困難：sandbox 建立 loopback 介面遭拒，唯讀命令與 `apply_patch` 的預設執行均失敗。
- 解法：唯讀診斷取得授權後完成；session 與程序均未修改。工作紀錄改以相對路徑重試。

## 未解事項或風險

- 尚未終止 PID `2359865`，其中可能仍有未完成工作；應優先回到 `pts/3` 正常退出。
- PID 只代表本次檢查時的程序；重啟後應重新用 `lsof` 確認。
- 官方文件未記載此特定錯誤；根因是依本機 `lsof` 與程序樹的直接證據判定。
