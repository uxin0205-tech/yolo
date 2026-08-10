# Issue tracker：GitHub

本 repo 的 issue 與規格位於 GitHub Issues：

- Repository：`uxin0205-tech/yolo`
- 使用 `gh` CLI 執行操作。
- 在 repo 內執行時，由 `git remote` 自動判斷 repository。

## 常用操作

- 建立：`gh issue create --title "..." --body "..."`
- 讀取：`gh issue view <number> --comments`
- 列出：`gh issue list --state open`
- 留言：`gh issue comment <number> --body "..."`
- 加入標籤：`gh issue edit <number> --add-label "..."`
- 移除標籤：`gh issue edit <number> --remove-label "..."`
- 關閉：`gh issue close <number> --comment "..."`

多行內容使用 heredoc。讀取列表時，依需求取得
`number,title,body,labels,comments` 等 JSON 欄位並以 `jq` 過濾。

## Pull requests 作為需求入口

PRs as a request surface：no。

外部 PR 預設不加入 issue triage 流程。若未來要更改，可直接將此設定改成 `yes`。

GitHub Issues 與 Pull Requests 共用編號；遇到 `#42` 時，先執行
`gh pr view 42`，若不是 PR，再執行 `gh issue view 42`。

## Engineering skill 約定

- 「publish to the issue tracker」：建立 GitHub Issue。
- 「fetch the relevant ticket」：執行 `gh issue view <number> --comments`。

## Wayfinder 約定

- Map：使用一個標記 `wayfinder:map` 的 issue。
- Child ticket：優先使用 GitHub sub-issue；若不可用，改用 task list，並在 child 開頭加入 `Part of #<map>`。
- Child 類型標籤：`wayfinder:research`、`wayfinder:prototype`、`wayfinder:grilling`、`wayfinder:task`。
- Blocking：優先使用 GitHub native issue dependencies；不可用時，在 issue 開頭加入 `Blocked by: #<n>`。
- Claim：`gh issue edit <n> --add-assignee @me`。
- Resolve：先留言結果、關閉 issue，再把 context pointer 加入 map 的 Decisions-so-far。
