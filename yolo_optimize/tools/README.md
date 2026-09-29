# 維護工具

| 工具 | 用途與執行限制 |
| --- | --- |
| archive_research.py | `--name` 建立全新封存，拒絕覆寫既有目的地 |
| archive_addendum.py | 已完成的一次性補充封存，固定目的地已存在，不重跑 |
| prepare_report_publication.py | 在隔離 worktree 產生報告發布樹，本身不 commit／push |
| organize_workspace.py | 上一輪入口整理的歷史工具，不能用來還原本次新目錄 |
| restructure_layout.py | 本次一次性實體遷移工具；已有遷移紀錄時拒絕重跑 |

從 optimize 根層使用 `tools/`。輸入是本機成果，輸出為指定新封存／發布樹與稽核紀錄；不載入 checkpoint、不做訓練、不清除既有結果。工具與報告可納入 Git，封存與模型 bytes 不發布。

舊 `scripts/maintenance/` 已搬到此處；舊研究 scripts 在 [experiments/scripts](<../experiments/scripts/README.md>)。版本提交由主代理完成範圍稽核後另行執行，不是這些工具的自動副作用。
