## 使用者全域工作偏好

遵循 `/home/uxin/.codex/AGENTS.md`。目前採單一主代理模式，不啟動或委派子代理；技能中的分工步驟由主代理直接完成。此偏好取代先前的主動協作要求，並保留清楚簡潔的中文表達、適量驗證與自主完成任務規則。

最新變更見 [2026-09-08 停用子代理工作紀錄](docs/worklogs/2026-09-08-disable-subagents.md)。

## Agent skills

### Issue tracker

Issues and specs are tracked in GitHub Issues for `uxin0205-tech/yolo`. See `docs/agents/issue-tracker.md`.

### Domain docs

This repository uses a single-context domain documentation layout. See `docs/agents/domain.md`.

## PTQ、QAT、Training 與 GPU 工作監測

依全域規則，PTQ、QAT、training 及其他 GPU 工作期間每 600 秒監測一次，異常在任務範圍內自主處理，完成後直接接續分析；具體細節由各專案決定。規則變更見 [2026-09-08 工作紀錄](docs/worklogs/2026-09-08-gpu-monitoring-policy.md)。

## BBAT5 棒球資料集

- 所有新 BBAT5 detection、pose 與融合實驗一律使用不可變的 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`。
- 正式 Pose YAML 是 `configs/pose.yaml`；正式 ball/bat Detect YAML 是 `configs/detect.yaml`；全域 machine-readable registry 是 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。
- `/home/uxin/yolo/original/pose/dataset/` 與 `detect_dataset/` 是唯讀原始／歷史來源，只用於稽核與重建，不得再作新訓練入口。
- 子專案 `artifacts/datasets/` 只可建立保留 bbat5-v1 assignment 與 labels 的可重建 runtime View，不得成為另一個資料版本。
- 後續凡是使用 BBAT5、BBT5 或棒球專用資料集的工作，不得自行重新切分、抽樣、替換來源或改動影像與標註，除非使用者明確授權建立新版本。
- 若路徑或設定有問題，先記錄問題並只修正必要的路徑或設定，不得用另建 split 規避。
- 完整規範與 GitHub snapshot 的發行邊界見 `docs/agents/bbat5-datasets.md`。
- 依使用者 2026-08-23 明確授權，GitHub 另保存 `original/` 的原始 Pose、歷史 Detect 與
  canonical lineage；這只增加可追溯來源，不把 raw/basic split 升格為正式訓練入口。
- `original/` 發布永久排除 weights、checkpoint、cache、run 與重複 archive；Git 中既有的
  `original/**/*.pt` 必須移除，本機權重不得因此刪除。

## 語言與工作紀錄

- 所有進度回報、研究報告、實驗結論、交付說明與困難說明一律使用中文；程式識別字、指令、檔案路徑及必要原文可保留原語言。
- 每次修改、實驗或驗證都必須留下中文工作紀錄，至少包含變更內容與原因、驗證方式與結果、遇到的困難及解法、未解事項或風險。沒有困難時也要明記「無」。
- 工作紀錄放在 `docs/worklogs/`，並同步更新 `docs/worklogs/README.md` 的索引；根層 `README.md` 必須持續提供資料集規範與工作紀錄入口。

## Cleanup-v1 發行政策（2026-09-29）

- 本分支只發布程式、測試、設定、文件、報告、統計圖表與資料 lineage；不發布 checkpoint、模型、分片權重、資料集影像／標註或 LFS 指標。
- 上述歷史 snapshot 發布授權仍保留作紀錄；Cleanup-v1 的發行內容以本節為準。本機 canonical BBAT5 v1、原始資料、split 和 labels 仍不可變。
- 本次清理以遠端 main 的內容為來源；本機未提交的其他研究修改留待後續獨立更新。
- 提交前執行 `python3 tools/check_repository_hygiene.py`；單檔上限 25 MiB，CI 同步檢查。禁止以 `git add -f` 規避大型產物政策。
- 歷史數值、manifest 與 checksum 不因排除資產而重新宣稱已驗證；完整重跑需要外部資料與權重。
