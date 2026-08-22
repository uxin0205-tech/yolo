# Repository Guidelines

## Purpose and Remote Workflow

This workspace is for developing and testing YOLO11 models. Run training and substantial model operations directly in this local YOLO workspace using its available GPU.

GitHub is used only for repository synchronization and publishing completed work; it is not the training execution environment.

Keep this workspace synchronized with the project repository:

```bash
git clone https://github.com/uxin0205-tech/yolo11.git
git pull --ff-only
git push origin <branch>
```

Complete work should be committed and pushed, then pulled into this directory. Use focused commits with imperative subjects, for example: `train: add pose experiment config`.

## Root Layout and Data Placement

- `origin/weight/` contains the baseline YOLO11m weights. Treat these as source assets; do not overwrite them.
- `origin/pose/` holds weights and data intended for future YOLO11 pose training.
- Place shared datasets, validation inputs, and evaluation outputs directly at the repository root unless a work-specific directory is more appropriate.

Create one top-level directory per work item, using a concise descriptive name such as `pose-aug-ablation/`. Keep experiments, scripts, logs, and notes for that work inside its directory. Each work directory may contain its own `AGENTS.md` with more specific instructions; those instructions supplement this guide.

## Inherited Rules for Subdirectories

The following rules apply to every work directory and its descendants: this is a YOLO11 development/testing workspace; use the remote host for training; keep the Git clone synchronized and push completed work; store shared datasets and validation material at the root; isolate work by directory; and keep directories tidy. The `origin/` asset descriptions above apply only at this repository root.

## Cleanliness and Safety

Do not leave loose temporary files in the root. Remove unneeded generated artifacts when they are no longer useful, but never delete data or weights without confirming their purpose and recovery path. Keep large checkpoints and datasets out of Git unless the repository explicitly uses Git LFS.

## Agent skills

### Issue tracker

Issues and specs are tracked in GitHub Issues for `uxin0205-tech/yolo`. See `docs/agents/issue-tracker.md`.

### Domain docs

This repository uses a single-context domain documentation layout. See `docs/agents/domain.md`.

## BBAT5 棒球資料集

- BBAT5 detection 一律使用 `/home/uxin/yolo/original/pose/detect_dataset/`。
- BBAT5 pose 一律使用 `/home/uxin/yolo/original/pose/dataset/`。
- 後續凡是使用 BBAT5、BBT5 或棒球專用資料集的工作，必須依任務類型使用上述既有資料集與既有 split；不得自行重新切分、抽樣、複製成另一份資料集、替換來源或改動影像與標註，除非使用者明確授權。
- 若路徑或設定有問題，先記錄問題並只修正必要的路徑或設定，不得用另建 split 規避。
- 完整規範與使用者已核准的 `bbat5-v1`／GitHub snapshot 例外見 `docs/agents/bbat5-datasets.md`。

## 語言與工作紀錄

- 所有進度回報、研究報告、實驗結論、交付說明與困難說明一律使用中文；程式識別字、指令、檔案路徑及必要原文可保留原語言。
- 每次修改、實驗或驗證都必須留下中文工作紀錄，至少包含變更內容與原因、驗證方式與結果、遇到的困難及解法、未解事項或風險。沒有困難時也要明記「無」。
- 工作紀錄放在 `docs/worklogs/`，並同步更新 `docs/worklogs/README.md` 的索引；根層 `README.md` 必須持續提供資料集規範與工作紀錄入口。
