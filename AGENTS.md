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
