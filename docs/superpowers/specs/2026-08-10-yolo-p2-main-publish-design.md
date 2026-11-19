# YOLO P2 Main Publish Design

## Goal

Publish a clean, reproducible `yolo_p2/` project to `uxin0205-tech/yolo` on `main` while preserving every existing top-level project and keeping the four validated trained checkpoints in Git.

## Repository integration

- Preserve the current remote `main` history and all existing content, including `original/`, `yolo_binaryqk/`, and `yolo_masf/`.
- Add the curated project only under the top-level `yolo_p2/` directory.
- Include the repository-level agent configuration in `AGENTS.md` and `docs/agents/`.
- Build the change in an isolated worktree based on the latest `origin/main`; do not use the dirty, diverged local `main` checkout as the publication source.
- Push the verified commit directly to `origin/main`, as explicitly requested by the user.

## Curated `yolo_p2/` contents

The published directory must contain everything needed to inspect the study, install the exact fork, rerun the focused checks, and retrain against a separately provisioned COCO 2017 dataset:

- Project entry points and legal/package metadata: `README.md`, `LICENSE`, `P2_PLAN.md`, `pyproject.toml`, and the project `.gitignore`.
- Exact model implementation: the complete `ultralytics/` Python package, including model YAML files and package assets.
- Study orchestration: the complete tracked `p2_study/` tree.
- Focused verification: `tests/test_p2_study.py` and `tests/README.md`.
- P2 design history: the P2 cleanup design and implementation plan under `docs/superpowers/`.
- Four validated checkpoints under `p2_study/results/weights/`:
  - `A0_yolo11m.pt`
  - `A1_best.pt`
  - `A2_stage1_best.pt`
  - `A2_best.pt`
- Seed and deterministic settings in `p2_study/config.yaml` and archived metadata.
- Reports, metrics, CSV/JSON/YAML metadata, comparison image, and README indexes under `p2_study/results/`.

## Local-only retained data

COCO 2017 labels and annotations remain available in the existing local `yolo_p2/p2_study/data/` paths for retraining, but are not copied into the outer Git repository or uploaded to GitHub. Image symlinks likewise remain local. The published project retains:

- `p2_study/coco2017.yaml`
- `p2_study/data/README.md`
- instructions that describe the expected dataset layout

No dataset archive, image, annotation JSON, generated label text, or dataset cache is committed.

## Excluded material

The GitHub copy excludes content that is generated, duplicated, environment-specific, or unrelated to reproducing this P2 study:

- Nested `.git/` history and worktree metadata
- `.venv/`, `ultralytics.egg-info/`, build and distribution output
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.cache` files, and bytecode
- COCO images, labels, annotation JSON, symlinks, and downloaded archives
- Failed or invalidated experiments, duplicate `last.pt`, predictions, large logs, and temporary artifacts
- Upstream `.github/`, `docker/`, `examples/`, general documentation, localization, and unrelated upstream test suites
- Nested upstream `AGENTS.md` and `CLAUDE.md`, so the outer repository has one authoritative agent policy

Disposable local caches may be deleted from the source checkout. The source checkout's nested Git history and local COCO data are retained locally and excluded only from publication.

## Assembly process

1. Fetch `origin/main` and create an isolated publication worktree from that exact commit.
2. Copy the approved repository-level agent configuration into the worktree.
3. Create `yolo_p2/` from an explicit allowlist of files in the local `p2-detect-head-with-weights` snapshot rather than recursively copying the source directory.
4. Confirm no nested repository, cache, dataset payload, broken symlink, or file larger than GitHub's 100 MB single-file limit entered the staging tree.
5. Stage only `AGENTS.md`, `docs/agents/`, this design/its implementation plan, and `yolo_p2/`.
6. Commit once the staged manifest and verification checks pass.
7. Push the resulting commit directly to `origin/main` without force-pushing.

## Verification

Before publication:

- Compare all four checkpoint SHA-256 values against `p2_study/results/weights/README.md` and the archive manifest.
- Load or inspect each checkpoint to confirm A0 has three detection scales and A1/A2 checkpoints have four scales with P2 stride 4.
- Run `pytest yolo_p2/tests/test_p2_study.py -q` from the staged project using the existing environment.
- Run Ruff on the P2 study code and focused test file.
- Validate Markdown links in the curated project.
- Verify `seed: 0`, deterministic training configuration, and portable pretrained/dataset paths are documented.
- Confirm every existing remote top-level project is unchanged by comparing the staged tree to `origin/main` outside the approved paths.
- Confirm the remote `main` commit equals the local publication commit after push.

## Failure handling

- Abort before commit if an allowlisted source file is missing, a checkpoint hash differs, a dataset payload is staged, or a focused test fails.
- Abort rather than force-push if remote `main` advances during assembly; fetch the new tip and rebuild or rebase the publication commit.
- Keep the isolated worktree until the remote commit is verified, then remove it without touching the original local training checkout.

## Success criteria

- GitHub `main` contains a top-level `yolo_p2/` with the curated reproducible project and all four checkpoints.
- Existing `main` projects remain byte-for-byte unchanged.
- No `5090` branch exists remotely.
- No COCO payload, cache, nested Git metadata, invalid experiment, duplicate weight, or large log is committed.
- The focused tests, lint checks, link checks, checkpoint checks, and remote commit comparison all pass.
