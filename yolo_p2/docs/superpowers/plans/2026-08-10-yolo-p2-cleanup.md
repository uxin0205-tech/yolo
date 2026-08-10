# YOLO11m P2 Project Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the completed YOLO11m P2 experiment into the primary `yolo_p2` checkout while preserving retraining capability, seed 0, four valid checkpoints, formal reports, metrics, and figures, then safely remove obsolete generated data and the hidden worktree.

**Architecture:** Keep runtime outputs in regenerable `p2_study/artifacts` and archive accepted experiment outputs in stable `p2_study/results`. Merge the tracked feature branch into the primary checkout only after retained ignored files have been copied to a checksummed staging directory. Remove the worktree only after checkpoint, dataset, test, documentation, and editable-install verification pass.

**Tech Stack:** Git worktrees, Bash/Coreutils, SHA-256, Python 3.12, PyYAML, PyTorch 2.11, Ultralytics 8.4.90, pytest, Markdown.

## Global Constraints

- Preserve seed `0` and `deterministic: true` in the main config, every retained training args file, the machine-readable summary, README, and report.
- Preserve exactly four valid checkpoints: A0 official YOLO11m, A1 formal best, A2 stage-one best, and A2 final formal best.
- Preserve `comparison.png`, `comparison.csv`, `summary.json`, `REPORT.md`, formal COCO metrics, benchmark JSON, training args, and training CSV histories.
- Preserve COCO2017 train/val labels, official annotation JSON, and the image symlink to `/home/uxin/yolo/coco2017/images`.
- Delete invalidated, gate, health, download ZIP, large logs, predictions JSON, duplicate last weights, initialization weights, and rebuildable caches.
- Do not edit, move, or delete anything outside `/home/uxin/yolo/yolo_p2`, `/home/uxin/yolo/.venv`, and a task-specific directory created by `mktemp -d` under `/tmp`.
- Do not remove `.worktrees/p2-study` until all retained files have matching pre/post SHA-256 values and all four checkpoints load successfully.
- Keep runtime `p2_study/artifacts` separate from archived `p2_study/results`; retraining may recreate artifacts but must not overwrite results.
- Add README files only to project-owned functional directories and major source/test directories, not every Python package or generated directory.
- Preserve user changes and stop if either checkout contains unexpected tracked modifications.

---

## File Map

**Create:**

- `README.md` — project-facing overview replacing the upstream marketing README.
- `p2_study/data/README.md` — dataset layout, counts, links, and cache behavior.
- `p2_study/results/README.md` — accepted-result index and archive policy.
- `p2_study/results/weights/README.md` — four checkpoint provenance, epoch, strides, sizes, and SHA-256.
- `p2_study/results/metrics/README.md` — COCO/Ultralytics/benchmark definitions.
- `p2_study/results/metadata/README.md` — seed, environment, and historical metadata semantics.
- `p2_study/results/metadata/archive_manifest.json` — retained file paths, source paths, byte sizes, and SHA-256.
- `tests/README.md` — focused P2 verification commands.
- `ultralytics/README.md` — upstream source ownership and P2 modifications.

**Move or copy into the new archive:**

- `p2_study/results/REPORT.md`
- `p2_study/results/comparison.png`
- `p2_study/results/comparison.csv`
- `p2_study/results/summary.json`
- `p2_study/results/weights/{A0_yolo11m,A1_best,A2_stage1_best,A2_best}.pt`
- `p2_study/results/metrics/{A0,A1,A2}/coco_metrics.json`
- `p2_study/results/metrics/{A0,A1,A2}/benchmark.json`
- `p2_study/results/metrics/A1/{args.yaml,results.csv}`
- `p2_study/results/metrics/A2/stage1/{args.yaml,results.csv}`
- `p2_study/results/metrics/A2/stage2/{args.yaml,results.csv}`
- `p2_study/results/metadata/{config.yaml,preflight.json,state.json,early_stop.json,batch_manifest.json,model_info.json,initial_checkpoints.json,a1_weight_transfer.json}`

**Modify:**

- `p2_study/config.yaml:14` — use portable archived A0 path.
- `p2_study/worker.py:26-36` — resolve relative pretrained path against repository root.
- `tests/test_p2_study.py` — test relative pretrained path resolution.
- `p2_study/README.md` — document current archive/runtime split and final staged epoch count.
- `.gitignore` — track small archived reports/metrics/figures while continuing to ignore data, runtime artifacts, and archived `*.pt`.

**Delete after preservation and verification:**

- The entire remaining `.worktrees/p2-study` directory via `git worktree remove --force`.
- Primary-checkout Python/pytest/Ruff caches.
- Primary top-level obsolete `coco2017.yaml`.

---

### Task 1: Establish a Checksummed Safety Baseline

**Files:**

- Read: `p2_study/artifacts/**`
- Create temporarily: `/tmp/yolo-p2-cleanup.XXXXXX/manifest-before.sha256`
- Create temporarily: `/tmp/yolo-p2-cleanup.XXXXXX/inventory-before.txt`

**Interfaces:**

- Consumes: Approved cleanup specification and current clean feature worktree.
- Produces: `CLEANUP_STAGE`, a validated staging directory containing all retained ignored assets and their hashes.

- [ ] **Step 1: Verify both worktrees and branches**

Run:

```bash
git -C /home/uxin/yolo/yolo_p2 status --short --branch
git -C /home/uxin/yolo/yolo_p2/.worktrees/p2-study status --short --branch
git -C /home/uxin/yolo/yolo_p2 worktree list --porcelain
```

Expected: primary branch `p2-detect-head` has only the already-known untracked `.worktrees/`, `P2_PLAN.md`, and `coco2017.yaml`; feature branch is clean.

- [ ] **Step 2: Create a uniquely scoped staging directory**

Run:

```bash
mktemp -d /tmp/yolo-p2-cleanup.XXXXXX
```

Record the exact returned path as `CLEANUP_STAGE`. Never use an unresolved or empty variable in later deletion commands.

- [ ] **Step 3: Record exact source paths, sizes, and hashes for four weights**

Sources:

```text
/home/uxin/yolo/original/weight/yolo11m.pt
p2_study/artifacts/runs/formal/A1/weights/best.pt
p2_study/artifacts/runs/staged/A2_head/weights/best.pt
p2_study/artifacts/runs/formal/A2/weights/best.pt
```

Run `sha256sum` and `stat --format='%s %n'` on each source. Expected sizes are approximately 38.8–39.6 MiB. Abort if any file is missing or zero bytes.

- [ ] **Step 4: Copy the four weights and all accepted report assets to staging**

Create these staging subdirectories:

```text
weights/
metrics/A0/
metrics/A1/
metrics/A2/stage1/
metrics/A2/stage2/
metadata/
report/
data/
```

Copy the exact files listed in the File Map. Copy `p2_study/data/coco2017` without `labels/train2017.cache` and `labels/val2017.cache`; preserve the `images` symlink. Copy `p2_study/data/annotations`.

- [ ] **Step 5: Verify staged copies**

Generate SHA-256 for every staged file and compare the four weight hashes to Step 3. Verify:

```text
train labels directory exists
val labels directory exists
instances_train2017.json exists
instances_val2017.json exists
images symlink target is /home/uxin/yolo/coco2017/images
comparison.png exists and is non-empty
seed: 0 appears in config and all three retained args.yaml files
```

Do not proceed on any mismatch.

### Task 2: Make the Archived A0 Path Portable

**Files:**

- Modify: `p2_study/config.yaml:14`
- Modify: `p2_study/worker.py:26-36`
- Modify: `tests/test_p2_study.py`

**Interfaces:**

- Consumes: A repository-relative `study.pretrained` string.
- Produces: `load_config(path: str | Path) -> dict` with `study.pretrained` resolved against `ROOT`.

- [ ] **Step 1: Add the failing config-resolution test**

Append:

```python
from pathlib import Path

import yaml

from p2_study import ROOT
from p2_study.worker import load_config


def test_load_config_resolves_relative_pretrained_from_repo_root(tmp_path):
    config = {
        "study": {
            "dataset": "p2_study/coco2017.yaml",
            "dataset_root": "p2_study/data/coco2017",
            "annotation_root": "p2_study/data/annotations",
            "pretrained": "p2_study/results/weights/A0_yolo11m.pt",
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    loaded = load_config(path)

    assert loaded["study"]["pretrained"] == str(
        (ROOT / "p2_study/results/weights/A0_yolo11m.pt").resolve()
    )
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
/home/uxin/yolo/.venv/bin/python -m pytest tests/test_p2_study.py::test_load_config_resolves_relative_pretrained_from_repo_root -q
```

Expected: FAIL because `study.pretrained` remains relative.

- [ ] **Step 3: Resolve pretrained in `load_config`**

Add:

```python
data["study"]["pretrained"] = str((ROOT / data["study"]["pretrained"]).resolve())
```

Set `p2_study/config.yaml`:

```yaml
pretrained: p2_study/results/weights/A0_yolo11m.pt
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
/home/uxin/yolo/.venv/bin/python -m pytest tests/test_p2_study.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the portable configuration change**

```bash
git add p2_study/config.yaml p2_study/worker.py tests/test_p2_study.py
git commit -m "Make P2 pretrained path portable"
```

### Task 3: Fast-Forward the Feature Work into the Primary Checkout

**Files:**

- Move temporarily: primary `P2_PLAN.md` and `coco2017.yaml` into `CLEANUP_STAGE/primary-untracked/`
- Merge: `feature/yolo11m-p2-study` into `p2-detect-head`

**Interfaces:**

- Consumes: Clean feature branch with Task 2 committed and checksummed staging from Task 1.
- Produces: Primary checkout containing all tracked P2 implementation and design/plan history.

- [ ] **Step 1: Compare primary untracked files with feature equivalents**

Use `cmp` for `P2_PLAN.md`. Archive both top-level untracked files into `CLEANUP_STAGE/primary-untracked`; do not delete them yet.

- [ ] **Step 2: Verify fast-forward ancestry**

Run:

```bash
git -C /home/uxin/yolo/yolo_p2 merge-base --is-ancestor p2-detect-head feature/yolo11m-p2-study
```

Expected: exit 0.

- [ ] **Step 3: Fast-forward the primary branch**

Run:

```bash
git -C /home/uxin/yolo/yolo_p2 merge --ff-only feature/yolo11m-p2-study
```

Expected: primary branch advances to the feature branch tip without a merge commit.

- [ ] **Step 4: Confirm tracked integration**

Verify `p2_study/*.py`, `tests/test_p2_study.py`, `ultralytics/cfg/models/11/yolo11-p2.yaml`, the design, and this plan exist in the primary checkout.

### Task 4: Materialize the Stable Data and Results Archive

**Files:**

- Create: `p2_study/data/**`
- Create: `p2_study/results/**`
- Create: `p2_study/results/metadata/archive_manifest.json`
- Modify: `.gitignore`

**Interfaces:**

- Consumes: Verified files in `CLEANUP_STAGE`.
- Produces: Stable, documented archive in the primary checkout; runtime artifacts remain absent and regenerable.

- [ ] **Step 1: Restore retained data to the primary checkout**

Copy staged labels, annotations, and the images symlink into `/home/uxin/yolo/yolo_p2/p2_study/data`. Verify the symlink target and exact annotation counts:

```text
train images: 118287
val images: 5000
train annotations: 860001
val annotations: 36781
classes: 80
```

- [ ] **Step 2: Create the accepted results tree**

Copy report assets, four weights, metrics, training histories, and metadata to the exact File Map paths. Keep `REPORT.md` and `comparison.png` in the same directory so the existing relative image link remains valid.

- [ ] **Step 3: Generate archive manifest**

Write JSON with this shape:

```json
{
  "seed": 0,
  "created_at": "ISO-8601 timestamp",
  "weights": {
    "A0": {"path": "p2_study/results/weights/A0_yolo11m.pt", "bytes": 0, "sha256": "..."},
    "A1": {"path": "p2_study/results/weights/A1_best.pt", "bytes": 0, "sha256": "..."},
    "A2_stage1": {"path": "p2_study/results/weights/A2_stage1_best.pt", "bytes": 0, "sha256": "..."},
    "A2": {"path": "p2_study/results/weights/A2_best.pt", "bytes": 0, "sha256": "..."}
  }
}
```

Populate actual sizes and hashes; zero and ellipsis are schema examples only and must never appear in the written manifest.

- [ ] **Step 4: Update ignore rules**

At the end of `.gitignore`, keep:

```gitignore
/p2_study/artifacts/
/p2_study/data/
!/p2_study/results/
!/p2_study/results/**
/p2_study/results/weights/*.pt
```

This tracks reports, JSON, CSV, PNG, YAML, and README files while leaving four large checkpoints local/ignored.

### Task 5: Build the README Navigation Layer

**Files:**

- Replace: `README.md`
- Modify: `p2_study/README.md`
- Create: `p2_study/data/README.md`
- Create: `p2_study/results/README.md`
- Create: `p2_study/results/weights/README.md`
- Create: `p2_study/results/metrics/README.md`
- Create: `p2_study/results/metadata/README.md`
- Create: `tests/README.md`
- Create: `ultralytics/README.md`

**Interfaces:**

- Consumes: Final archive paths and manifest generated in Task 4.
- Produces: A navigable project where purpose, training, results, and every project-owned functional directory are understandable without reading implementation code.

- [ ] **Step 1: Replace root README with project documentation**

Include:

```text
Project purpose and P2 Detect Head explanation
A0/A1/A2 definitions
P2 output strides 4/8/16/32
Environment activation and editable install
COCO2017 location and counts
Seed 0 and deterministic settings
Training/status/resume commands
Validation and inference examples
Headline result table
Links to REPORT.md, comparison.png, metrics, and four weights
Final directory tree
Upstream Ultralytics version and license notice
```

- [ ] **Step 2: Update operations README**

Document that `artifacts` is runtime/regenerable, `results` is accepted/archive-only, and A2 actually completed 20 + 73 epochs with final evaluation from stage-two epoch 39.

- [ ] **Step 3: Add focused directory READMEs**

Each README must answer:

```text
What is this directory?
What files belong here?
Which files are generated or immutable?
How is it used?
What does it depend on?
```

- [ ] **Step 4: Verify links and required statements**

Run a local script that extracts relative Markdown links and asserts every local target exists. Assert `seed 0`, all four checkpoint names, `comparison.png`, and `REPORT.md` are present in the documentation.

### Task 6: Verify the Consolidated Project Before Deletion

**Files:**

- Read: primary checkout and `p2_study/results/**`
- Read: `CLEANUP_STAGE`

**Interfaces:**

- Consumes: Consolidated primary checkout.
- Produces: Fresh evidence that allows worktree deletion.

- [ ] **Step 1: Compare four checkpoint hashes**

Run `sha256sum` on staged and final weights. Expected: four exact matches.

- [ ] **Step 2: Load all checkpoints**

Use `YOLO(path)` and assert:

```text
A0: Detect nl=3, strides=[8.0, 16.0, 32.0]
A1: Detect nl=4, strides=[4.0, 8.0, 16.0, 32.0]
A2_stage1: Detect nl=4, strides=[4.0, 8.0, 16.0, 32.0]
A2: Detect nl=4, strides=[4.0, 8.0, 16.0, 32.0]
```

- [ ] **Step 3: Reinstall editable package from primary checkout**

Run:

```bash
/home/uxin/yolo/.venv/bin/python -m pip install --no-deps -e /home/uxin/yolo/yolo_p2
```

Verify `Path(ultralytics.__file__).resolve()` is under `/home/uxin/yolo/yolo_p2`, not under `.worktrees`.

- [ ] **Step 4: Run tests**

Run:

```bash
cd /home/uxin/yolo/yolo_p2
/home/uxin/yolo/.venv/bin/python -m pytest tests/test_p2_study.py -q
git diff --check
```

Expected: all focused tests pass and `git diff --check` exits 0.

- [ ] **Step 5: Verify retained report and data**

Recompute CSV epoch counts, report-required strings, annotation counts, seed locations, benchmark sample counts, and `comparison.png` non-zero size.

### Task 7: Remove the Worktree and Rebuildable Waste

**Files:**

- Delete: `/home/uxin/yolo/yolo_p2/.worktrees/p2-study`
- Delete: primary cache directories only
- Delete: `/home/uxin/yolo/yolo_p2/coco2017.yaml`

**Interfaces:**

- Consumes: Successful Task 6 evidence and exact validated worktree path.
- Produces: One clean primary checkout with no duplicate experiment tree or rebuildable caches.

- [ ] **Step 1: Record the deletion manifest**

Before deletion, record sizes for invalidated, gate, health, downloads, logs, cache, predictions, duplicate last checkpoints, and the full worktree. Store the human-readable summary in `p2_study/results/metadata/cleanup_summary.md`.

- [ ] **Step 2: Remove the linked worktree using Git**

Resolve and assert the target equals exactly:

```text
/home/uxin/yolo/yolo_p2/.worktrees/p2-study
```

Then run:

```bash
git -C /home/uxin/yolo/yolo_p2 worktree remove --force /home/uxin/yolo/yolo_p2/.worktrees/p2-study
```

Never substitute a broader path.

- [ ] **Step 3: Delete merged feature branch**

Run:

```bash
git -C /home/uxin/yolo/yolo_p2 branch -d feature/yolo11m-p2-study
```

Expected: branch deletes only because it is fully merged.

- [ ] **Step 4: Remove primary caches and obsolete YAML**

Enumerate cache targets under the exact primary checkout, print them, verify each resolved path has `/home/uxin/yolo/yolo_p2` as a parent, then remove only directories named `__pycache__`, `.pytest_cache`, or `.ruff_cache`. Remove the known obsolete top-level `coco2017.yaml`.

- [ ] **Step 5: Remove temporary staging after one final hash comparison**

Compare final weight hashes against `CLEANUP_STAGE` once more. Only then remove the exact `CLEANUP_STAGE` directory.

### Task 8: Final Documentation and Repository Verification

**Files:**

- Modify: `p2_study/results/metadata/cleanup_summary.md`
- Track: small results/docs/metrics/config files
- Ignore: data and four `*.pt` checkpoints

**Interfaces:**

- Consumes: Single-checkout cleaned project.
- Produces: Final evidence, tracked documentation, and a clean reproducible handoff.

- [ ] **Step 1: Verify final structure and size**

Record:

```text
du -sh /home/uxin/yolo/yolo_p2
git worktree list --porcelain
git status --short --branch
find p2_study/results -maxdepth 4 -type f
```

Expected: no linked `p2-study` worktree, all documented result paths exist, and total size is approximately 1.3–1.5 GiB.

- [ ] **Step 2: Run final full verification**

Freshly run:

```bash
/home/uxin/yolo/.venv/bin/python -m pytest tests/test_p2_study.py -q
git diff --check
```

Repeat four-checkpoint load/hash checks, seed/documentation checks, data counts, and Markdown link checks.

- [ ] **Step 3: Inspect tracked and ignored files**

Confirm:

```text
README and result metadata appear in git status
four checkpoint files exist but are ignored
COCO data exists but is ignored
runtime artifacts directory is absent or ignored
no invalidated, gate, health, downloads, logs, predictions, last.pt, or cache directories remain
```

- [ ] **Step 4: Commit the consolidated project**

```bash
git add .gitignore README.md P2_PLAN.md p2_study tests/README.md ultralytics/README.md docs/superpowers
git commit -m "Organize reproducible YOLO11m P2 study"
```

Do not force-add `p2_study/results/weights/*.pt` or `p2_study/data`.

- [ ] **Step 5: Verify commit and handoff**

Run:

```bash
git status --short --branch
git show --stat --oneline --summary HEAD
```

Report the final size, reclaimed size, commit, four checkpoint paths/hashes, report path, test count, and the fact that deleted generated/invalidated data is not recoverable from the project directory.
