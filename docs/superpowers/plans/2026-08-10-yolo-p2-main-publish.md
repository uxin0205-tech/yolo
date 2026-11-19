# YOLO P2 Main Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a curated, reproducible `yolo_p2/` project with four trained checkpoints to `uxin0205-tech/yolo` on `main` without changing its existing projects.

**Architecture:** Treat `/home/uxin/yolo/yolo_p2` as the protected source checkout and assemble an explicit allowlisted snapshot in an isolated worktree based on the latest `origin/main`. Keep COCO payloads and nested Git metadata local, validate the staged snapshot, then make one non-force publication commit and push `HEAD:main`.

**Tech Stack:** Git worktrees, Git archive, Bash, Python 3, PyTorch, Ultralytics YOLO, pytest, Ruff, Markdown.

## Global Constraints

- Preserve remote `main` history and every existing project, especially `original/`, `yolo_binaryqk/`, and `yolo_masf/`.
- Publish only the repository agent docs, this design and plan, and the new top-level `yolo_p2/` directory.
- Source P2 content from commit `dd1ab311f1e9a6559973bcaf147ee20072eb9c03` (`p2-detect-head-with-weights`).
- Keep local COCO labels, annotations, image symlinks, and nested `.git/`; never publish them.
- Keep exactly `A0_yolo11m.pt`, `A1_best.pt`, `A2_stage1_best.pt`, and `A2_best.pt` as trained checkpoints.
- Do not publish caches, virtual environments, egg-info, failed experiments, duplicate weights, predictions, archives, large logs, upstream CI/examples/docker/general docs, or unrelated upstream tests.
- No file may exceed GitHub's 100 MB single-file limit.
- Push directly to `origin/main` only after confirming remote `main` has not advanced; never force-push.

---

### Task 1: Prepare a clean publication baseline

**Files:**
- Preserve: `/home/uxin/yolo/yolo_p2/.git/`
- Preserve: `/home/uxin/yolo/yolo_p2/p2_study/data/`
- Delete locally: generated cache directories under `/home/uxin/yolo/yolo_p2/`
- Create: `/tmp/yolo-main-publish-20260810/` as an isolated Git worktree

**Interfaces:**
- Consumes: outer repository remote `origin` and source commit `dd1ab311f1e9a6559973bcaf147ee20072eb9c03`.
- Produces: clean worktree `/tmp/yolo-main-publish-20260810` at the latest `origin/main`, on local branch `agent/publish-yolo-p2-main`.

- [ ] **Step 1: Verify the exact repository and source state**

Run:

```bash
git -C /home/uxin/yolo remote -v
git -C /home/uxin/yolo/yolo_p2 rev-parse p2-detect-head-with-weights
git -C /home/uxin/yolo/yolo_p2 status --short --branch
git -C /home/uxin/yolo ls-remote --heads origin 5090
```

Expected: the outer remote is `uxin0205-tech/yolo`, the source hash is `dd1ab311f1e9a6559973bcaf147ee20072eb9c03`, and the final command prints nothing because remote branch `5090` is absent.

- [ ] **Step 2: Audit disposable local cache targets before deletion**

Run:

```bash
find /home/uxin/yolo/yolo_p2 -xdev -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name ultralytics.egg-info \) -print
find /home/uxin/yolo/yolo_p2/p2_study/data -maxdepth 2 -print | sed -n '1,80p'
git -C /home/uxin/yolo/yolo_p2 rev-parse --git-dir
```

Expected: the first command lists only caches or package metadata; the data and `.git` checks prove the retained local resources still exist.

- [ ] **Step 3: Delete only the audited cache classes**

Run:

```bash
find /home/uxin/yolo/yolo_p2 -xdev -depth -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name ultralytics.egg-info \) -exec rm -rf -- {} +
```

Expected: the audited cache directories are removed; `p2_study/data/` and `.git/` remain.

- [ ] **Step 4: Fetch current `main` and create the isolated worktree**

Run:

```bash
test ! -e /tmp/yolo-main-publish-20260810
git -C /home/uxin/yolo fetch origin main
git -C /home/uxin/yolo branch --list agent/publish-yolo-p2-main
git -C /home/uxin/yolo worktree add -b agent/publish-yolo-p2-main /tmp/yolo-main-publish-20260810 origin/main
git -C /tmp/yolo-main-publish-20260810 status --short --branch
```

Expected: the target path and branch were unused, the worktree is clean, and its branch starts exactly at `origin/main`.

### Task 2: Assemble the explicit curated snapshot

**Files:**
- Create: `/tmp/yolo-main-publish-20260810/AGENTS.md`
- Create: `/tmp/yolo-main-publish-20260810/docs/agents/domain.md`
- Create: `/tmp/yolo-main-publish-20260810/docs/agents/issue-tracker.md`
- Create: `/tmp/yolo-main-publish-20260810/docs/superpowers/specs/2026-08-10-yolo-p2-main-publish-design.md`
- Create: `/tmp/yolo-main-publish-20260810/docs/superpowers/plans/2026-08-10-yolo-p2-main-publish.md`
- Create: `/tmp/yolo-main-publish-20260810/yolo_p2/` from the allowlist below

**Interfaces:**
- Consumes: clean publication worktree from Task 1 and immutable P2 source commit `dd1ab31`.
- Produces: a staged candidate tree containing only approved repository docs and curated P2 files.

- [ ] **Step 1: Copy the approved repository-level docs**

Run:

```bash
install -D /home/uxin/yolo/AGENTS.md /tmp/yolo-main-publish-20260810/AGENTS.md
install -D /home/uxin/yolo/docs/agents/domain.md /tmp/yolo-main-publish-20260810/docs/agents/domain.md
install -D /home/uxin/yolo/docs/agents/issue-tracker.md /tmp/yolo-main-publish-20260810/docs/agents/issue-tracker.md
install -D /home/uxin/yolo/docs/superpowers/specs/2026-08-10-yolo-p2-main-publish-design.md /tmp/yolo-main-publish-20260810/docs/superpowers/specs/2026-08-10-yolo-p2-main-publish-design.md
install -D /home/uxin/yolo/docs/superpowers/plans/2026-08-10-yolo-p2-main-publish.md /tmp/yolo-main-publish-20260810/docs/superpowers/plans/2026-08-10-yolo-p2-main-publish.md
```

Expected: five approved documentation files exist in the publication worktree.

- [ ] **Step 2: Export only the approved P2 paths**

Run:

```bash
git -C /home/uxin/yolo/yolo_p2 archive --format=tar --output=/tmp/yolo-p2-curated-20260810.tar p2-detect-head-with-weights .gitignore LICENSE P2_PLAN.md README.md pyproject.toml p2_study ultralytics tests/README.md tests/test_p2_study.py docs/superpowers/plans/2026-08-10-yolo-p2-cleanup.md docs/superpowers/specs/2026-08-10-yolo-p2-cleanup-design.md
mkdir /tmp/yolo-main-publish-20260810/yolo_p2
tar -xf /tmp/yolo-p2-curated-20260810.tar -C /tmp/yolo-main-publish-20260810/yolo_p2
```

Expected: the archive succeeds from tracked objects, so no ignored dataset, cache, symlink target, or nested `.git` can enter the snapshot.

- [ ] **Step 3: Make active dataset paths portable in the publication copy**

Use `apply_patch` in `/tmp/yolo-main-publish-20260810` to apply exactly:

```diff
*** Begin Patch
*** Update File: yolo_p2/p2_study/coco2017.yaml
@@
-path: /home/uxin/yolo/yolo_p2/p2_study/data/coco2017
+path: p2_study/data/coco2017
*** Update File: yolo_p2/README.md
@@
-cd /home/uxin/yolo/yolo_p2
+cd yolo_p2
@@
-- `images` 是指向 `/home/uxin/yolo/coco2017/images` 的符號連結
+- `images` 是指向本機 COCO2017 `images/` 目錄的符號連結
*** Update File: yolo_p2/p2_study/data/README.md
@@
-│   ├── images -> /home/uxin/yolo/coco2017/images
+│   ├── images -> <COCO2017_ROOT>/images
*** End Patch
```

Expected: executable config and user-facing setup instructions no longer depend on `/home/uxin/yolo`; archived training arguments and historical plans remain unchanged as provenance.

- [ ] **Step 4: Verify the allowlist and forbidden payload boundaries**

Run:

```bash
find /tmp/yolo-main-publish-20260810/yolo_p2 -type l -print
find /tmp/yolo-main-publish-20260810/yolo_p2 -type f -size +99M -print
find /tmp/yolo-main-publish-20260810/yolo_p2 -type d \( -name .git -o -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name ultralytics.egg-info \) -print
find /tmp/yolo-main-publish-20260810/yolo_p2/p2_study/data -mindepth 1 -maxdepth 2 -print
```

Expected: the first three commands print nothing; the data command prints only `p2_study/data/README.md`.

- [ ] **Step 5: Stage only the approved publication scope**

Run:

```bash
git -C /tmp/yolo-main-publish-20260810 add AGENTS.md docs/agents docs/superpowers/specs/2026-08-10-yolo-p2-main-publish-design.md docs/superpowers/plans/2026-08-10-yolo-p2-main-publish.md yolo_p2
git -C /tmp/yolo-main-publish-20260810 diff --cached --check
git -C /tmp/yolo-main-publish-20260810 diff --cached --name-status
```

Expected: only the approved docs and `yolo_p2/` appear; no existing project is modified or deleted.

### Task 3: Verify reproducibility and preserved results

**Files:**
- Test: `/tmp/yolo-main-publish-20260810/yolo_p2/tests/test_p2_study.py`
- Verify: `/tmp/yolo-main-publish-20260810/yolo_p2/p2_study/config.yaml`
- Verify: `/tmp/yolo-main-publish-20260810/yolo_p2/p2_study/results/weights/*.pt`
- Verify: all Markdown files under `/tmp/yolo-main-publish-20260810/yolo_p2/`

**Interfaces:**
- Consumes: staged candidate from Task 2.
- Produces: fresh evidence that code, checkpoints, documentation, and unchanged existing projects satisfy the design.

- [ ] **Step 1: Verify all four checkpoint hashes**

Run:

```bash
cd /tmp/yolo-main-publish-20260810/yolo_p2
sha256sum p2_study/results/weights/*.pt
```

Expected:

```text
d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95  p2_study/results/weights/A0_yolo11m.pt
1f3b96464b1867f0e7c5ba9cb956049f2faaa9aa7102969c20152a62608d1005  p2_study/results/weights/A1_best.pt
2e7067454a42564b54e7520f4fb049bc6ec1dac62496aa6c5d7c8a417df3a767  p2_study/results/weights/A2_best.pt
d2b51e1790a6c7747809ef7dd9e17773b86730fbcc6cd56a29d6fbde8101a658  p2_study/results/weights/A2_stage1_best.pt
```

- [ ] **Step 2: Run focused architecture, transfer, and configuration tests**

Run:

```bash
cd /tmp/yolo-main-publish-20260810/yolo_p2
/home/uxin/yolo/.venv/bin/python -m pytest tests/test_p2_study.py -q
/home/uxin/yolo/.venv/bin/python -m ruff check p2_study/worker.py tests/test_p2_study.py
```

Expected: six tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 3: Inspect the four serialized checkpoint heads**

Run from `/tmp/yolo-main-publish-20260810/yolo_p2`:

```bash
/home/uxin/yolo/.venv/bin/python - <<'PY'
from pathlib import Path

from ultralytics import YOLO

expected = {
    "A0_yolo11m.pt": (3, [8.0, 16.0, 32.0]),
    "A1_best.pt": (4, [4.0, 8.0, 16.0, 32.0]),
    "A2_stage1_best.pt": (4, [4.0, 8.0, 16.0, 32.0]),
    "A2_best.pt": (4, [4.0, 8.0, 16.0, 32.0]),
}
root = Path("p2_study/results/weights")
for name, (scales, strides) in expected.items():
    head = YOLO(root / name).model.model[-1]
    assert head.nl == scales, (name, head.nl)
    assert head.stride.tolist() == strides, (name, head.stride.tolist())
PY
```

Expected: the script exits without an assertion; A0 has three scales and all trained P2 checkpoints have stride-4 P2 output.

- [ ] **Step 4: Validate Markdown links and reproducibility settings**

Run from `/tmp/yolo-main-publish-20260810/yolo_p2`:

```bash
/home/uxin/yolo/.venv/bin/python - <<'PY'
from pathlib import Path
import re

root = Path.cwd().resolve()
pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
errors = []
for document in root.rglob("*.md"):
    for raw in pattern.findall(document.read_text(errors="replace")):
        target = raw.strip().split()[0].strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        resolved = (document.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"{document.relative_to(root)} -> {target}")
assert not errors, "Broken local links:\n" + "\n".join(errors)
print(f"Markdown links valid across {sum(1 for _ in root.rglob('*.md'))} files")
PY
rg -n '^  seed: 0$|^  deterministic: true$|pretrained: p2_study/results/weights/A0_yolo11m.pt' p2_study/config.yaml
if rg -n '/home/uxin/yolo|\.worktrees' README.md p2_study/config.yaml p2_study/coco2017.yaml p2_study/README.md p2_study/data/README.md p2_study/worker.py tests/README.md tests/test_p2_study.py ultralytics/README.md; then exit 1; fi
```

Expected: every local Markdown link resolves; all three config settings are found; the active-file absolute path and stale worktree search prints nothing. Historical plans and archived result metadata are intentionally outside this portability search.

- [ ] **Step 5: Prove existing remote projects are unchanged and forbidden files are absent**

Run:

```bash
git -C /tmp/yolo-main-publish-20260810 diff --quiet origin/main -- original yolo_binaryqk yolo_masf
if git -C /tmp/yolo-main-publish-20260810 diff --cached --name-only | rg '(^|/)(\.git|__pycache__|\.pytest_cache|\.ruff_cache|ultralytics\.egg-info|annotations|labels|images)(/|$)|(^|/)(last\.pt|predictions|invalidated|\.cache$)'; then exit 1; fi
git -C /tmp/yolo-main-publish-20260810 status --short
```

Expected: the unchanged-project comparison succeeds, the forbidden-file search prints nothing, and status lists only the approved staged additions.

### Task 4: Commit, publish directly to `main`, and verify GitHub

**Files:**
- Commit: all staged approved additions from Tasks 2 and 3
- Remote update: `refs/heads/main` on `git@github.com:uxin0205-tech/yolo.git`

**Interfaces:**
- Consumes: fully verified staged candidate and unchanged `origin/main` base.
- Produces: one publication commit on GitHub `main`, then a verified and removable local worktree.

- [ ] **Step 1: Commit the verified snapshot**

Run:

```bash
git -C /tmp/yolo-main-publish-20260810 config user.name Codex
git -C /tmp/yolo-main-publish-20260810 config user.email codex@local
git -C /tmp/yolo-main-publish-20260810 commit -m "Add reproducible YOLO11m P2 study"
git -C /tmp/yolo-main-publish-20260810 status --short --branch
```

Expected: one commit is created and the publication worktree is clean.

- [ ] **Step 2: Detect a remote-main race before pushing**

Run:

```bash
git -C /tmp/yolo-main-publish-20260810 fetch origin main
git -C /tmp/yolo-main-publish-20260810 rev-parse HEAD^
git -C /tmp/yolo-main-publish-20260810 rev-parse origin/main
```

Expected: the two hashes are identical. If they differ, do not push; rebuild the commit on the newly fetched `origin/main`.

- [ ] **Step 3: Push the publication commit without force**

Run:

```bash
GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' git -C /tmp/yolo-main-publish-20260810 push git@github.com:uxin0205-tech/yolo.git HEAD:main
```

Expected: GitHub reports a fast-forward update to `main`.

- [ ] **Step 4: Verify remote `main` and checkpoint paths**

Run:

```bash
git -C /tmp/yolo-main-publish-20260810 rev-parse HEAD
git -C /tmp/yolo-main-publish-20260810 ls-remote git@github.com:uxin0205-tech/yolo.git refs/heads/main refs/heads/5090
git -C /tmp/yolo-main-publish-20260810 ls-tree -r --name-only HEAD yolo_p2/p2_study/results/weights
```

Expected: local `HEAD` equals remote `refs/heads/main`, no `refs/heads/5090` line exists, and the four `.pt` files plus their README are listed.

- [ ] **Step 5: Remove only temporary publication resources**

Run after remote verification:

```bash
git -C /home/uxin/yolo worktree remove /tmp/yolo-main-publish-20260810
git -C /home/uxin/yolo branch -d agent/publish-yolo-p2-main
rm -f /tmp/yolo-p2-curated-20260810.tar
git -C /home/uxin/yolo worktree list
```

Expected: the temporary worktree, local publication branch, and archive are gone; `/home/uxin/yolo/yolo_p2`, its nested Git history, and local COCO data remain.
