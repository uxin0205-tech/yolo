# BinaryQK README and History Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a complete BinaryQK project README and consolidate the current research history into one `first research` commit.

**Architecture:** Keep generated experiment reports and weights unchanged; make the README a curated navigation and interpretation layer over those canonical artifacts. Preserve the initial repository commit and replace all later commits with one verified snapshot commit, guarded by a local backup and `--force-with-lease`.

**Tech Stack:** Markdown, Bash, Git, Python 3, pytest, YOLO11/Ultralytics project tooling

## Global Constraints

- The latest `main` tree is authoritative.
- The final commit subject must be exactly `first research`.
- Preserve `d16a898` as the parent of the replacement commit.
- Do not stage `original/pose/bbt5.v1i.yolov8/` or `pose_dataset/`.
- Do not modify or delete canonical weights, source weights, datasets, or generated reports.
- Push the rewritten `main` with `--force-with-lease` only after verification.

---

### Task 1: Write the project entry documentation

**Files:**
- Modify: `yolo_binaryqk/README.md`
- Reference: `yolo_binaryqk/OPERATIONS.md`
- Reference: `yolo_binaryqk/artifacts/reports/binary_attention_final_report.md`
- Reference: `yolo_binaryqk/logs/formal-plan/final-audit.json`

**Interfaces:**
- Consumes: canonical metrics, experiment definitions, artifact paths, and maintenance commands
- Produces: one self-contained README with valid relative links

- [x] **Step 1: Replace the short README with the complete research overview**

Include purpose, methodology, 26 variants, key metrics, conclusions, limitations,
directory tree, module table, artifact contract, weight loading, and maintenance
commands. Use canonical values already stored in the report and audit.

- [x] **Step 2: Check every README-local link**

Run a local Markdown-link parser against `yolo_binaryqk/README.md` and require
every relative file or directory target to exist.

- [x] **Step 3: Check formatting**

Run: `git diff --check`

Expected: exit code 0 with no output.

### Task 2: Verify the research snapshot

**Files:**
- Test: `yolo_binaryqk/tests/test_binary_attention.py`
- Test: `yolo_binaryqk/tests/test_model_construction.py`
- Test: `yolo_binaryqk/tests/test_plan_contract.py`
- Verify: `yolo_binaryqk/logs/formal-plan/final-audit.json`

**Interfaces:**
- Consumes: current code, artifacts, reports, and archived weights
- Produces: fresh test and audit evidence for the snapshot

- [x] **Step 1: Run tests**

Run from `yolo_binaryqk/`:

```bash
../.venv/bin/python -m pytest -q
```

Expected: `31 passed`.

- [x] **Step 2: Run the formal audit without rewriting its stored output**

Run from `yolo_binaryqk/`:

```bash
../.venv/bin/python -m binary_attention.cli audit
```

Expected JSON: `ok=true`, `verified_variant_count=26`,
`archived_weight_count=26`, `area_metric_count=26`, and `errors=[]`.

### Task 3: Consolidate and publish Git history

**Files:**
- Modify: Git refs and index only
- Include: all tracked files in the verified current tree plus the README and planning documents
- Exclude: `original/pose/bbt5.v1i.yolov8/`, `pose_dataset/`

**Interfaces:**
- Consumes: verified latest working tree and expected remote tip `2e7db1c`
- Produces: `main` and `origin/main` at one commit named `first research`

- [ ] **Step 1: Save a local backup ref**

Create `backup/pre-first-research-20260808` at the pre-rewrite `main` tip and
verify it resolves to `2e7db1c`.

- [ ] **Step 2: Stage only intended files**

Stage tracked changes and the two planning documents explicitly. Confirm
`git status --short` does not show either unrelated dataset as staged.

- [ ] **Step 3: Create the replacement commit**

Create a commit from the staged final tree with parent `d16a898` and subject
exactly `first research`, then move local `main` to it without changing files.

- [ ] **Step 4: Re-run final checks on the replacement commit**

Run tests, audit summary checks, `git diff --check`, and inspect the two-commit
graph. Require no tracked working-tree differences.

- [ ] **Step 5: Publish safely**

Run:

```bash
git push --force-with-lease=main:2e7db1c1d3e9739f139caa419833cb1249206171 origin main
```

Then fetch and verify `main` equals `origin/main` and the remote-visible commit
subject is `first research`.
