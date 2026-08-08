# Phase 1 Recovery, B0/M7, and Per-Class Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the validated B1 artifacts, finish the interrupted B0/M7 extension, add first-class Ball and Bat diagnostics, and launch the remaining Phase 1 GPU workflow as a session-independent user service.

**Architecture:** Keep the existing hash-gated `FormalPipeline`, isolated training worker, fresh-process checkpoint finalizer, and linear dependency DAG. Generalize the existing Ball-only IoU diagnostics into category-aware summaries while retaining legacy Ball fields used by M2/M3 selection, then make the final audit and Markdown report require both classes. Recover stale `smoke_m7` by rerunning only that incomplete stage and let systemd own all long-running GPU work.

**Tech Stack:** Python 3.14, PyTorch 2.11.0, CUDA 12.8, Ultralytics 8.4.90, faster-coco-eval 1.7.2, pytest, systemd user services, Git.

## Global Constraints

- Do not retrain B1-A or B1-B; their native and canonical checkpoint hashes must remain unchanged.
- Do not overwrite `../original/weight/yolo11m.pt` or `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`.
- Keep `configs/static-phase1.yaml` byte-identical so its locked hash and completed-stage reuse remain valid.
- B0 is `data_exposed=true`, evaluation-only, profile-only, and selection-ineligible.
- BEST_PARTIAL selection uses only M2 and M3 validation data and is frozen before test evaluation.
- Preserve the locked common batch size 16 and the existing training hyperparameters.
- Large datasets, caches, runs, logs, and checkpoints stay outside normal Git tracking.
- Long-running training must execute through the user systemd service and survive closing Codex.

---

### Task 1: Reconcile and Commit the Interrupted B0/M7 Recovery Implementation

**Files:**
- Modify: `codex_plan.md`
- Modify: `configs/b0-reference.yaml`
- Modify: `configs/m7-priority.yaml`
- Modify: `masf_yolo/cli.py`
- Modify: `masf_yolo/evaluation/reference.py`
- Modify: `masf_yolo/evaluation/runner.py`
- Modify: `masf_yolo/pipeline.py`
- Modify: `masf_yolo/reporting.py`
- Modify: `masf_yolo/training/completion.py`
- Modify: `masf_yolo/training/runner.py`
- Modify: `masf_yolo/training/worker.py`
- Modify: `masf_yolo/artifacts/finalize.py`
- Modify: `masf_yolo/variants.py`
- Modify: `masf_yolo/workflow.py`
- Test: `tests/artifacts/test_finalize.py`
- Test: `tests/evaluation/test_reference.py`
- Test: `tests/evaluation/test_runner.py`
- Test: `tests/models/test_builder.py`
- Test: `tests/models/test_mfam.py`
- Test: `tests/models/test_transfer.py`
- Test: `tests/training/test_completion.py`
- Test: `tests/training/test_runner.py`
- Test: `tests/training/test_worker.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_reporting.py`
- Test: `tests/test_variants.py`
- Test: `tests/test_workflow.py`

**Interfaces:**
- Consumes: the validated B1 run under `artifacts/static-phase1/runs/b1-b` and B1 canonical checkpoint under `artifacts/static-phase1/training/b1_b`.
- Produces: `TrainingOutputState`, isolated training worker manifests, fresh-process canonical finalization, M7 registry/DAG stages, B0 reference inspection, and seven-model evaluation/profile loops.

- [ ] **Step 1: Record immutable hashes before touching implementation state**

Run:

```bash
sha256sum configs/static-phase1.yaml \
  artifacts/static-phase1/runs/b1-b/results.csv \
  artifacts/static-phase1/runs/b1-b/weights/best.pt \
  artifacts/static-phase1/runs/b1-b/weights/last.pt \
  artifacts/static-phase1/training/b1_b/canonical.pt
wc -l artifacts/static-phase1/runs/b1-b/results.csv
```

Expected: `results.csv` has 91 lines including its header; save the five hashes in the implementation notes and compare them again before launch.

- [ ] **Step 2: Review the interrupted diff for scope and generated files**

Run:

```bash
git diff --check
git status --short
git diff -- masf_yolo tests configs codex_plan.md
```

Expected: source changes are limited to the approved B0/M7/recovery extension; datasets, checkpoints, `labels.cache`, and run outputs remain unstaged.

- [ ] **Step 3: Run each affected suite independently**

Run:

```bash
../.venv/bin/python -m pytest tests/test_*.py -q
../.venv/bin/python -m pytest tests/models tests/data tests/artifacts -q
../.venv/bin/python -m pytest tests/evaluation -q
../.venv/bin/python -m pytest tests/training -q
```

Expected: all four invocations pass. The known one-invocation collection collision is handled in Task 2.

- [ ] **Step 4: Verify real B1 completion classification without mutation**

Run:

```bash
../.venv/bin/python -c 'from pathlib import Path; from masf_yolo.training.completion import classify_training_output; print(classify_training_output(Path("artifacts/static-phase1/runs/b1-b"), 90))'
wc -l artifacts/static-phase1/runs/b1-b/results.csv
```

Expected: `status='complete'`, `completed_epochs=90`, and the line count remains 91.

- [ ] **Step 5: Mark only evidence-backed amendment steps complete**

Update `codex_plan.md` checkboxes for tests and real artifacts already demonstrated. Leave formal M7/M0–M3 training, validation, selection, test, profiles, final audit, and report unchecked.

- [ ] **Step 6: Commit the recovered extension as one focused baseline**

Run:

```bash
git add yolo_masf/codex_plan.md yolo_masf/configs/b0-reference.yaml yolo_masf/configs/m7-priority.yaml \
  yolo_masf/masf_yolo yolo_masf/tests
git diff --cached --check
git diff --cached --name-only
git commit -m "pipeline: recover B1 and add B0 M7 stages"
```

Expected: no dataset, run, cache, log, or checkpoint is present in the commit.

### Task 2: Make the Complete Test Suite Collect Reliably

**Files:**
- Create: `tests/evaluation/__init__.py`
- Create: `tests/training/__init__.py`

**Interfaces:**
- Consumes: duplicate basenames `tests/evaluation/test_runner.py` and `tests/training/test_runner.py`.
- Produces: distinct import names `tests.evaluation.test_runner` and `tests.training.test_runner` during one pytest invocation.

- [ ] **Step 1: Reproduce the collection failure**

Run:

```bash
../.venv/bin/python -m pytest --collect-only -q
```

Expected: collection fails with an import-file mismatch between the two `test_runner.py` files.

- [ ] **Step 2: Add explicit test package boundaries**

Create both files with only a package docstring:

```python
"""Evaluation test package."""
```

and:

```python
"""Training test package."""
```

- [ ] **Step 3: Verify collection and the complete suite**

Run:

```bash
../.venv/bin/python -m pytest --collect-only -q
../.venv/bin/python -m pytest -q
```

Expected: collection succeeds and every test passes in one invocation.

- [ ] **Step 4: Commit the collection fix**

Run:

```bash
git add yolo_masf/tests/evaluation/__init__.py yolo_masf/tests/training/__init__.py
git commit -m "test: isolate runner test modules"
```

### Task 3: Add Category-Aware Ball and Bat Diagnostics

**Files:**
- Modify: `masf_yolo/evaluation/metrics.py`
- Modify: `masf_yolo/evaluation/runner.py`
- Modify: `tests/evaluation/test_metrics.py`
- Modify: `tests/evaluation/test_runner.py`

**Interfaces:**
- Produces: `ObjectObservation(short_side: float, aspect_ratio: float, matched: bool)`.
- Produces: `summarize_class_diagnostics(ground_truth: dict[str, Any], predictions: list[dict[str, Any]], *, category_id: int, iou_threshold: float = 0.5) -> dict[str, Any]`.
- Keeps: `BallObservation`, `ball_observations_from_coco`, `summarize_ball_subsets`, `ball_recall`, `ball_ap`, `ball_ap_s`, and `ball_subsets` for selection compatibility.
- Adds: `class_diagnostics.ball` and `class_diagnostics.bat` to every `metrics.json`.

- [ ] **Step 1: Write failing literal two-class matching tests**

Add a fixture with two Ball GT boxes, one Bat GT box, one matched Ball prediction, one Ball false positive, and one matched Bat prediction. Assert:

```python
assert ball == {
    "gt_count": 2,
    "prediction_count": 2,
    "true_positive_count": 1,
    "missed_count": 1,
    "false_positive_count": 1,
    "precision": 0.5,
    "recall": 0.5,
    "subsets": expected_ball_subsets,
}
assert bat["gt_count"] == 1
assert bat["prediction_count"] == 1
assert bat["precision"] == 1.0
assert bat["recall"] == 1.0
```

Also assert a class with zero support returns `None` for unsupported precision/recall and preserves zero counts.

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```bash
../.venv/bin/python -m pytest tests/evaluation/test_metrics.py -q
```

Expected: import or assertion failure because `summarize_class_diagnostics` does not exist.

- [ ] **Step 3: Implement one score-ordered greedy matcher for any category**

Use a private matcher with this shape:

```python
def _match_category(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    category_id: int,
    iou_threshold: float,
) -> tuple[list[ObjectObservation], int, int]:
    """Return GT observations, prediction count, and matched prediction count."""
```

Filter both GT and predictions by `category_id`, sort predictions by descending score, and allow each prediction and GT annotation to match at most once. Compute precision from matched predictions over predictions, recall from matched GT over GT, and retain `None` when the relevant denominator is zero.

- [ ] **Step 4: Enrich faster-coco-eval per-class fields**

For both category IDs, expose this exact mapping in `per_class[name]`:

```python
{
    "ap": class_stats[0],
    "ap50": class_stats[1],
    "ap75": class_stats[2],
    "ap_s": class_stats[3],
    "ap_m": class_stats[4],
    "ap_l": class_stats[5],
    "ar100": class_stats[8],
    "ar_s": class_stats[9],
    "ar_m": class_stats[10],
    "ar_l": class_stats[11],
    "gt_count": category_gt_count,
}
```

- [ ] **Step 5: Attach Ball and Bat diagnostics while preserving selection fields**

In `run_variant_evaluation`, call `summarize_class_diagnostics` for category 0 and 1, write them beneath `class_diagnostics`, and derive existing top-level Ball fields from `class_diagnostics["ball"]` plus `per_class["ball"]`.

- [ ] **Step 6: Verify focused and evaluation suites**

Run:

```bash
../.venv/bin/python -m pytest tests/evaluation/test_metrics.py tests/evaluation/test_runner.py -q
../.venv/bin/python -m pytest tests/evaluation -q
```

Expected: Ball legacy assertions and new Ball/Bat assertions all pass.

- [ ] **Step 7: Commit category-aware metrics**

Run:

```bash
git add yolo_masf/masf_yolo/evaluation/metrics.py yolo_masf/masf_yolo/evaluation/runner.py \
  yolo_masf/tests/evaluation/test_metrics.py yolo_masf/tests/evaluation/test_runner.py
git commit -m "metrics: report Ball and Bat separately"
```

### Task 4: Require and Render Complete Per-Class Results

**Files:**
- Modify: `masf_yolo/reporting.py`
- Modify: `masf_yolo/pipeline.py`
- Modify: `tests/test_reporting.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `metrics.json` containing `per_class` and `class_diagnostics` mappings for `ball` and `bat`.
- Produces: Markdown subsections for each model/split/class and final-audit errors for absent class data.

- [ ] **Step 1: Write failing report assertions with literal metrics**

Create val/test fixtures for B0 containing distinct Ball and Bat values, rebuild the report, and assert it includes:

```python
assert "### B0" in text
assert "Ball: AP50=" in text
assert "Bat: AP50=" in text
assert "GT=12" in text
assert "predictions=10" in text
assert "false positives=2" in text
```

Keep the B0 data-exposure warning assertion.

- [ ] **Step 2: Write failing final-audit class-matrix assertions**

Populate all required metric files with `{}` and assert the audit reports missing Ball/Bat sections. Then populate both `per_class` and `class_diagnostics` keys for each model and split and assert the class-related errors disappear.

- [ ] **Step 3: Run focused tests and observe RED**

Run:

```bash
../.venv/bin/python -m pytest tests/test_reporting.py tests/test_pipeline.py -q
```

Expected: failures because the report is one-line overall-only and the audit checks file presence only.

- [ ] **Step 4: Add fail-closed class validation to the final audit**

For each model and split, load `metrics.json` and require exactly the named class entries `ball` and `bat` under both `per_class` and `class_diagnostics`. Append errors using the stable form:

```python
errors.append(f"missing {split} {section} class {class_name}: {variant}")
```

- [ ] **Step 5: Render side-by-side overall, Ball, and Bat sections**

Add small formatting helpers rather than embedding nested conditionals in `rebuild_report`. Render overall mAP50–95 and profile data once per model, then validation and test rows for each class with AP50, AP50–95, precision, recall, GT, predictions, missed, false positives, and size/blur diagnostics. Render unsupported metrics as `null`, not `0`.

- [ ] **Step 6: Verify report and pipeline tests**

Run:

```bash
../.venv/bin/python -m pytest tests/test_reporting.py tests/test_pipeline.py -q
```

Expected: all tests pass, including B0 warning and selection-isolation tests.

- [ ] **Step 7: Commit reporting and audit enforcement**

Run:

```bash
git add yolo_masf/masf_yolo/reporting.py yolo_masf/masf_yolo/pipeline.py \
  yolo_masf/tests/test_reporting.py yolo_masf/tests/test_pipeline.py
git commit -m "report: add Ball and Bat comparison sections"
```

### Task 5: Prove Stale-Stage Recovery and Preserve Completed B1

**Files:**
- Modify: `tests/test_workflow.py`
- Modify: `tests/test_pipeline.py`
- Modify: `codex_plan.md`

**Interfaces:**
- Consumes: a `PipelineState(status="running")` for `smoke_m7` with matching inputs but no output hashes.
- Produces: an incremented attempt that invokes `smoke_m7` exactly once while all completed predecessors remain reused.

- [ ] **Step 1: Add a stale-running workflow regression test**

Create completed states through `m7_gate`, write a literal stale `smoke_m7`
state, call `run_stage`, then reload `stages/smoke_m7.json` with
`PipelineState.from_dict` as `recovered` and assert:

```python
assert calls == ["smoke_m7"]
assert recovered.attempt == stale.attempt + 1
assert recovered.status == "completed"
assert recovered.output_hashes == {"canonical": "new-hash"}
```

- [ ] **Step 2: Add a pipeline reuse regression test**

Use fakes for audit, verify, preflight, batch probe, and B1 actions. Seed their completed manifests plus the real dependency hashes and assert executing through `smoke_m7` calls none of those predecessor actions.

- [ ] **Step 3: Run focused tests**

Run:

```bash
../.venv/bin/python -m pytest tests/test_workflow.py tests/test_pipeline.py -q
```

Expected: tests pass with no production change if the current workflow semantics are sufficient; otherwise make only the smallest source change demonstrated by the failing test.

- [ ] **Step 4: Recheck immutable B1 and configuration hashes**

Repeat Task 1 Step 1 and compare every value byte-for-byte with the recorded baseline. Also run:

```bash
sha256sum configs/static-phase1.yaml
```

Expected: no B1 or static config hash changed.

- [ ] **Step 5: Commit recovery evidence**

Run:

```bash
git add yolo_masf/tests/test_workflow.py yolo_masf/tests/test_pipeline.py yolo_masf/codex_plan.md
git commit -m "test: prove stale M7 recovery preserves B1"
```

### Task 6: Full Verification, Publish, and Background Launch

**Files:**
- Modify: `OPERATIONS.md`
- Modify: `codex_plan.md`
- Generate ignored: `artifacts/static-phase1/**`

**Interfaces:**
- Consumes: clean committed source, validated existing B1/B0/M7-gate artifacts, CUDA device 0, and user systemd.
- Produces: pushed commits, active unit `masf-yolo-phase1-ae7dc09d6681.service`, and eventually complete Phase 1 artifacts/report.

- [ ] **Step 1: Document exact detached-session operations**

Add these commands to `OPERATIONS.md` with their read-only/write behavior:

```bash
../.venv/bin/python -m masf_yolo.cli pipeline start --config configs/static-phase1.yaml
../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml
journalctl --user-unit masf-yolo-phase1-ae7dc09d6681.service -f
../.venv/bin/python -m masf_yolo.cli report --config configs/static-phase1.yaml
```

State explicitly that the systemd user service survives closing Codex.

- [ ] **Step 2: Run repository-wide verification**

Run:

```bash
git diff --check
../.venv/bin/python -m compileall -q masf_yolo tests
../.venv/bin/python -m pytest -q
../.venv/bin/python -m masf_yolo.cli verify --config configs/static-phase1.yaml
```

Expected: no whitespace errors, compilation succeeds, the whole suite passes in one invocation, and CUDA/environment/asset pins verify.

- [ ] **Step 3: Run the real M7 CUDA acceptance check**

Run the plan's M7 gate against batch 16 on CUDA device 0 and validate `artifacts/static-phase1/m7_gate/report.json` contains finite loss, optimizer-step evidence, checkpoint hash, variant hash, peak GPU memory, and strict-load success. Do not alter the shared batch on failure.

- [ ] **Step 4: Commit documentation and final checkbox evidence**

Run:

```bash
git add yolo_masf/OPERATIONS.md yolo_masf/codex_plan.md
git commit -m "docs: add detached Phase 1 operations"
git status --short --branch
```

Expected: tracked files are clean; only ignored/untracked datasets, caches, runs, or checkpoints may remain.

- [ ] **Step 5: Push completed source work**

Run:

```bash
git push origin main
```

Expected: `origin/main` contains every focused implementation commit before formal training begins.

- [ ] **Step 6: Start the background pipeline once**

Run:

```bash
../.venv/bin/python -m masf_yolo.cli pipeline start --config configs/static-phase1.yaml
../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml
systemctl --user is-active masf-yolo-phase1-ae7dc09d6681.service
```

Expected: the start response identifies pipeline `ae7dc09d6681`, status reports `active=true`, and systemd reports `active`. The first newly executed stage is `smoke_m7`; B1 stages remain completed.

- [ ] **Step 7: Verify detached progress without waiting for formal completion**

Close no processes. Read status and the service journal after the worker creates the M7 run directory:

```bash
../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml
journalctl --user-unit masf-yolo-phase1-ae7dc09d6681.service --no-pager -n 80
```

Expected: a live worker or advancing M7 epoch is visible. Report the unit name and status/report commands to the user; closing Codex must not stop it.

- [ ] **Step 8: Apply terminal acceptance when the user returns**

After systemd finishes, run:

```bash
../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml
../.venv/bin/python -m masf_yolo.cli report --config configs/static-phase1.yaml
../.venv/bin/python -c 'import json; from pathlib import Path; p=Path("artifacts/static-phase1/final_audit.json"); d=json.loads(p.read_text()); assert d["ok"] is True and not d["errors"]; print(d)'
```

Expected: final state `completed/report`, all seven models have val/test/profile artifacts, Ball and Bat sections exist for every model, selection is M2 or M3, and `final_audit.ok=true`.
