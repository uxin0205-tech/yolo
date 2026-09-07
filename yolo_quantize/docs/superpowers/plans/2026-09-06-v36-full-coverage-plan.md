# V36 Full-Coverage qSiLU Weight Search Implementation Plan

## 2026-09-07完成度稽核補充

- [x] CPU 148×9 profile、immutable writer與routes已產出。
- [x] special 12 + uniform 12 + final 1 PTQ實驗已完成。
- [x] external-control continuation已實作，3 epochs QAT完成；epoch1/2搜尋deployment gate通過。
- [x] 真實148路權重覆蓋與plan/source/checkpoint hashes已核對，最新相關pytest 32 passed。
- [ ] PTQ loader仍是較早qSiLU parent；未完成設計要求的CPU/PTQ/QAT同parent比較。
- [ ] 十大區域獨立accuracy sensitivity、逐區累積新比較、export reload與整數部署尚待執行。
- [ ] 後續四天計畫尚未接為live queue，詳見[整合報告](../../reports/2026-09-07-full-model-audit-four-day-plan.md)。

下方為原始建立時checklist，保留作歷史；最新完成度以上表為準。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** 建立承接 V35 的 148 層全覆蓋 PTQ/sensitivity queue，依 special→uniform→final PTQ→短 continuation QAT 產生可重現結果。

**Architecture:** `full_coverage_successor.py` 提供 CPU profile、候選 policy 與 phase supervisor；既有 `Full35MixedPolicySearchPlan` 負責八指標 GPU PTQ；既有 QAT runtime 增加 hash-pinned external-control continuation seam。所有產物獨立於舊 V30/V35 artifact，避免原地覆寫。

**Tech Stack:** Python 3.12、PyTorch、PyYAML、pytest、既有 Full35 validator/QAT runtime。

**Spec:** `docs/superpowers/specs/2026-09-06-v36-full-coverage-design.md`

## Global Constraints

- activation 固定 qSiLU+A8；不使用 poly_quality。
- 148 個 deployment paths 全部至少 W8；BBAT5 v1 assignment 不變。
- mAP50 最差 total drop ≤ 0.015；mAP50-95 最差 total drop ≤ 0.04。
- GPU queue poll interval 固定 600 秒；短 QAT 最多 3 epochs、patience 5、無新增雜訊。

### Task 1: CPU matrix and candidate policy

**Files:**
- Modify: `src/yolo_quantize/full_coverage_successor.py`
- Test: `tests/test_full_coverage_successor.py`
- Create: `artifacts/manifests/v35-qsilu-mixed-11path-epoch3-locked-parent-v1.yaml`

- [x] Add the deployment-only 148×9 analyzer plan and strict per-path summary.
- [ ] Add parent loader and immutable CPU profile writer.
- [ ] Add route cohort selection and full-coverage policy materialization.
- [ ] Verify CPU profile coverage and parent hashes.

### Task 2: Phase PTQ queue

**Files:**
- Modify: `src/yolo_quantize/full_coverage_successor.py`
- Test: `tests/test_full_coverage_successor.py`
- Create: `artifacts/queues/v36-qsilu-full-coverage-progressive-v1/`

- [ ] Materialize special and uniform plans from the V35 base plan.
- [ ] Run/resume reports in backbone→neck→head order and select only green/recover routes.
- [ ] Materialize and validate final complete-policy PTQ.
- [ ] Expose compact status fields for the blocking monitor.

### Task 3: External-control continuation QAT

**Files:**
- Modify: `src/yolo_quantize/qat_plan.py`
- Modify: `src/yolo_quantize/qat_runtime.py`
- Modify: `src/yolo_quantize/full_coverage_successor.py`
- Test: `tests/test_qat_plan.py`, `tests/test_qat_runtime.py`, `tests/test_full_coverage_successor.py`

- [ ] Parse an explicit external V35 control reference for a `qat`-only continuation.
- [ ] Skip only the plan-local sham assertion when the external reference is hash-valid.
- [ ] Preserve accepted total gates and report unpaired status.
- [ ] Queue at most one 3-epoch continuation after final PTQ is green/recover.

### Task 4: Verification and worklog

**Files:**
- Create: `docs/worklogs/2026-09-06-v36-full-coverage-search.md`
- Modify: `docs/worklogs/README.md`

- [ ] Run focused tests, full relevant tests, ruff, and diff check.
- [ ] Run CPU profile/preflight before GPU.
- [ ] Start queue only after artifacts are hash-pinned; then enter 600-second blocking monitor.
