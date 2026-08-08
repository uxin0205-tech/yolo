# MASF-YOLO Phase 1 BEST_PARTIAL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a fail-closed, one-command YOLO11m P2/MFAM experiment pipeline that audits BBT5, trains B1 and M0–M3, selects one BEST_PARTIAL using validation only, evaluates it on test, profiles every variant, and emits reproducible artifacts.

**Architecture:** A repository-local `masf_yolo` package owns immutable experiment contracts, dataset grouping and split generation, P2/MFAM model construction, training/evaluation adapters, strict checkpoints, and a dependency-DAG workflow. The CLI is the only operator interface; a user systemd transient service runs the workflow independently of an interactive Codex process. Generated datasets, checkpoints, logs, and reports stay under ignored work directories.

**Tech Stack:** Python 3.14, PyTorch 2.11.0+cu128, Ultralytics 8.4.90, faster-coco-eval 1.7.2, OpenCV 5.0.0.93, PyYAML 6.0.3, pytest 9.1.1, NVIDIA RTX 4080 SUPER.

## Global Constraints

- Preserve `MFAM_plan.md` as the research source and implement Phase 1 only: `B1 → M0/M1/M2/M3 → BEST_PARTIAL`.
- Use only `../original/weight/yolo11m.pt` for official pretrained initialization; never overwrite it and never initialize formal runs from pose-derived BBT5 weights.
- Pin and verify Ultralytics `8.4.90`, faster-coco-eval `1.7.2`, the official model/default YAML SHA-256 values, PyTorch/CUDA versions, and source-weight SHA-256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`.
- Keep `imgsz=640`, `seed=42`, `deterministic=True`, `amp=True`, optimizer `SGD`, momentum `0.937`, cosine policy, and `nbs=64`; formal variants use one common probed batch size.
- B1-A trains 10 epochs with official backbone indices `0–10` frozen and `lr0=0.01`; B1-B trains 90 epochs unfrozen with `lr0=0.001`; M0–M3 each run a 3-epoch smoke and a 100-epoch formal run.
- Formal B1-B and M0–M3 resolved arguments may differ only in model path, output path/name, and epochs.
- All selection decisions use validation data. Freeze `selection.json` before any test evaluation.
- Strict checkpoint reuse requires matching data, variant, config, environment, and checkpoint hashes plus fresh-process strict reload validation.
- Do not modify Ultralytics installation files or global registries. Install MFAM slots in the model before the first forward pass.
- Keep `bbt5-detect-baseline/` and `field_check/` outside this work's staging set. Do not push before formal training preflight; the single requested commit message is `4080 hplab` on `main`.
- Never delete source data or weights automatically. Generated training artifacts remain ignored and are not pushed.

---

### Task 1: Repository contracts, documentation, and configuration

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `OPERATIONS.md`
- Create: `docs/design/phase1-best-partial.md`
- Create: `configs/static-phase1.yaml`
- Create: `.gitignore`
- Create: `masf_yolo/__init__.py`
- Create: `masf_yolo/contracts.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces: `canonical_json(value: object) -> bytes`, `sha256_file(path: Path) -> str`, `sha256_value(value: object) -> str`, and frozen Pydantic-free dataclasses used by every later task.
- Produces: one documented CLI contract for `audit`, `verify`, `pipeline start/status/execute`, and `report`.

- [x] **Step 1: Write contract tests first**

  Add tests proving canonical hashes ignore mapping insertion order, file hashes change with content, invalid split ratios fail, and serialized manifests round-trip without losing tuple/list or path values.

- [x] **Step 2: Verify the tests fail for missing package code**

  Run `../.venv/bin/python -m pytest tests/test_contracts.py -q` and confirm failure is caused by absent `masf_yolo.contracts` behavior.

- [x] **Step 3: Implement immutable contracts and exact config schema**

  Define `EnvironmentManifest`, `DatasetManifest`, `CheckpointManifest`, and `PipelineState`. Reject unknown config keys, non-80/10/10 ratios, class maps other than `{0: "ball", 1: "bat"}`, non-640 image sizes, and unsupported Phase 2 variants.

- [x] **Step 4: Write operator and design documents**

  Document artifact layout, preflight failures, systemd lifecycle, retry classes, resume rules, status/report commands, and the explicit boundary that Phase 1 never launches S0 or Temporal Motion.

- [x] **Step 5: Run contract tests**

  Run `../.venv/bin/python -m pytest tests/test_contracts.py -q` and retain the passing output in the implementation log.

### Task 2: Variant definitions and MFAM blocks

**Files:**
- Create: `masf_yolo/variants.py`
- Create: `masf_yolo/models/__init__.py`
- Create: `masf_yolo/models/mfam.py`
- Test: `tests/models/test_mfam.py`
- Test: `tests/test_variants.py`

**Interfaces:**
- Produces: `VariantDefinition`, `VARIANTS`, `get_variant(variant_id: str) -> VariantDefinition`, `MFAM`, and `PartialMFAM`.
- `VariantDefinition` fields: `variant_id`, `kernel_branches`, `processed_ratio`, `p2_slot`, `p3_slot`, and derived `config_hash`.

- [x] **Step 1: Write failing numeric, shape, and gradient tests**

  Use hand-set convolution/BatchNorm parameters to prove M0 sums identity plus 3/5/7/9 branches before 1×1 fusion; prove M1 omits only 7/9; prove M2 and M3 process exactly the leading half/quarter channels and concatenate untouched bypass channels; verify nonzero gradients reach every processed branch and input channel.

- [x] **Step 2: Verify red state**

  Run `../.venv/bin/python -m pytest tests/models/test_mfam.py tests/test_variants.py -q` and confirm missing classes/functions cause the expected failures.

- [x] **Step 3: Implement blocks from Ultralytics primitives**

  Compose Ultralytics `Conv` modules so each spatial branch is depthwise Conv+BN+SiLU, rectangular 7/9 paths are `1×k` then `k×1`, and the full block ends with one 1×1 Conv. Partial blocks add no shuffle, gate, or bypass projection.

- [x] **Step 4: Implement frozen variant registry and hash contract**

  Register only B1, M0, M1, M2, and M3. B1 uses identity slots; M0 uses 3/5/7/9 at P2; M1 uses 3/5 at P2; M2/M3 wrap M1 at ratios 1/2 and 1/4.

- [x] **Step 5: Run MFAM and variant tests**

  Run `../.venv/bin/python -m pytest tests/models/test_mfam.py tests/test_variants.py -q`.

### Task 3: Leakage-safe dataset audit and deterministic manifests

**Files:**
- Create: `masf_yolo/data/__init__.py`
- Create: `masf_yolo/data/labels.py`
- Create: `masf_yolo/data/grouping.py`
- Create: `masf_yolo/data/split.py`
- Create: `masf_yolo/data/export.py`
- Create: `masf_yolo/data/audit.py`
- Test: `tests/data/test_labels.py`
- Test: `tests/data/test_grouping.py`
- Test: `tests/data/test_split.py`
- Test: `tests/data/test_export.py`

**Interfaces:**
- Consumes: source `bbt5-detect-baseline/dataset/{train,valid}` but writes only into the configured ignored artifact root.
- Produces: `audit_dataset(config) -> DatasetManifest`, YOLO image lists, `data.yaml`, `dataset_profile.json`, `audit.json`, and COCO JSON for val/test in the exact 640-letterbox coordinate space.

- [x] **Step 1: Write malformed-label and grouping tests**

  Cover missing labels, wrong field counts, non-finite/out-of-range values, unsupported classes, Roboflow `.rf.<hash>` normalization, source/video stem extraction, exact-image duplicate hashing, and augmented siblings that must share one union-find group.

- [x] **Step 2: Verify label/group tests fail**

  Run `../.venv/bin/python -m pytest tests/data/test_labels.py tests/data/test_grouping.py -q`.

- [x] **Step 3: Implement fail-closed indexing and union grouping**

  Resolve symlinked images, hash decoded source bytes, parse YOLO boxes, derive normalized original stems and sequence names, and union records that share any source identity or content hash.

- [x] **Step 4: Write deterministic split tests**

  Use literal synthetic groups to prove seed-42 reproducibility, 80/10/10 objective behavior by unique frame and class/size-bin counts, zero group/hash overlap, both classes in every split, and hard failure when val/test cannot each contain 50 balls.

- [x] **Step 5: Implement deterministic greedy assignment and gates**

  Sort groups by rarity/size with a seed-derived stable tie key, greedily minimize normalized deviation from target totals, and run an explicit post-assignment audit before writing output.

- [x] **Step 6: Write and implement export tests**

  Assert that non-square images receive correct 640-letterbox offsets/scales, COCO boxes/areas match hand-derived coordinates, tiny/small/medium/large bins have exact counts, and all output paths remain beneath the artifact root.

- [x] **Step 7: Run all dataset tests and real audit**

  Run `../.venv/bin/python -m pytest tests/data -q`, then `../.venv/bin/python -m masf_yolo.cli audit --config configs/static-phase1.yaml`. Stop before GPU work if the real audit fails any research gate.

### Task 4: P2 template, slot installation, and explicit official-weight mapping

**Files:**
- Create: `configs/models/yolo11m-p2-slots.yaml`
- Create: `masf_yolo/models/builder.py`
- Create: `masf_yolo/models/transfer.py`
- Create: `masf_yolo/models/inspect.py`
- Test: `tests/models/test_builder.py`
- Test: `tests/models/test_transfer.py`

**Interfaces:**
- Produces: `build_model(variant, source_weights=None, checkpoint=None)`, `compare_official_structure(model)`, `transfer_official_weights(model, source) -> TransferReport`, and `transfer_b1_canonical(model, source) -> TransferReport`.
- Detect inputs are `[P2, P3, P4, P5]` with strides `[4, 8, 16, 32]`; slots are installed before any forward.

- [x] **Step 1: Write failing P2 structure and slot tests**

  Assert four Detect inputs, literal stride order, B1 identity behavior, MFAM module presence before first forward, no mutation of Ultralytics globals, and structure diff limited to the P2 top-down/bottom-up path, slots, four-scale Detect, and `nc=2`.

- [x] **Step 2: Verify builder tests fail**

  Run `../.venv/bin/python -m pytest tests/models/test_builder.py -q`.

- [x] **Step 3: Implement the fixed P2-slot YAML and builder**

  Preserve official backbone layers 0–10 and shared P3–P5 neck semantics. Add the P2 path and use a repository-owned placeholder module that is replaced immediately after YAML parsing, before stride initialization or forward.

- [x] **Step 4: Write failing transfer-map tests**

  Prove official backbone/shared neck and P3–P5 Detect tensors map explicitly by semantic role and shape; P2 tensors remain random; every matched, missing, unexpected, and shape-mismatch key appears in a JSON-serializable report. Prove M0–M3 load the same B1-B canonical EMA except their own MFAM tensors.

- [x] **Step 5: Implement semantic weight transfer and reports**

  Never accept a silent partial `load_state_dict`. Save source/destination names, shapes, hashes, and transfer disposition for every tensor.

- [x] **Step 6: Run builder/transfer tests and CPU forwards**

  Run `../.venv/bin/python -m pytest tests/models -q`, then build every variant and run a 640×640 no-grad CPU forward to inspect all four output scales.

### Task 5: Artifact schemas and strict checkpoint gate

**Files:**
- Create: `masf_yolo/artifacts/__init__.py`
- Create: `masf_yolo/artifacts/io.py`
- Create: `masf_yolo/artifacts/checkpoints.py`
- Create: `masf_yolo/artifacts/state.py`
- Create: `masf_yolo/artifacts/strict_reload.py`
- Test: `tests/artifacts/test_io.py`
- Test: `tests/artifacts/test_checkpoints.py`
- Test: `tests/artifacts/test_state.py`

**Interfaces:**
- Produces: atomic JSON writes, advisory pipeline lock, strict float32 EMA state-dict checkpoints, hash-validated stage reuse, and a subprocess reload command with machine-readable output.

- [x] **Step 1: Write failing atomicity, lock, and hash tests**

  Prove interrupted temp files never replace last-good state, two owners cannot hold one pipeline lock, completed stages reject changed data/config/variant/checkpoint hashes, and stale running state is distinguishable from completed state.

- [x] **Step 2: Verify artifact tests fail**

  Run `../.venv/bin/python -m pytest tests/artifacts -q`.

- [x] **Step 3: Implement atomic artifacts and state transitions**

  Use same-directory temporary files, `fsync`, atomic replace, explicit stage attempts, failure classification, epoch, timestamps, and all input/output hashes.

- [x] **Step 4: Write failing fresh-process reload tests**

  Save a small real variant checkpoint, launch a new Python subprocess, rebuild by variant/config, strict-load every tensor, assert float32 tensors, strides `[4,8,16,32]`, four Detect inputs, and one validation call. Corrupt each manifest hash separately and prove rejection.

- [x] **Step 5: Implement canonical EMA export and reload gate**

  Extract EMA model tensors without serializing live module objects, convert floating tensors to CPU float32, hash the state dict deterministically, and require the subprocess gate before marking a training stage complete.

- [x] **Step 6: Run artifact tests**

  Run `../.venv/bin/python -m pytest tests/artifacts -q`.

### Task 6: Training profiles, batch probe, and Ultralytics adapter

**Files:**
- Create: `masf_yolo/training/__init__.py`
- Create: `masf_yolo/training/profiles.py`
- Create: `masf_yolo/training/preflight.py`
- Create: `masf_yolo/training/runner.py`
- Create: `masf_yolo/training/resume.py`
- Test: `tests/training/test_profiles.py`
- Test: `tests/training/test_preflight.py`
- Test: `tests/training/test_resume.py`

**Interfaces:**
- Produces: resolved B1-A/B, smoke, and formal profiles; common-batch probe over `16,8,4,2,1`; real forward/backward finite-loss preflight; native `last.pt` resume with at most three transient attempts.

- [x] **Step 1: Write failing resolved-profile equivalence tests**

  Resolve against the pinned Ultralytics defaults and compare every key. Assert B1-B/M0–M3 differ only by model/output/name/epochs; assert exact freeze, LR, optimizer, momentum, cosine, seed, AMP, image size, and `nbs` values.

- [x] **Step 2: Verify profile tests fail**

  Run `../.venv/bin/python -m pytest tests/training/test_profiles.py -q`.

- [x] **Step 3: Implement profile resolution and immutable run manifests**

  Store the complete resolved argument mapping and its hash before calling Ultralytics. Refuse unknown overrides and any post-lock mutation.

- [x] **Step 4: Write failing preflight and retry tests**

  Exercise real small-tensor model loss where feasible, classify CUDA OOM/process interruption as transient, classify NaN/hash/data/architecture/research-gate failures as permanent, prove batch selection requires all five variants, and prove attempt four is never launched.

- [x] **Step 5: Implement trainer adapter and resume behavior**

  Keep native `last.pt` only for same-stage resume. After each run, export best EMA through the canonical checkpoint layer and invoke strict reload in a new subprocess.

- [x] **Step 6: Run training tests**

  Run `../.venv/bin/python -m pytest tests/training -q`.

### Task 7: Evaluation metrics, BEST_PARTIAL selection, and profiling

**Files:**
- Create: `masf_yolo/evaluation/__init__.py`
- Create: `masf_yolo/evaluation/metrics.py`
- Create: `masf_yolo/evaluation/selection.py`
- Create: `masf_yolo/evaluation/profiling.py`
- Create: `masf_yolo/evaluation/galleries.py`
- Test: `tests/evaluation/test_metrics.py`
- Test: `tests/evaluation/test_selection.py`
- Test: `tests/evaluation/test_profiling.py`

**Interfaces:**
- Produces: val/test metric JSON, immutable `selection.json`, BEST_PARTIAL ranking, params/MACs/GFLOPs/activation/operator/traffic profiles, and false-positive galleries.

- [x] **Step 1: Write failing metric aggregation tests**

  Use literal detections/ground truth to verify mAP fields, per-class AP, Ball AP/Recall/GT counts, COCO AP_S/M/L, baseball `<8`, `8–16`, `>16 px` bins, aspect-ratio `>2` blur proxy, and `null` metrics with retained counts when a subset lacks support.

- [x] **Step 2: Verify metric tests fail**

  Run `../.venv/bin/python -m pytest tests/evaluation/test_metrics.py -q`.

- [x] **Step 3: Implement Ultralytics/faster-coco-eval adapters**

  Keep raw predictions, COCO inputs, evaluator version, confidence/NMS settings, and subset match counts alongside derived metrics.

- [x] **Step 4: Write failing selection-boundary tests**

  Prove M3 wins when `abs(M2−M3 mAP50-95) <= 0.002` and Ball Recall difference is strictly `<0.01`; prove equality at `0.01` uses ranking; prove ranking order is mAP50-95, AP_S, Ball Recall, Ball AP_S, tiny recall, blur recall, then GFLOPs, params, peak activation, traffic.

- [x] **Step 5: Implement immutable selection and test-order guard**

  Refuse any test evaluation until a hash-complete val-only selection file exists. Refuse rewriting a selection with different content for the same pipeline ID.

- [x] **Step 6: Write and implement profiling tests**

  Verify parameter and operator counts on hand-sized blocks, distinguish MACs from GFLOPs, measure P2 activation, estimate peak live activation and feature traffic from traced tensor lifetimes, and count depthwise/1×1 operators.

- [x] **Step 7: Run evaluation tests**

  Run `../.venv/bin/python -m pytest tests/evaluation -q`.

### Task 8: Dependency-DAG workflow and CLI/systemd lifecycle

**Files:**
- Create: `masf_yolo/workflow.py`
- Create: `masf_yolo/cli.py`
- Create: `masf_yolo/reporting.py`
- Test: `tests/test_workflow.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: the sole ordered DAG `audit → verify → preflight → batch_probe → B1-A → B1-B → smoke(M0..M3) → formal(M0..M3) → val → selection → test → profile → final_audit → report`.
- CLI commands: `audit`, `verify`, `pipeline start`, `pipeline execute`, `pipeline status`, and `report`.

- [x] **Step 1: Write failing DAG, idempotence, and order tests**

  Prove dependency order, valid-stage reuse, invalid-hash rebuild, a single lock owner, test-after-selection, three-attempt limit, permanent failure stop, and that Phase 1 contains no S0/TM node.

- [x] **Step 2: Verify workflow tests fail**

  Run `../.venv/bin/python -m pytest tests/test_workflow.py tests/test_cli.py -q`.

- [x] **Step 3: Implement the orchestrator and reporting**

  Make every node declare input hashes, outputs, dependencies, transient policy, and success gate. Final success requires all canonical artifacts, one BEST_PARTIAL, complete test/profile data, `final_audit.ok=true`, and zero audit errors.

- [x] **Step 4: Implement one-shot systemd start**

  `pipeline start` validates clean tracked worktree, CUDA device 0, pinned environment, dataset audit, systemd user bus, and absence of a live/complete duplicate. It then launches one transient user service whose `ExecStart` calls `pipeline execute` with absolute paths and restart behavior bounded by workflow state.

- [x] **Step 5: Implement status and report rebuild**

  Status reads state/service information without mutating the run. Report can be regenerated from immutable artifacts and includes environment, data audit, transfers, training runs, val selection, test metrics, profiles, failures, and final audit.

- [x] **Step 6: Run workflow/CLI tests**

  Run `../.venv/bin/python -m pytest tests/test_workflow.py tests/test_cli.py -q`.

### Task 9: Full verification, cleanup, requested commit, and formal launch

**Files:**
- Modify: checkbox states in this plan as each task completes
- Generate (ignored): configured pipeline artifact root

**Interfaces:**
- Consumes every prior task and produces the only Git commit plus one running background service.

- [x] **Step 1: Run the complete CPU test suite**

  Run `../.venv/bin/python -m pytest -q`. Record exact pass/fail/skip counts and retain failure output if nonzero.

- [x] **Step 2: Run static and import verification**

  Run `../.venv/bin/python -m compileall -q masf_yolo tests` and a CLI help/import smoke. Verify tracked files contain no generated datasets, runs, logs, or checkpoints.

- [ ] **Step 3: Run real audit and GPU acceptance preflight**

  Run audit against the real BBT5 union, then outside the restricted sandbox run CUDA checks, all five real forward/backward finite-loss/save-reload gates, common batch probe, and each 3-epoch engineering smoke. Do not start the formal matrix unless every gate passes.

- [x] **Step 4: Remove validation-only generated artifacts**

  First enumerate cleanup targets. Remove only this work's `.pytest_cache`,
  `__pycache__`, temporary audit/preflight/smoke artifacts, test runs, and
  incomplete test checkpoints. Preserve source datasets, original weights, and
  any canonical checkpoint required for formal resume. Re-run the full tests if
  cleanup removes anything unexpectedly referenced by the implementation.

- [x] **Step 5: Review exact staging scope**

  Run `git status --short`, `git diff --check`, and `git diff --cached --name-only`. Stage only new/modified Phase 1 files under `yolo_masf/`, explicitly excluding `bbt5-detect-baseline/`, `field_check/`, generated artifacts, datasets, logs, runs, and large checkpoints.

- [ ] **Step 6: Commit on main without pushing**

  Create the single requested commit with `git commit -m "4080 hplab"`. Verify the commit file list and keep it local until the formal launch decision; do not push in this task unless the user separately requests it.

- [ ] **Step 7: Start the formal background pipeline once**

  Run `../.venv/bin/python -m masf_yolo.cli pipeline start --config configs/static-phase1.yaml`. Capture pipeline ID, systemd unit, active state, main PID, state path, log path, and first running stage.

- [ ] **Step 8: Notify the user immediately after verified launch**

  Report that training has started only after both `systemctl --user is-active <unit>` and pipeline status show a live process. Include the exact status/report commands and explain that closing Codex does not stop the user service.

- [ ] **Step 9: Verify terminal Phase 1 artifacts when the service finishes**

  Confirm B1-A/B and M0–M3 canonical checkpoints, frozen selection, complete tests/profiles, one BEST_PARTIAL, strict-reload evidence, `final_audit.ok=true`, zero audit errors, and the human-readable report. Phase 1 stops normally without S0 or Temporal Motion.

## Self-Review Record

- Every Phase 1 requirement in `MFAM_plan.md` and the approved handoff maps to Tasks 1–9.
- Public interfaces use one spelling across producers and consumers.
- Generated data and model artifacts are excluded from Git; source data and source weights are read-only.
- The plan contains no unresolved implementation choices: thresholds, hashes, versions, stage order, retry limits, training epochs, selection order, and commit timing are explicit.
- User overrides are honored: inline execution, work on `main`, no early push, commit message `4080 hplab`, exclude baseline/field directories, and notify after verified training start.
