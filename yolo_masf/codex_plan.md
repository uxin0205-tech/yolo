# MASF-YOLO Phase 1 BEST_PARTIAL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a fail-closed, one-command YOLO11m P2/MFAM experiment pipeline that audits BBT5, trains B1, priority M7, and M0–M3, measures the pose-derived B0 detection reference, selects one BEST_PARTIAL using validation only, evaluates it on test, profiles every model, and emits reproducible artifacts.

**Architecture:** A repository-local `masf_yolo` package owns immutable experiment contracts, dataset grouping and split generation, P2/MFAM model construction, training/evaluation adapters, strict checkpoints, and a dependency-DAG workflow. The CLI is the only operator interface; a user systemd transient service runs the workflow independently of an interactive Codex process. Generated datasets, checkpoints, logs, and reports stay under ignored work directories.

**Tech Stack:** Python 3.14, PyTorch 2.11.0+cu128, Ultralytics 8.4.90, faster-coco-eval 1.7.2, OpenCV 5.0.0.93, PyYAML 6.0.3, pytest 9.1.1, NVIDIA RTX 4080 SUPER.

## Global Constraints

- Preserve `MFAM_plan.md` as the research source and implement Phase 1 only: `B1 → priority M7 → M0/M1/M2/M3 → BEST_PARTIAL` plus report-only B0.
- Use only `../original/weight/yolo11m.pt` for official pretrained initialization; never overwrite it and never initialize formal runs from pose-derived BBT5 weights.
- Pin and verify Ultralytics `8.4.90`, faster-coco-eval `1.7.2`, the official model/default YAML SHA-256 values, PyTorch/CUDA versions, and source-weight SHA-256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`.
- Keep `imgsz=640`, `seed=42`, `deterministic=True`, `amp=True`, optimizer `SGD`, momentum `0.937`, cosine policy, and `nbs=64`; formal variants use one common probed batch size.
- B1-A trains 10 epochs with official backbone indices `0–10` frozen and `lr0=0.01`; B1-B trains 90 epochs unfrozen with `lr0=0.001`; M7 and M0–M3 each run a 3-epoch smoke and a 100-epoch formal run.
- Formal B1-B, M7, and M0–M3 resolved arguments may differ only in model path, output path/name, and epochs.
- B0 is the exact pose-derived detection checkpoint with SHA-256 `9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d`; label it data-exposed, evaluate/profile it, and never use it for initialization or selection.
- All selection decisions use validation data. Freeze `selection.json` before any test evaluation.
- Strict checkpoint reuse requires matching data, variant, config, environment, and checkpoint hashes plus fresh-process strict reload validation.
- Do not modify Ultralytics installation files or global registries. Install MFAM slots in the model before the first forward pass.
- Keep `bbt5-detect-baseline/` and `field_check/` outside this work's staging set. Do not push before formal training preflight; the single requested commit message is `4080 hplab` on `main`.
- Never delete source data or weights automatically. Generated training artifacts remain ignored and are not pushed.

---

## Completed Original Implementation (Tasks 1–9)

Tasks 1–9 record the original five-variant implementation and its completed red/green cycles. Tasks 10–15 are the approved amendment and supersede the original registry, matrix, recovery, launch, and terminal-acceptance wording wherever they differ.

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

  Register the original set B1, M0, M1, M2, and M3. B1 uses identity slots; M0 uses 3/5/7/9 at P2; M1 uses 3/5 at P2; M2/M3 wrap M1 at ratios 1/2 and 1/4. Task 10 extends this registry with M7.

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

> Outstanding launch and terminal steps in this original task are superseded by Task 15, which adds completed-run recovery, M7, and B0 acceptance.

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

  Confirm B1-A/B and M7/M0–M3 canonical checkpoints, B0 reference evidence, frozen selection, complete tests/profiles, one BEST_PARTIAL, strict-reload evidence, `final_audit.ok=true`, zero audit errors, and the human-readable report. Phase 1 stops normally without S0 or Temporal Motion.

### Task 10: M7 priority variant and non-invalidating extension contract

**Files:**
- Modify: `masf_yolo/variants.py`
- Modify: `masf_yolo/models/builder.py`
- Create: `configs/m7-priority.yaml`
- Modify: `tests/test_variants.py`
- Modify: `tests/models/test_mfam.py`
- Modify: `tests/models/test_builder.py`
- Modify: `tests/models/test_transfer.py`

**Interfaces:**
- Produces: `M7 = VariantDefinition("M7", (3, 5, 7), 1.0, "mfam")`.
- Keeps: the existing `configs/static-phase1.yaml` bytes and config hash unchanged so completed B1 stages remain reusable.
- Produces constants that separate `CORE_VARIANTS`, `PRIORITY_VARIANTS`, `TRAINED_VARIANTS`, `EVALUATED_MODELS`, and `SELECTION_CANDIDATES` rather than overloading one tuple.

- [x] **Step 1: Write the failing M7 registry and numerical tests**

  Assert `get_variant("M7")` has kernels `(3, 5, 7)`, ratio `1.0`, P2 `mfam`, and a stable hash distinct from M0/M1. Hand-set Conv/BN parameters and prove the forward value equals identity plus direct 3/5 branches plus the sequential 1x7/7x1 branch before 1x1 fusion. Backpropagate a scalar loss and assert every M7 branch receives a finite nonzero gradient.

- [x] **Step 2: Run the focused tests and observe RED**

  Run `../.venv/bin/python -m pytest tests/test_variants.py tests/models/test_mfam.py -q`. Expected failure: `unsupported variant: M7` or missing M7 registry entry.

- [x] **Step 3: Implement the minimal M7 definition**

  Add `M7` without changing M0/M1/M2/M3 semantics. Reuse `MFAM(channels, kernels=(3, 5, 7))`; do not introduce a new operator implementation, shuffle, gate, bypass, or projection.

- [x] **Step 4: Add builder and transfer ownership tests**

  Build M7 and assert strides `[4, 8, 16, 32]`, four Detect inputs, and M7 in the P2 slot before forward. Transfer the B1 canonical state and prove shared tensors match B1 while all `model.20` M7 tensors retain their pre-transfer random values and appear as missing/variant-owned tensors in the report.

- [x] **Step 5: Add and verify the extension manifest**

  Write `configs/m7-priority.yaml` with `variant_id: M7`, kernels `[3, 5, 7]`, processed ratio `1.0`, smoke epochs `3`, formal epochs `100`, and `priority_before: M0`. Test strict unknown-key rejection and verify `sha256sum configs/static-phase1.yaml` is unchanged from the running pipeline's config hash input.

- [x] **Step 6: Run the M7 model suite**

  Run `../.venv/bin/python -m pytest tests/test_variants.py tests/models -q` and one real CUDA forward/backward/save/reload for M7 at batch 16 before any M7 training.

### Task 11: Completed-run classification and B1-B recovery

**Files:**
- Create: `masf_yolo/training/completion.py`
- Modify: `masf_yolo/training/resume.py`
- Modify: `masf_yolo/pipeline.py`
- Create: `tests/training/test_completion.py`
- Modify: `tests/training/test_resume.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `TrainingOutputState(status, completed_epochs, expected_epochs, best, last, results_hash, best_hash, last_hash, reason)`.
- Produces: `classify_training_output(run_dir: Path, expected_epochs: int) -> TrainingOutputState` with statuses `absent`, `incomplete`, `complete`, or `invalid`.
- A complete output skips Ultralytics training and enters canonical finalization; only an incomplete output may use native resume.

- [x] **Step 1: Write literal absent/incomplete/complete/invalid tests**

  Create real small Ultralytics-style checkpoint dictionaries with `train_args["epochs"]`, finite CSV rows, and loadable model payloads. Assert absent has no last file; incomplete has 1 of 3 rows; complete has exactly 3 rows plus best/last; invalid covers 4 of 3 rows, non-finite final values, mismatched checkpoint epochs, unreadable checkpoint, missing best, and contradictory CSV/checkpoint state.

- [x] **Step 2: Run completion tests and observe RED**

  Run `../.venv/bin/python -m pytest tests/training/test_completion.py -q`. Expected failure: missing `masf_yolo.training.completion`.

- [x] **Step 3: Implement fail-closed classification**

  Parse `results.csv` with `csv.DictReader`, require the literal `epoch` column to be consecutive `1..N`, require all metric/loss/LR cells finite, load checkpoints on CPU, compare `train_args["epochs"]`, and SHA-256 all accepted files. Never infer completion from file existence alone and never mutate the run directory.

- [x] **Step 4: Write orchestration tests for skip versus resume**

  Given a completed 90-row B1-B fixture, assert `_initial_model`, `run_training`, and resume are never called and the next action is finalization. Given an incomplete fixture, assert the exact `last.pt` is passed once as native resume. Given invalid artifacts, assert `PermanentTrainingError` before a GPU model is built.

- [x] **Step 5: Implement classification before model construction**

  Refactor `_train` so it classifies output first. Return the existing best/last for `complete`; call the worker with no resume for `absent`; call it with the exact last path for `incomplete`; reject `invalid`. Record the classification and all three file hashes in `training/<stage>/run.json`.

- [x] **Step 6: Prove the real B1-B output is recoverable**

  Run the classifier against `artifacts/static-phase1/runs/b1-b` and require `status=complete`, `completed_epochs=90`, unchanged 90-row results hash, and both checkpoint hashes. Record `wc -l results.csv` before and after finalization and require 91 lines including the header both times.

### Task 12: Memory-safe trainer and fresh-process canonical finalization

**Files:**
- Modify: `masf_yolo/training/runner.py`
- Create: `masf_yolo/training/worker.py`
- Create: `masf_yolo/artifacts/finalize.py`
- Modify: `masf_yolo/pipeline.py`
- Modify: `masf_yolo/cli.py`
- Modify: `masf_yolo/runtime.py`
- Create: `tests/training/test_runner.py`
- Create: `tests/training/test_worker.py`
- Create: `tests/artifacts/test_finalize.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- `RepositoryDetectionTrainer.final_eval()` strips optimizer state from last/best but deliberately omits Ultralytics' redundant final best validation.
- Produces internal commands `python -m masf_yolo.training.worker ...` and `python -m masf_yolo.artifacts.finalize ...`; they write atomic machine-readable result manifests.
- The externally documented CLI remains unchanged.

- [x] **Step 1: Write the failing final-eval regression test**

  Construct a trainer with temporary best/last checkpoints and a validator callable that raises if invoked. Call `final_eval()`, assert the validator call count stays zero, and assert both checkpoints are stripped/loadable with the final `train_results` retained.

- [x] **Step 2: Run runner regression and observe RED**

  Run `../.venv/bin/python -m pytest tests/training/test_runner.py -q`. Expected failure: inherited Ultralytics `final_eval` invokes the sentinel validator.

- [x] **Step 3: Override only duplicate final validation**

  Copy the pinned 8.4.90 optimizer-stripping portion using `torch_distributed_zero_first`, `LOCAL_RANK`, `RANK`, and `strip_optimizer`. Do not change per-epoch validation, checkpoint choice, optimizer, augmentation, or training arguments.

- [x] **Step 4: Write worker-boundary tests**

  Launch real subprocesses against tiny fixtures. Assert exit 0 yields a JSON result containing best/last paths and hashes; SIGKILL/nonzero exit becomes `TransientTrainingError`; malformed/hash-mismatched output becomes `PermanentTrainingError`; a completed run bypasses worker launch.

- [x] **Step 5: Implement isolated training and finalization workers**

  Move model construction and `RepositoryDetectionTrainer.train()` into the training worker. The parent retains only paths/hashes. In a second fresh process, load best EMA, attach `masf_variant` and its hash, call `save_canonical_checkpoint`, then exit before the existing strict-reload/validation subprocess runs.

- [x] **Step 6: Add systemd OOM containment**

  Make `build_systemd_command()` include `--property=OOMPolicy=continue` and memory accounting. Test the exact property and keep `Restart=on-failure`, three bounded starts, and the venv interpreter contract.

- [x] **Step 7: Run focused process tests**

  Run `../.venv/bin/python -m pytest tests/training tests/artifacts tests/test_cli.py -q`. Confirm no test leaves worker processes, semaphore warnings, temporary checkpoints, or live locks.

### Task 13: Priority DAG, M7 gate, and formal scheduling

**Files:**
- Modify: `masf_yolo/workflow.py`
- Modify: `masf_yolo/pipeline.py`
- Modify: `masf_yolo/training/preflight.py`
- Modify: `tests/test_workflow.py`
- Modify: `tests/training/test_preflight.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Produces exact order `b1_b -> m7_gate -> smoke_m7 -> formal_m7 -> smoke_m0 -> smoke_m1 -> smoke_m2 -> smoke_m3 -> formal_m0 -> formal_m1 -> formal_m2 -> formal_m3`.
- `m7_gate` consumes the existing common batch and proves M7 can perform real AMP forward/backward/SGD/save/reload at batch 16.

- [x] **Step 1: Write the failing DAG-order and reuse tests**

  Assert M7 gate/smoke/formal precede every M0 stage, `b1_b` still depends directly on `b1_a`, and no completed pre-M7 stage gains a new dependency. Seed completed B1-A plus failed-but-recoverable B1-B state and prove the workflow does not invoke audit, verify, preflight, batch probe, or B1-A actions again.

- [x] **Step 2: Run workflow tests and observe RED**

  Run `../.venv/bin/python -m pytest tests/test_workflow.py tests/test_pipeline.py -q`. Expected failure: missing M7 stage names/order.

- [x] **Step 3: Add M7 stages and handlers**

  Insert the three nodes without changing earlier node names or input hashes. Use the existing smoke/formal profile builders, B1 canonical transfer, training-output classifier, isolated worker, canonical finalizer, and strict reload gate.

- [x] **Step 4: Add the real M7 acceptance gate**

  Load the locked common batch (16), build M7 from B1-B canonical, perform finite AMP loss and one SGD optimizer step, save/reload, verify strides and four scales, and atomically write `m7_gate.json`. Fail permanently rather than silently lowering the shared batch.

- [x] **Step 5: Run DAG and real GPU gate**

  Run focused tests, then execute only the M7 gate on CUDA device 0. Record loss, peak GPU memory, checkpoint hash, variant hash, and strict-load outcome.

### Task 14: B0 reference inspection, common evaluation, and reporting

**Files:**
- Create: `configs/b0-reference.yaml`
- Create: `masf_yolo/evaluation/reference.py`
- Modify: `masf_yolo/pipeline.py`
- Modify: `masf_yolo/workflow.py`
- Modify: `masf_yolo/reporting.py`
- Create: `tests/evaluation/test_reference.py`
- Modify: `tests/evaluation/test_metrics.py`
- Modify: `tests/evaluation/test_profiling.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_reporting.py`
- Modify: `tests/test_workflow.py`

**Interfaces:**
- Produces `inspect_b0_reference(definition_path: Path, output_path: Path) -> ReferenceManifest` in a fresh subprocess.
- Evaluated models are `B0, B1, M7, M0, M1, M2, M3`; selection candidates remain exactly `M2, M3`.
- B0 checkpoint path resolves to `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt` and must hash to `9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d`.

- [x] **Step 1: Write failing reference-definition tests**

  Cover exact SHA-256, `ball=0/bat=1`, task detect, three scales, strides `[8,16,32]`, Ultralytics 8.4.90, pose-derived provenance, and `data_exposed=true`. Corrupt each field independently and require a fail-closed error.

- [x] **Step 2: Run reference tests and observe RED**

  Run `../.venv/bin/python -m pytest tests/evaluation/test_reference.py -q`. Expected failure: missing reference loader/manifest.

- [x] **Step 3: Implement B0 definition and fresh-process inspection**

  Load the checkpoint on CPU, record source metadata without trusting it for validation, inspect the actual model task/names/strides, run a 640 forward, hash the checkpoint, and atomically write `references/b0.json`. Never rewrite or copy the source checkpoint.

- [x] **Step 4: Add B0 to common val/test/profile loops**

  Route `_best_checkpoint("B0")` to the immutable reference weight. Run the same prediction export, Ultralytics metrics, faster-coco-eval, Ball AP/Recall/counts, size bins, blur proxy, FP gallery, params, MACs/GFLOPs, peak activation, operator counts, and feature traffic as other models.

- [x] **Step 5: Preserve selection isolation**

  Test that B0 and M7 can have arbitrarily better metrics without entering `CandidateMetrics`, `selection.json`, or changing the selected M2/M3 result. Require frozen selection before B0 test evaluation along with every other test evaluation.

- [x] **Step 6: Extend audit and report**

  Require B0 plus six trained models in val/test/profile artifacts. Add a visible report section: `B0 is pose-derived and data-exposed; its metrics are operational reference values, not a leak-free comparison.` Include checkpoint hash, provenance, val/test metrics, per-class/ball summaries, and hardware profile.

- [ ] **Step 7: Run B0 focused tests and real val**

  Run `../.venv/bin/python -m pytest tests/evaluation tests/test_pipeline.py tests/test_reporting.py -q`, then run B0 validation against the frozen union val manifest. Confirm 781 images, 1,099 instances, complete metrics JSON, and no mutation of selection or test artifacts.

### Task 15: Documentation, full verification, cleanup, commit, and safe restart

**Files:**
- Modify: `README.md`
- Modify: `OPERATIONS.md`
- Modify: `docs/design/phase1-best-partial.md`
- Modify: `codex_plan.md`
- Generate (ignored): B1-B recovery, M7, B0, and pipeline artifacts

**Interfaces:**
- Produces a clean tracked `main`, one pushed `4080 hplab` commit, and one live background service that advances from recovered B1-B to M7 without repeating B1.

- [ ] **Step 1: Update operator and research documentation**

  Document B0 provenance/exposure, M7 equation and priority, 615 total planned epochs, completed-run classification, no duplicate final validation, worker boundaries, B1-B recovery evidence, and exact status/log/report commands.

- [ ] **Step 2: Run the complete fresh verification suite**

  Run `../.venv/bin/python -m pytest -q`, `../.venv/bin/python -m compileall -q masf_yolo tests`, `git diff --check`, and CLI import/help smoke. Record exact pass/fail counts; any failure blocks commit and restart.

- [ ] **Step 3: Perform real recovery and B0/M7 gates before launch**

  Classify B1-B as complete, finalize canonical in a fresh subprocess, strict-load and fresh-val it, prove `results.csv` remains 90 epochs, inspect B0, and run M7 batch-16 GPU gate. Do not start the service until every artifact hash and gate passes.

- [ ] **Step 4: Clean generated validation garbage and review scope**

  Enumerate first, then remove only `.pytest_cache`, `__pycache__`, temporary test/audit/preflight outputs, the generated `bbt5-detect-baseline/dataset/train/labels.cache`, and incomplete worker fixtures. Preserve source data, B0 weights, B1-A/B valid runs, canonical checkpoints, and formal artifacts. Verify Git staging excludes `bbt5-detect-baseline/`, `field_check/`, datasets, caches, logs, runs, and checkpoints.

- [ ] **Step 5: Commit and push the focused implementation**

  Commit on `main` with exact subject `4080 hplab`, inspect the committed file list, push `origin main`, and verify local `HEAD == origin/main`. Never add the B0 checkpoint or generated artifacts.

- [ ] **Step 6: Restart once and prove no B1 retraining**

  Start the existing pipeline identity through `pipeline start`. Require B1-A and B1-B stage manifests to be reused/completed, B1-B CSV to remain 91 lines including header, and the live stage to reach `m7_gate`, `smoke_m7`, or `formal_m7`. Any attempt to launch B1 training is a permanent acceptance failure and the service must be stopped.

  Automated recovery coverage now seeds completed predecessors plus a stale
  `smoke_m7` state and proves only `smoke_m7` is invoked, its attempt increments,
  and all B1/preflight handlers remain unused. Real-service acceptance remains
  unchecked until launch.

- [ ] **Step 7: Notify and monitor terminal acceptance**

  Immediately report the verified M7 launch with unit/PID/stage and explain systemd persistence. At terminal completion require canonical B1-A/B/M7/M0–M3, B0 reference, all seven val/test/profile sets, immutable M2/M3 selection, `final_audit.ok=true`, zero errors, and a human-readable report that carries the B0 exposure warning.

## Self-Review Record

- Every Phase 1 requirement in `MFAM_plan.md`, the approved handoff, and the B0/M7 recovery amendment maps to Tasks 1–15.
- Public interfaces use one spelling across producers and consumers.
- Generated data and model artifacts are excluded from Git; source data and source weights are read-only.
- The plan contains no unresolved implementation choices: thresholds, hashes, versions, B0 exposure, M7 branches, stage order, recovery states, retry limits, training epochs, selection order, and commit timing are explicit.
- User overrides are honored: inline execution, work on `main`, no early push, commit message `4080 hplab`, exclude baseline/field directories, and notify after verified training start.
