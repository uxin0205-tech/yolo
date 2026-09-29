# Single-Worker Experiment Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persistent, fail-closed, single-worker queue that can prepare and later execute the complete YOLO26m experiment funnel one job at a time, while remaining CPU-only unless `--execute` is explicitly supplied.

**Architecture:** Keep persistence, workflow expansion/selection, execution, and CLI as separate modules. Queue state is immutable typed data serialized atomically to `artifacts/queue/queue.json`; pure policy functions decide winners from standardized result JSON; an executor with an injected backend owns the only path that may call training or evaluation.

**Tech Stack:** Python 3.10+, dataclasses/enums, JSON/JSONL, `fcntl` file locking on Linux, PyTorch/Ultralytics through existing package seams, pytest, Ruff.

## Global Constraints

- Do not start GPU training, validation, CUDA initialization, dataset download, or weight download during implementation or verification.
- One queue may contain at most one `queued` or one `running` job, never both.
- `queue init/status/validate/next/retry` are CPU-safe and never call the execution backend.
- `queue run-next` and `queue run` remain dry-run unless `--execute` is present.
- I/H/T5 use the same P0 model checkpoint even though scheduling dependencies serialize them.
- Missing checkpoints, metrics, profiles, YAML files, or required result fields fail closed.
- Optional quantization, ternary, and SD4 are not materialized or executed.
- Do not create Git commits: this project is an untracked shared directory in the parent repository and the user requested direct edits without commits.

---

## File Map

| File | Responsibility |
|---|---|
| `src/yolo_attention/queue_model.py` | Typed job/state/result schema and invariant validation |
| `src/yolo_attention/queue_store.py` | Atomic JSON persistence, JSONL events, exclusive worker lock |
| `src/yolo_attention/queue_policy.py` | Pure winner/gate functions; no filesystem or Ultralytics |
| `src/yolo_attention/queue_workflow.py` | Initial jobs, readiness refresh, winner-derived YAML and dynamic expansion |
| `src/yolo_attention/queue_executor.py` | Dry-run preview, injected backend dispatch, completion/failure transitions |
| `src/yolo_attention/evaluation.py` | Future evaluate/P0 backend seams and standardized result extraction |
| `configs/evaluation/coco2017.yaml` | Shared COCO evaluator resource settings; no training epochs/optimizer |
| `src/yolo_attention/cli.py` | `queue` command group only; no policy logic |
| `tests/test_queue_model_store.py` | Schema, persistence, event and lock contracts |
| `tests/test_queue_policy.py` | Every numeric selection/gate rule |
| `tests/test_queue_workflow.py` | Initial graph, readiness and dynamic expansion |
| `tests/test_queue_executor.py` | No-execute safety, one-worker execution and recovery |
| `tests/test_queue_cli.py` | CLI JSON and dry-run contracts |
| `artifacts/README.md` | Queue/run artifact layout and commit policy |

---

### Task 1: Typed Queue State and Atomic Store

**Files:**
- Create: `src/yolo_attention/queue_model.py`
- Create: `src/yolo_attention/queue_store.py`
- Create: `tests/test_queue_model_store.py`

**Interfaces:**
- Produces: `JobKind`, `JobStatus`, `QueueJob`, `QueueState`, `QueueResult`, `QueueValidationError`.
- Produces: `QueueStore(root: Path)`, `.initialize(state)`, `.load()`, `.save(state)`, `.append_event(...)`, `.worker_lock()`.
- Queue JSON schema version is exactly `1`.

- [ ] **Step 1: Write failing schema and round-trip tests**

```python
def test_queue_state_rejects_two_active_jobs() -> None:
    jobs = (
        QueueJob.minimal("a", status=JobStatus.QUEUED),
        QueueJob.minimal("b", status=JobStatus.RUNNING),
    )
    with pytest.raises(QueueValidationError, match="one active job"):
        QueueState(schema_version=1, jobs=jobs).validate()


def test_store_round_trip_and_refuses_overwrite(tmp_path: Path) -> None:
    store = QueueStore(tmp_path / "queue")
    state = QueueState.initial((QueueJob.minimal("b26-fp", status=JobStatus.READY),))
    store.initialize(state)
    assert store.load() == state
    with pytest.raises(FileExistsError):
        store.initialize(state)
```

- [ ] **Step 2: Run RED**

Run: `../.venv/bin/pytest -q tests/test_queue_model_store.py`
Expected: collection fails because `queue_model` and `queue_store` do not exist.

- [ ] **Step 3: Implement exact enums and dataclasses**

```python
class JobKind(StringEnum):
    VALIDATE = "validate"
    TRAIN = "train"
    EVALUATE = "evaluate"
    SELECT = "select"


class JobStatus(StringEnum):
    BLOCKED = "blocked"
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class QueueResult:
    map50_95: float | None = None
    row_sum_max_error: float | None = None
    checkpoint_path: str | None = None
    metrics_path: str | None = None
    profile_path: str | None = None
```

`QueueJob` includes every field from the accepted spec, including `evaluation_path`, plus `order: int`, `model_parent_job_id: str | None`, and `result: QueueResult | None`. `QueueState.validate()` rejects duplicate IDs/order, missing dependencies, cycles, invalid transitions encoded in current state, more than one active job, and non-lowercase IDs.

- [ ] **Step 4: Implement atomic persistence and non-blocking lock**

`QueueStore.save()` writes JSON to `queue.json.tmp`, flushes and `os.fsync()`s, then calls `os.replace()`. `worker_lock()` opens `worker.lock` and uses `fcntl.flock(fd, LOCK_EX | LOCK_NB)`; contention raises `QueueLockedError`. `append_event()` writes one sorted JSON object plus newline to `events.jsonl`.

- [ ] **Step 5: Verify GREEN**

Run: `../.venv/bin/pytest -q tests/test_queue_model_store.py`
Expected: all tests pass and a second lock acquisition fails immediately.

---

### Task 2: Pure Winner and Gate Policies

**Files:**
- Create: `src/yolo_attention/queue_policy.py`
- Create: `tests/test_queue_policy.py`

**Interfaces:**
- Consumes: `QueueJob.result.map50_95`, `row_sum_max_error`, and profile JSON.
- Produces: `SelectionDecision(winners: tuple[str, ...], skipped: tuple[str, ...], reason: str, expand: tuple[str, ...])`.
- Produces: `select_architecture`, `select_recovery`, `select_bias`, `select_n0`, `select_d1`, `select_r_denominator`.

- [ ] **Step 1: Write failing policy tests with boundary values**

```python
def test_recovery_tie_prefers_direct() -> None:
    decision = select_recovery({"w-dir": 0.4000, "w-prog": 0.4009})
    assert decision.winners == ("w-dir",)


def test_d1_tie_prefers_simpler_sharing() -> None:
    decision = select_d1({"d1-shared": 0.3900, "d1-pattn": 0.3908, "d1-phead": 0.3909})
    assert decision.winners == ("d1-shared",)


def test_r1_only_expands_newton_when_gate_is_missed() -> None:
    decision = select_r_denominator(
        r0_map=0.4000, r1_map=0.3979, r1_row_sum_max_error=0.005
    )
    assert decision.expand == ("r1-newton",)
```

Also test exact threshold behavior (`< 0.001` is tie; loss `<= 0.01` passes; R1 loss `> 0.002` or error `> 0.01` expands Newton), missing metrics, NaN, N0 maximum of two, and final selection by mAP band → `estimated_memory_traffic` → `arithmetic_cost_proxy`.

- [ ] **Step 2: Run RED**

Run: `../.venv/bin/pytest -q tests/test_queue_policy.py`
Expected: missing module failure.

- [ ] **Step 3: Implement policies without filesystem access**

Use explicit priority tuples:

```python
ARCH_COST_ORDER = ("i-scr", "t5-scr", "h-scr")
D1_COMPLEXITY_ORDER = ("d1-shared", "d1-pattn", "d1-phead")
```

All public policy functions call `_finite_metric_map()` and raise `SelectionInputError` for missing/non-finite values. Do not silently coerce strings to floats.

- [ ] **Step 4: Verify GREEN**

Run: `../.venv/bin/pytest -q tests/test_queue_policy.py`
Expected: all policy tests pass.

---

### Task 3: Initial Graph, Readiness, and Generated Configs

**Files:**
- Create: `src/yolo_attention/queue_workflow.py`
- Create: `tests/test_queue_workflow.py`
- Modify: `src/yolo_attention/config.py`
- Modify: `src/yolo_attention/run_config.py`

**Interfaces:**
- Consumes: `QueueState`, `SelectionDecision`, existing `VariantConfig`, `TrainingRecipe`.
- Produces: `create_initial_state(project_root: Path) -> QueueState`.
- Produces: `refresh_readiness(state: QueueState) -> QueueState` and `next_runnable_job(state) -> QueueJob | None`; the latter returns an existing queued retry first, otherwise the lowest-order ready job.
- Produces: `derive_variant(parent: VariantConfig, *, name: str, changes: dict[str, object])` and `materialize_after_selection(...)`.

- [ ] **Step 1: Write failing initial graph tests**

```python
def test_initial_queue_is_serial_and_excludes_quantization() -> None:
    state = create_initial_state(ROOT)
    assert [job.id for job in state.jobs] == [
        "b26-fp", "p0", "i-scr", "h-scr", "t5-scr", "architecture-select"
    ]
    assert state.job("b26-fp").status is JobStatus.READY
    assert state.job("h-scr").parent_job_ids == ("p0", "i-scr")
    assert state.job("h-scr").model_parent_job_id == "p0"
    assert all(not job.id.startswith("q") for job in state.jobs)
```

Add tests that a successful dependency unlocks exactly one next job, a failed dependency keeps children blocked, and missing model-parent checkpoint prevents readiness.

- [ ] **Step 2: Run RED**

Run: `../.venv/bin/pytest -q tests/test_queue_workflow.py`
Expected: missing workflow module.

- [ ] **Step 3: Implement initial graph and readiness**

Define job order explicitly, not by dictionary iteration. A job becomes ready only when all scheduling parents are terminal-success and its model parent has a checkpoint when required. Selection jobs require successful metric-bearing inputs but no GPU.

- [ ] **Step 4: Test and implement winner-derived YAML**

First add a test that derives W-DIR/W-PROG from an H winner while retaining H basis/bias/scale and changing only name/progressive. Then implement with `dataclasses.replace(parent, ...)`; write generated files under `artifacts/queue/generated/<job-id>/`. Never mutate committed templates and never use Identity defaults in place of the winner.

- [ ] **Step 5: Implement dynamic expansion in reviewable stages**

Add one failing test and one minimal expansion at a time:

1. architecture-select → W-DIR/W-PROG/recovery-select.
2. recovery-select → scale and bias jobs → A0-select.
3. A0-select → N0 graph and D0.
4. N0-select → at most two N1 jobs.
5. D0 → D1 sharing → D2 projection → R0/R1/R2.
6. denominator gate → optional Newton only when required.
7. normalization-select + bdcn-select → final-select → A-FINAL evaluate.

Every expansion is idempotent: calling it twice produces the same job IDs and never overwrites generated YAML.

- [ ] **Step 6: Verify GREEN**

Run: `../.venv/bin/pytest -q tests/test_queue_workflow.py tests/test_queue_policy.py`
Expected: all graph, readiness, derivation and selection expansion tests pass.

---

### Task 4: Evaluation and Standard Result Contract

**Files:**
- Create: `src/yolo_attention/evaluation.py`
- Create: `tests/test_queue_evaluation.py`
- Create: `configs/evaluation/coco2017.yaml`
- Create: `configs/evaluation/README.md`
- Modify: `src/yolo_attention/artifacts.py`

**Interfaces:**
- Produces: `EvaluationRecipe`, `EvaluationRequest`, `EvaluationBackend` protocol, `UltralyticsEvaluationBackend`.
- Produces: `standardize_metrics(result: object) -> dict[str, float]`.
- Produces: `resolve_run_outputs(run_dir: Path) -> QueueResult`.

- [ ] **Step 1: Write failing result extraction tests using small fakes**

```python
class FakeBox:
    map = 0.401
    map50 = 0.590
    map75 = 0.430


class FakeMetrics:
    box = FakeBox()


def test_standardize_metrics_uses_explicit_coco_keys() -> None:
    assert standardize_metrics(FakeMetrics())["map50_95"] == 0.401
```

Test `EvaluationRecipe.from_yaml()` for `data`, `imgsz`, `batch`, `device`, `workers`, `split`, and `plots`; also test rejection of missing `box.map`, non-finite numbers, missing `best.pt`, and writing `metrics/queue-result.json` without initializing CUDA.

- [ ] **Step 2: Run RED**

Run: `../.venv/bin/pytest -q tests/test_queue_evaluation.py`
Expected: missing module failure.

- [ ] **Step 3: Implement backend seams**

`UltralyticsEvaluationBackend.evaluate_official()` calls `YOLO(weights).val(**args)` only when the executor has `execute=True`. `evaluate_variant()` loads the parent model, applies `convert_yolo26_model()` to both paths, then validates. Constructors and pure extraction functions must not call either method. P0 validation is a separate backend method returning equivalence metrics; no shelling out to pytest from production code.

- [ ] **Step 4: Implement standardized artifact output**

Write exact keys `map50_95`, `map50`, `map75`, `maps`, `checkpoint_path`, `metrics_path`, `profile_path`, `row_sum_max_error`. Fields unavailable for a job are JSON null; fields required by that job are validated before success.

- [ ] **Step 5: Verify GREEN**

Run: `../.venv/bin/pytest -q tests/test_queue_evaluation.py`
Expected: all extraction/resolution tests pass with fake objects only.

---

### Task 5: Single-Worker Executor and Failure Recovery

**Files:**
- Create: `src/yolo_attention/queue_executor.py`
- Create: `tests/test_queue_executor.py`
- Modify: `src/yolo_attention/runner.py`

**Interfaces:**
- Consumes: `QueueStore`, queue workflow functions, `launch_training`, evaluation backend.
- Produces: `QueueExecutor.preview_next()`, `.run_next(execute: bool)`, `.run_all(execute: bool)`, `.retry(job_id)`.
- Executor accepts `backend: QueueBackend` so tests never call Ultralytics/GPU.

- [ ] **Step 1: Write failing dry-run safety test**

```python
def test_run_next_without_execute_never_calls_backend(queue_store, fake_backend) -> None:
    executor = QueueExecutor(queue_store, backend=fake_backend)
    preview = executor.run_next(execute=False)
    assert preview.job_id == "b26-fp"
    assert fake_backend.calls == []
    assert queue_store.load().job("b26-fp").status is JobStatus.READY
```

- [ ] **Step 2: Run RED**

Run: `../.venv/bin/pytest -q tests/test_queue_executor.py`
Expected: missing executor module.

- [ ] **Step 3: Implement preview and execute state machine**

Under the worker lock: reload, refresh, select one job, save queued, save running, dispatch by `JobKind`, validate result, save succeeded, append events, expand selection, refresh readiness. On ordinary exception save failed with `failure_reason` and re-raise a typed `QueueExecutionError`; on `KeyboardInterrupt` save interrupted before re-raising.

- [ ] **Step 4: Add failure/retry/locking tests and implementation**

Tests must prove:

- two executors cannot hold the lock;
- succeeded jobs never dispatch again;
- failure preserves attempts and message;
- retry accepts only failed/interrupted, changes it to queued, and `next_runnable_job()` selects it before ready jobs;
- `run_all(execute=False)` previews only one job and terminates;
- `run_all(execute=True)` remains sequential and stops on failure/block/no-ready.

- [ ] **Step 5: Verify GREEN**

Run: `../.venv/bin/pytest -q tests/test_queue_executor.py tests/test_queue_model_store.py`
Expected: all executor and recovery tests pass.

---

### Task 6: Queue CLI

**Files:**
- Modify: `src/yolo_attention/cli.py`
- Create: `tests/test_queue_cli.py`

**Interfaces:**
- Adds: `queue init|status|validate|next|run-next|run|retry`.
- Common option: `--queue-root`, default `artifacts/queue`.
- JSON commands emit one JSON document to stdout; errors go to stderr and return nonzero.

- [ ] **Step 1: Write failing CLI init/status/next tests**

```python
def test_queue_init_and_next_are_cpu_safe(tmp_path: Path, capsys) -> None:
    root = tmp_path / "queue"
    assert main(["queue", "init", "--queue-root", str(root), "--json"]) == 0
    assert main(["queue", "next", "--queue-root", str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert payload["job_id"] == "b26-fp"
    assert payload["will_execute"] is False
```

Monkeypatch the executor constructor so the test fails if any backend execution method is called.

- [ ] **Step 2: Run RED**

Run: `../.venv/bin/pytest -q tests/test_queue_cli.py`
Expected: argparse rejects `queue`.

- [ ] **Step 3: Implement nested argparse commands**

Put parser construction in `_add_queue_parser(subparsers)` and handling in `_run_queue_command(args)`. `run-next` and `run` always include `will_execute` in dry-run output. `retry` changes state but never executes it.

- [ ] **Step 4: Add validation report tests**

Assert `queue validate --json` returns:

```json
{"valid": true, "errors": [], "warnings": ["optional quantization is locked"]}
```

For missing weight/data, assert `valid=false`, nonzero exit, and exact missing paths in `errors`; do not touch CUDA.

- [ ] **Step 5: Verify GREEN**

Run: `../.venv/bin/pytest -q tests/test_queue_cli.py tests/test_trainer_profiling_cli.py`
Expected: old and new CLI tests pass.

---

### Task 7: Documentation and Repository Navigation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/architecture.md`
- Modify: `src/README.md`
- Modify: `configs/README.md`
- Create: `artifacts/README.md`
- Modify: `tests/README.md`
- Modify: `plan.md`

**Interfaces:**
- Documents exactly the commands and paths implemented in Tasks 1–6.

- [ ] **Step 1: Add a documentation contract test**

Add to `tests/test_queue_cli.py` an assertion that every documented queue subcommand exists in `build_parser()` and that `artifacts/README.md` exists. Run it and observe RED before creating the README/updating parser help.

- [ ] **Step 2: Update all documentation in one synchronized patch**

Root README gets a CPU-safe quick start:

```bash
../.venv/bin/python -m yolo_attention.cli queue init
../.venv/bin/python -m yolo_attention.cli queue validate
../.venv/bin/python -m yolo_attention.cli queue next
```

State prominently that these do not run GPU, and that only future `run-next --execute`/`run --execute` may do so. AGENTS records the single-worker lock, no implicit retry, parent checkpoint rule, and Optional Phase Q lock. Architecture documents module seams. `artifacts/README.md` explains which queue/run files are generated and not committed.

- [ ] **Step 3: Verify documentation contract**

Run: `../.venv/bin/pytest -q tests/test_queue_cli.py`
Expected: documentation contract passes.

---

### Task 8: Full CPU Verification and Queue Initialization

**Files:**
- Create at runtime: `artifacts/queue/queue.json`
- Create at runtime: `artifacts/queue/events.jsonl`

**Interfaces:**
- Uses the public CLI only.

- [ ] **Step 1: Run focused queue tests**

Run:

```bash
../.venv/bin/pytest -q \
  tests/test_queue_model_store.py \
  tests/test_queue_policy.py \
  tests/test_queue_workflow.py \
  tests/test_queue_evaluation.py \
  tests/test_queue_executor.py \
  tests/test_queue_cli.py
```

Expected: all pass.

- [ ] **Step 2: Run complete regression and lint**

Run:

```bash
../.venv/bin/pytest -q
../.venv/bin/ruff check src tests scripts
```

Expected: all tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 3: Initialize the real queue without executing it**

Run:

```bash
../.venv/bin/python -m yolo_attention.cli queue init --json
../.venv/bin/python -m yolo_attention.cli queue status --json
../.venv/bin/python -m yolo_attention.cli queue validate --json
../.venv/bin/python -m yolo_attention.cli queue next --json
```

Expected: `b26-fp` is the only ready/next job, every later job is blocked, `will_execute=false`, no run directory is created, and no CUDA call occurs. Validation may report missing external prerequisites as errors but must not modify the queue.

- [ ] **Step 4: Verify no training or GPU artifacts were created**

Confirm `artifacts/runs/` has no new queue-created run and `events.jsonl` contains only the initialization event. Report GPU training and COCO metrics as `not_run`.
