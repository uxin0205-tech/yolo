# Phase 1 Recovery, B0/M7 Extension, and Per-Class Report Design

## Goal

Recover the interrupted YOLO11 Phase 1 workflow without retraining the already
validated B1 model, complete the B0, M7, and M0–M3 work, and run the remaining
GPU pipeline independently of the Codex session. The terminal result must be an
auditable comparison of B0, B1, M7, M0, M1, M2, and M3 with separate Ball and
Bat reporting.

## Existing State and Recovery Boundary

The existing B1-A and B1-B outputs are retained. B1-B has a completed 90-epoch
Ultralytics output, a canonical checkpoint, strict-reload evidence, and matching
stage hashes. The completed M7 CUDA gate and inspected B0 reference are also
retained when their manifests and files pass the existing hash checks.

The interrupted `smoke_m7` stage contains a resolved argument manifest and a
worker request, but no training run, checkpoint, or worker result. It is stale,
not complete. Recovery must safely rerun `smoke_m7`; it must not infer success
from the stale `running` state and must not invalidate earlier completed stages.

Generated datasets, runs, checkpoints, metrics, and reports remain ignored
artifacts. Source weights and the B0 reference checkpoint are immutable and are
never overwritten or committed as new large Git objects.

## Model Roles

- **B0** is the exact pose-derived detection checkpoint defined by
  `configs/b0-reference.yaml`. It is evaluated and profiled but never retrained,
  used for initialization, or admitted to model selection. Every report labels
  it `data_exposed=true` and explains that its numbers are operational reference
  values rather than a leak-free comparison.
- **B1** reuses the existing validated B1-B canonical checkpoint. It participates
  in the same validation, test, profiling, and reporting loops as the trained
  variants but is not retrained.
- **M7** uses MFAM kernels 3/5/7 at full processed ratio. Its smoke and formal
  training run before M0–M3.
- **M0–M3** retain their existing definitions and formal training contract.
- **BEST_PARTIAL** remains selected only from M2 and M3 using validation data.
  B0, B1, M7, M0, and M1 cannot affect selection, even if their metrics are
  better. Test data remains inaccessible until selection is frozen.

## Execution Architecture and Data Flow

The repository-local `masf_yolo` workflow remains the sole orchestrator. It
classifies each stage from immutable inputs and hash-verified outputs before
deciding whether to reuse, resume, restart, or fail. The remaining order is:

1. recover stale `smoke_m7` and run M7 smoke;
2. run M7 formal training;
3. run M0–M3 smoke training;
4. run M0–M3 formal training;
5. evaluate all seven models on validation;
6. freeze M2/M3 selection;
7. evaluate all seven models on test;
8. profile all seven models;
9. run the final audit and render the report.

Training runs inside isolated worker processes so CUDA memory is released at
process boundaries. Canonical checkpoint finalization and strict validation run
in fresh processes. The parent workflow exchanges only paths, hashes, and atomic
JSON manifests with workers.

After code and tests pass, the workflow is launched as a transient user systemd
service with bounded restart behavior, OOM containment, and memory accounting.
It must continue when Codex or the interactive shell is closed. Status and report
commands are read-only operator entry points for a later session.

## Confirmed Context

- B1-A completed 10 epochs and has a canonical strict checkpoint.
- B1-B completed all 90 epochs. Its `results.csv` has 90 data rows and its
  loadable `best.pt` and `last.pt` are each approximately 40 MB.
- B1-B was killed during Ultralytics' duplicate final validation, after weights
  were stripped and after epoch 90 validation had already completed.
- Kernel evidence records a global OOM kill, a 13.1 GB service memory peak,
  2.7 GB swap peak, and 88 leaked multiprocessing semaphores.
- On restart, the pipeline treated the completed stripped `last.pt` as
  resumable. Ultralytics correctly rejected it with “training to 90 epochs is
  finished, nothing to resume.”
- B0 is `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`, SHA-256 `9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d`.
- B0 metadata shows Ultralytics 8.4.90, classes `ball` and `bat`, strides
  `[8, 16, 32]`, and a detection model.

## Interruption and Error Handling

Recovery is fail-closed:

- A completed stage is reused only when its declared inputs and output hashes
  match and every required file is present.
- A complete Ultralytics run is finalized without calling training or native
  resume again.
- An incomplete run may resume only from its exact validated `last.pt`.
- A stale stage with no accepted training output restarts that stage from its
  immutable inputs.
- Contradictory CSV/checkpoint state, hash mismatch, non-finite results, invalid
  architecture, or corrupt manifests are permanent errors.
- Process interruption and supported CUDA OOM cases are transient, subject to
  the existing bounded retry policy. The shared batch size is not silently
  lowered after it has been locked.

No recovery action deletes weights or datasets. Stale lock files are handled by
lock ownership/liveness checks rather than deletion based only on age.

## Ball and Bat Reporting

Every model receives overall, Ball, and Bat sections for validation and test.
The per-class report includes, when supported by the evaluator:

- AP50 and AP50–95;
- precision and recall;
- ground-truth and prediction counts;
- size-stratified AP/recall and support counts;
- blur-proxy recall, missed detections, and false-positive summaries.

Ball retains the research-specific small-object bins `<8 px`, `8–16 px`, and
`>16 px`, along with COCO small/medium/large summaries. Bat is promoted from an
implicit overall component to a first-class report section and receives the
same common per-class and diagnostic treatment. Unsupported subsets render a
`null` metric with their support count instead of fabricating zero.

The human-readable report compares B0, B1, M7, and M0–M3 side by side. It also
includes checkpoint identity, provenance, validation and test summaries,
hardware profiles, selection rationale, failures/retries, and the final audit.

## Verification and Acceptance

Before launch:

- eliminate the pytest collection collision between evaluation and training
  runner tests;
- run the complete test suite in one invocation, not only split suites;
- run format/static repository checks that already exist;
- verify B1, M7 gate, and B0 artifacts against their manifests;
- prove stale `smoke_m7` recovery with an automated regression test;
- run the real M7 CUDA smoke/gate required by the plan;
- verify the systemd command and confirm the launched unit is active.

Terminal acceptance requires:

- B1-A/B remain byte/hash-identical and are not retrained;
- M7 and M0–M3 smoke/formal checkpoints pass canonical finalization and strict
  reload;
- B0/B1/M7/M0/M1/M2/M3 each have validation, test, and profile artifacts;
- selection is frozen from M2/M3 validation data before any test evaluation;
- overall, Ball, and Bat sections are complete for every evaluated model;
- `final_audit.ok=true` with no audit errors;
- the human-readable final report is reproducible from immutable artifacts;
- the service can finish without an active Codex session.

## Git and Handoff

Implementation changes and documentation are committed in focused commits and
pushed to the project branch after verification. Large datasets, caches, runs,
and checkpoints remain outside normal Git tracking. When the user returns after
the service finishes, Codex will re-read the terminal artifacts and produce a
fresh concise synthesis of overall, Ball, Bat, model-quality, and efficiency
results.
