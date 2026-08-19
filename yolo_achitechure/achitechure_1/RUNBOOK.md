# achitechure_1 runbook

Run from `/home/uxin/yolo/yolo_achitechure/achitechure_1`. The commands below use the repository virtual
environment and the supplied attention implementation required to deserialize the parent checkpoint.

```bash
export PYTHONPATH="$PWD/src:$PWD/../../yolo_attention_final/src"
PYTHON="$PWD/../../.venv/bin/python"
```

Commands are dry-run by default where they can start training, validation, conversion, or profiling. Add
`--execute` only on the intended GPU host. Never change batch or enable accumulation after an OOM.

## Environment and tests

```bash
$PYTHON -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
$PYTHON -m achitechure_1.cli preflight
$PYTHON -m pytest
$PYTHON -m achitechure_1.cli smoke --variant full35 --device cuda:0
$PYTHON -m achitechure_1.cli smoke --variant partial75 --device cuda:0
```

Preflight must report `valid: true`, three Detect inputs `[16,19,22]`, strides `[8,16,32]`, `end2end: true`,
the two attention paths, MuSGD, fresh-process reload, and at least 95% Float reconstruction coverage. Formal COCO
validation also requires the optional evaluation dependency:

```bash
$PYTHON -m pip install -e '.[evaluation]'
```

## A0 immutable baseline

A0 is `inputs/parent/best.pt`; do not train it. Profile and validate the actual Bit-True checkpoint:

```bash
$PYTHON -m achitechure_1.cli profile \
  --checkpoint inputs/parent/best.pt --kind inference \
  --output artifacts/profiles/a0-inference.json --execute

$PYTHON -m achitechure_1.cli validate \
  --checkpoint inputs/parent/best.pt --run-id a0-bittrue --execute
```

## Prepare A1 and A2

These commands add MASF without changing any original parent tensor or graph index:

```bash
$PYTHON -m achitechure_1.cli prepare \
  --variant full35 --output artifacts/prepared/a1-full35.pt --execute
$PYTHON -m achitechure_1.cli prepare \
  --variant partial75 --output artifacts/prepared/a2-partial75.pt --execute
```

Before formal epochs, prove fixed batch 16/640 fits using the official YOLO loss. Do not pass a smaller batch:

```bash
$PYTHON -m achitechure_1.cli profile \
  --checkpoint artifacts/prepared/a1-full35.pt --kind training \
  --output artifacts/profiles/a1-train-smoke.json --steps 3 --execute
$PYTHON -m achitechure_1.cli profile \
  --checkpoint artifacts/prepared/a2-partial75.pt --kind training \
  --output artifacts/profiles/a2-train-smoke.json --steps 3 --execute
```

The 2026-08-17 attempt stopped with OOM while the separate parent attention queue held about 15.5 GiB VRAM.
After that queue completed, the exact probes were repeated on an RTX 5090 on 2026-08-18 and both passed without batch
reduction or accumulation: Full35 peaked at `18,397,301,760` bytes and Partial75 at `17,761,890,304` bytes. The
committed reports are `artifacts/profiles/a1-train-smoke.json` and `artifacts/profiles/a2-train-smoke.json`.

The current development host uses an RTX 4080 SUPER; formal training and final comparable profiling are intended for
the RTX 5090. Any 4080 smoke output must use a hardware-tagged filename such as
`a1-train-smoke-rtx4080super-YYYY-MM-DD.json` and must not overwrite the committed 5090 reports. Peak VRAM remains a
capacity diagnostic only and is not an architecture-selection criterion. These readiness checks are not formal
training runs and do not include P3 MASF MACs calculation. Do not begin formal epochs until the RTX 5090 is available
and the workspace owner explicitly clears the run.

## Autonomous Full35 → Partial75 trajectories

For the RTX 4080 SUPER recovery path, the completed Full35 Phase A1 can feed an autonomous, fail-closed queue. It
finishes Full35 before starting Partial75, and applies the same Bit-True validation and rollback gates to each
architecture's A1 → A2 → B → C trajectory. Every job fixes `batch=16` and `nbs=16`. A training child uses four
training DataLoader workers and zero simultaneously resident validation workers, preventing Ultralytics from turning
`workers=4` into 4 training plus 8 validation workers. A standalone formal validation uses four workers. Every job
runs in its own child process and waits for at least 3 GiB of available system RAM before launch. A completed child
exits before its successor starts, releasing DataLoader workers, shared memory, and CUDA state.

Phase C starts only when the detected GPU has that architecture's measured batch-16 peak plus one GiB of headroom.
If Phase C cannot fit, the queue records it as pending and continues the other architecture instead of launching a
known OOM configuration. After all currently safe work is complete, the final status is
`waiting_for_phase_c_gpu`.

```bash
$PYTHON scripts/run_architecture_recovery_queue.py \
  --run-tag rtx4080super-batch16-workers4-r1
```

Queue progress is written atomically to
`artifacts/queues/architecture-rtx4080super-batch16-workers4-r1/state.json`.

To continue after an already gated Full35 Phase B with at most eight resident workers, use a new hardware/settings
tag. This path requires at least 7 GiB of available RAM before every child process, retains the Phase-C VRAM gate,
and starts Partial75 without repeating Full35:

```bash
$PYTHON scripts/run_architecture_recovery_queue.py \
  --run-tag rtx4080super-batch16-workers8-r2 \
  --workers 8 \
  --continue-after-full35-state \
  artifacts/queues/architecture-rtx4080super-batch16-workers4-r1/state.json
```

The example below uses Full35. Repeat identically with `--variant partial75`, an `a2-partial75-*` run-id prefix,
and `artifacts/prepared/a2-partial75.pt`.

When the batch-16 queue reaches its Phase-C handoff, the recovery relay can test and run Phase C with a physical
microbatch of 8 and `nbs=16` (two-step gradient accumulation, effective batch 16). It keeps six DataLoader workers,
uses zero simultaneous in-training validation workers, and retains batch 16 for standalone Bit-True validation.
Before either architecture starts formal Phase-C epochs, a two-step real-loss profile must complete and its measured
peak plus 1 GiB must fit on the detected GPU. An OOM or insufficient margin is recorded as pending instead of being
retried blindly. The relay also recognizes the older workers=4 manifest-verification mismatch after a completed A1
and resumes from that completed checkpoint without interrupting or repeating the training job:

```bash
$PYTHON scripts/run_phase_c_recovery_after_queue.py \
  --source-state artifacts/queues/architecture-rtx4080super-batch16-workers6-r3/state.json \
  --source-run-tag rtx4080super-batch16-workers6-r3 \
  --source-full35-state artifacts/queues/architecture-rtx4080super-batch16-workers4-r1/state.json \
  --run-tag rtx4080super-batch8x2-workers6-r4 \
  --workers 6
```

### Phase A1

```bash
$PYTHON -m achitechure_1.cli train \
  --variant full35 --phase a1 \
  --weights artifacts/prepared/a1-full35.pt \
  --run-id a1-full35-phase-a1 --execute

$PYTHON -m achitechure_1.cli materialize-bittrue \
  --checkpoint artifacts/runs/a1-full35-phase-a1/ultralytics/weights/best.pt \
  --output artifacts/checkpoints/a1-full35-phase-a1-bittrue.pt --execute
$PYTHON -m achitechure_1.cli validate \
  --checkpoint artifacts/checkpoints/a1-full35-phase-a1-bittrue.pt \
  --run-id a1-full35-phase-a1-bittrue --execute
```

### Phase A2 and rollback gate

Phase A2 runs for at most ten epochs and stops after four consecutive Bit-True validation epochs without
improvement. `best.pt` remains the highest-scoring observation.

```bash
$PYTHON -m achitechure_1.cli train \
  --variant full35 --phase a2 \
  --weights artifacts/runs/a1-full35-phase-a1/ultralytics/weights/best.pt \
  --run-id a1-full35-phase-a2 --execute

$PYTHON -m achitechure_1.cli materialize-bittrue \
  --checkpoint artifacts/runs/a1-full35-phase-a2/ultralytics/weights/best.pt \
  --output artifacts/checkpoints/a1-full35-phase-a2-bittrue.pt --execute
$PYTHON -m achitechure_1.cli validate \
  --checkpoint artifacts/checkpoints/a1-full35-phase-a2-bittrue.pt \
  --run-id a1-full35-phase-a2-bittrue --execute

$PYTHON -m achitechure_1.cli gate \
  --parent-name phase-a1 --parent-map PARENT_MAP \
  --child-name phase-a2 --child-map CHILD_MAP
```

Use the selected phase's Float `best.pt` as Phase B input. A loss of exactly 0.001 is accepted; only a loss greater
than 0.001 rolls back.

### Phase B and rollback gate

Phase B uses the same patience-four rule with a maximum of ten epochs.

```bash
$PYTHON -m achitechure_1.cli train \
  --variant full35 --phase b --weights ACCEPTED_PHASE_A_FLOAT_BEST_PT \
  --run-id a1-full35-phase-b --execute
$PYTHON -m achitechure_1.cli materialize-bittrue \
  --checkpoint artifacts/runs/a1-full35-phase-b/ultralytics/weights/best.pt \
  --output artifacts/checkpoints/a1-full35-phase-b-bittrue.pt --execute
$PYTHON -m achitechure_1.cli validate \
  --checkpoint artifacts/checkpoints/a1-full35-phase-b-bittrue.pt \
  --run-id a1-full35-phase-b-bittrue --execute
$PYTHON -m achitechure_1.cli gate \
  --parent-name phase-a --parent-map PARENT_MAP \
  --child-name phase-b --child-map CHILD_MAP
```

Use the selected Float checkpoint as Phase C input.

### Phase C and strict parent retention

```bash
$PYTHON -m achitechure_1.cli train \
  --variant full35 --phase c --weights ACCEPTED_PHASE_B_PARENT_FLOAT_PT \
  --run-id a1-full35-phase-c --execute
$PYTHON -m achitechure_1.cli materialize-bittrue \
  --checkpoint artifacts/runs/a1-full35-phase-c/ultralytics/weights/best.pt \
  --output artifacts/checkpoints/a1-full35-phase-c-bittrue.pt --execute
$PYTHON -m achitechure_1.cli validate \
  --checkpoint artifacts/checkpoints/a1-full35-phase-c-bittrue.pt \
  --run-id a1-full35-phase-c-bittrue --execute
$PYTHON -m achitechure_1.cli gate \
  --policy phase-c-improve \
  --parent-name phase-b --parent-map PARENT_MAP \
  --child-name phase-c --child-map CHILD_MAP
```

Training itself converts a deep copy of the Float EMA to Bit-True every epoch, so `best.pt` and early stopping are
driven by Bit-True mAP50-95. Patience is four for A2/B and eight for Phase C. Phase C retains its parent unless the
child strictly improves.

## Architecture winner

Create one JSON file per A1/A2 candidate with these selection keys:

```json
{
  "name": "a1-full35",
  "map50_95": 0.0,
  "fp16_p50_ms": 0.0,
  "gflops": 0.0,
  "parameters": 0
}
```

Latency must be measured for both candidates on the same RTX 5090 with identical settings. An optional
`peak_vram_gib` field may be retained for diagnostics, but the selector deliberately ignores it.

Then select:

```bash
$PYTHON -m achitechure_1.cli select \
  --candidate results/a1-candidate.json --candidate results/a2-candidate.json
```

## T1/T2/T3 winner continuations

All three commands must reference the exact same accepted Float architecture-winner checkpoint:

```bash
$PYTHON -m achitechure_1.cli train --variant WINNER_VARIANT --phase tune --masf-lr 0.0002 \
  --weights WINNER_FLOAT_PT --run-id t1-lr-0002 --execute
$PYTHON -m achitechure_1.cli train --variant WINNER_VARIANT --phase tune --masf-lr 0.00038 \
  --weights WINNER_FLOAT_PT --run-id t2-lr-00038 --execute
$PYTHON -m achitechure_1.cli train --variant WINNER_VARIANT --phase tune --masf-lr 0.0006 \
  --weights WINNER_FLOAT_PT --run-id t3-lr-0006 --execute
```

Materialize and validate each best checkpoint using the same commands as above.

## Final profile and delivery

```bash
$PYTHON -m achitechure_1.cli profile \
  --checkpoint FINAL_BITTRUE_PT --kind inference \
  --output artifacts/profiles/final-inference.json --execute
$PYTHON -m achitechure_1.cli export-masf \
  --checkpoint FINAL_BITTRUE_PT --output results/masf-only.pt --execute
sha256sum FINAL_BITTRUE_PT results/masf-only.pt inputs/parent/best.pt
git status --short --branch
```

Copy, without reserializing, the selected reloadable Bit-True checkpoint to `results/best.pt`; record its SHA256 and
source Float checkpoint in the final results row. Update `results/results-template.csv`, including AP_S/M/L from
canonical COCO API, class 32/34 AP, recall, final alpha, latency, Params, GFLOPs, VRAM, total time, curves, best/stop
epochs, and stop reason. Explicitly retain the A0 causal-limitation statement from `EXPERIMENT_SPEC.md`.
