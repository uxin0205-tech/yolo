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
After that queue completed, the exact probes were repeated on 2026-08-18 and both passed without batch reduction or
accumulation: Full35 peaked at `18,397,301,760` bytes and Partial75 at `17,761,890,304` bytes. The reports are
`artifacts/profiles/a1-train-smoke.json` and `artifacts/profiles/a2-train-smoke.json`. This readiness check is not a
formal training run and does not include P3 MASF MACs calculation. The GPU is currently considered externally
blocked; do not rerun GPU checks or begin formal epochs until the workspace owner explicitly clears it.

## One architecture's A1 → A2 → B → C trajectory

The example below uses Full35. Repeat identically with `--variant partial75`, an `a2-partial75-*` run-id prefix,
and `artifacts/prepared/a2-partial75.pt`.

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

Training itself converts a deep copy of the Float EMA to Bit-True every epoch, so `best.pt` and patience five are
driven by Bit-True mAP50-95. Phase C retains its parent unless the child strictly improves.

## Architecture winner

Create one JSON file per A1/A2 candidate with exactly these keys:

```json
{
  "name": "a1-full35",
  "map50_95": 0.0,
  "fp16_p50_ms": 0.0,
  "gflops": 0.0,
  "parameters": 0,
  "peak_vram_gib": 0.0
}
```

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
