# achitechure_2 runbook

Commands below are dry runs unless `--execute` is present. Run from this directory with the repository environment:

```bash
export PYTHONPATH="$PWD/src:$PWD/../achitechure_1/src:$PWD/../../yolo_attention_final/src"
PY=/home/uxin/yolo/.venv/bin/python
$PY -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"
$PY -c "import pycocotools; print(pycocotools.__version__ if hasattr(pycocotools, '__version__') else 'installed')"
$PY -m achitechure_2 preflight
$PY -m pytest
```

Do not start formal work if `pycocotools` is absent, CUDA/batch 16 is unavailable, the repo revision is not recorded,
or the handoff is incomplete.

## Gate 1 — accept the final achitechure_1 handoff

Copy `handoff-manifest.example.json` outside tracked results, fill it with the final Float-PWL and Bit-True checkpoint
pair, actual SHA256 values and model-selection provenance, then preview and execute:

```bash
$PY -m achitechure_2 intake --manifest /absolute/path/handoff.json
$PY -m achitechure_2 intake --manifest /absolute/path/handoff.json --execute
```

The command rejects missing MASF, the wrong variant/backend, wrong attention paths, non-YOLO26m Detect geometry,
version/hash mismatches, and fresh-process reload failure. `artifacts/intake/accepted.json` is the only formal gate.

## Gate 2 — independently build C0–C3

```bash
for id in C0 C1 C2 C3; do
  $PY -m achitechure_2 build --candidate "$id"
done
for id in C0 C1 C2 C3; do
  $PY -m achitechure_2 build --candidate "$id" --execute
done
```

Each build reloads the same accepted Float parent, writes its own checkpoint and transfer report, and asserts class,
channels, outer repeats, e, inner depth, kernels, Rep flag, MASF, and both attention paths.

## Gate 3 — smoke, formal A, and conditional extension B

Preview each command first. Smoke metrics are never used for ranking.

```bash
$PY -m achitechure_2 train --candidate C0 --checkpoint artifacts/candidates/c0/float-parent.pt --stage smoke --run-id c0-smoke --smoke-epochs 3 --execute
$PY -m achitechure_2 train --candidate C0 --checkpoint artifacts/candidates/c0/float-parent.pt --stage formal --run-id c0-formal --execute
```

Repeat unchanged for C1, C2, and C3. Materialize and validate each selected Bit-True checkpoint in separate CLI processes, then profile with identical settings:

```bash
$PY -m achitechure_2 materialize-bittrue --checkpoint /path/to/best-float.pt --output /path/to/best-bittrue.pt --execute
$PY -m achitechure_2 validate-bittrue --checkpoint /path/to/best-bittrue.pt --run-id c0-fresh-bittrue --execute
$PY -m achitechure_2 profile --checkpoint /path/to/best-bittrue.pt --output artifacts/runs/c0-formal/profile.json --warmup 20 --iterations 100 --execute
```

Export the 100 one-based Bit-True mAP values as a JSON array and inspect the
extension gate:

```bash
$PY -m achitechure_2 extension-gate --metrics /path/to/epoch-map.json --best-epoch 91
$PY -m achitechure_2 train --candidate C0 --checkpoint /path/to/c0/last.pt --stage extension --run-id c0-extension --execute
```

Run the extension command only when the gate returns `extend=true`. It resumes optimizer/scheduler state from
`last.pt`. After every run, perform validation/profile in a fresh process and record Params, GFLOPs, FP16 batch-1
latency (same warmup/iterations/device), peak CUDA memory, resolved args, git revision, config/checkpoint hashes, best
epoch, and stop reason.

## Gate 4 — decide and run conditional branches

Each metrics JSON has fields `candidate_id`, `map50_95`, `latency_ms`, `gflops`, and `params`.

```bash
$PY -m achitechure_2 assess --c0 /path/c0.json --candidate /path/c1.json --candidate /path/c2.json --candidate /path/c3.json
$PY -m achitechure_2 assess --c0 /path/c0.json --candidate /path/c1.json --candidate /path/c2.json --candidate /path/c3.json --execute
```

If `triggers.c3_p5` is true, independently build/train `C3-P5`. If `triggers.r1` is true, independently build/train R1
and run its fusion equivalence check before selection or quantization. Never combine R1 with C1/C3. A null `c_best`
ends the quantization mainline.

## Gate 5 — Q0/Q1/Q2 for C0 and C_best only

`fuse-reference` materializes Q0 and proves it reloads. `quant-prepare` refuses any candidate other than C0 and the winner in `artifacts/selection.json`; pass the fused Q0 checkpoint into Q1; prepare the trainable Q2 graph separately from the unfused Float best checkpoint.

```bash
$PY -m achitechure_2 fuse-reference --candidate C0 --checkpoint /path/to/c0/best-float.pt --output artifacts/quant/c0/q0/fused.pt --execute
$PY -m achitechure_2 quant-prepare --candidate C0 --checkpoint artifacts/quant/c0/q0/fused.pt --output artifacts/quant/c0/q1/prepared.pt
$PY -m achitechure_2 quant-prepare --candidate C0 --checkpoint artifacts/quant/c0/q0/fused.pt --output artifacts/quant/c0/q1/prepared.pt --execute
$PY -m achitechure_2 quant-calibrate --checkpoint artifacts/quant/c0/q1/prepared.pt --calibration-tensors /path/to/fixed-calibration-images.pt --output artifacts/quant/c0/q1/calibrated.pt --execute
$PY -m achitechure_2 quant-prepare --candidate C0 --checkpoint /path/to/c0/best-float.pt --output artifacts/quant/c0/q2/prepared.pt --execute
$PY -m achitechure_2 train --candidate C0 --checkpoint artifacts/quant/c0/q2/prepared.pt --stage qat --run-id c0-q2 --execute
$PY -m achitechure_2 quant-report --q0 0.50 --q1 0.49 --q2 0.497
```

Export the fixed calibration split once as a normalized FP32 tensor with shape `[N, 3, 640, 640]`; `quant-calibrate` records the consumed batch count and freezes its observers before fake-quant Bit-True evaluation. For Q2, use the same adapter,
fine-tune 10–15 epochs at 0.1× the inherited architecture LR; the QAT trainer calls `configure_qat_epoch` and freezes observers after epoch 3. Both checkpoints require fresh-process Bit-True validation. All tables and manifests
must say `simulation_only=true`.

## Finalization

Only after all required fresh-process validations succeed, copy `results/results-template.csv` to `results/results.csv`
and `results/report-template.md` to `REPORT.md`, populate them without blanks, and check:

```bash
git status --short --branch
```

Keep datasets, weights, predictions, plots, calibration caches, and benchmark outputs out of source directories.
