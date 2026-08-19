# achitechure_1 experiment contract

Status: implementation complete; formal MASF training is blocked pending explicit GPU availability clearance.

## Scope and authority

This document is the formal contract for the YOLO26m Attention + P3 MASF experiment. It overrides conflicting
language in `references/plan.md`:

- `Partial75` means 25% context channels and 75% exact identity bypass.
- Both DW3 and DW5 outputs are summed and then passed through a required `C→C` 1×1 projection.
- This is a MASF-inspired P3 adaptation. It is not a claim of reproducing the complete MASF-YOLO paper.

No P2 feature or Detect input is added. P4/P5, Detect, loss, assignment, augmentation, and the two
hardware-friendly attention sites are not redesigned.

## Immutable parent

The immutable A0 parent is `inputs/parent/best.pt`, SHA256
`c989aeed09de7663ad093d32d098e5fc889cf04924fa1162efaf886869de0123`. The completed attention study ran all
19/19 queued jobs, including the seed-0 LR sweep and staged recovery. Its formal gate retained the verified
zero-train Bit-True checkpoint because the best trained observation improved mAP50-95 by only `0.0002017689` and no
three-seed mean was run. The completed final manifest, recipe, selector, queue state, and event history are preserved
under `inputs/parent/provenance/`. The unselected best-observed checkpoint is recorded by SHA256 but is not used as
A0.

The actual graph is end-to-end YOLO26m with Detect inputs `[16, 19, 22]`, strides `[8, 16, 32]`, and exactly two
hardware-friendly attention sites: `model.10.m.0.attn` and `model.22.m.0.1.attn`.

## Architecture

The P3 seam is discovered from the first actual Detect input and then validated. The existing C3k2 instance is
upgraded in place to a subclass, so graph index 16 and every original state-dict name remain unchanged.

For context tensor `x`:

```text
delta = Conv1x1(DWConv3x3(x) + DWConv5x5(x))
y = x + alpha * delta
alpha = learnable scalar initialized to 0.01
```

- A0: immutable parent, no MASF and no retraining.
- A1 Full35: all 256 P3 channels use the context path.
- A2 Partial75: the first 64 P3 channels use the context path; the final 192 channels are concatenated back without
  any operation and are bit-exact.

The three convolutions follow the installed Ultralytics Conv → BatchNorm → SiLU convention. DW3 and DW5 are true
depthwise convolutions; the projection is a dense 1×1 convolution.

## Common training contract

All A1/A2 phases use COCO2017, batch 16, image size 640, seed 0, eight workers, deterministic mode, AMP, identical
Ultralytics 8.4.90 augmentations, and MuSGD. `nbs=16` makes accumulation exactly one. Automatic OOM batch reduction
is disabled. If batch 16 does not fit, execution stops rather than changing batch or enabling accumulation.

Every phase creates a new trainer and therefore a fresh optimizer, scheduler, and EMA. Frozen BatchNorm running
buffers and frozen attention train/eval state are enforced immutable. Detect remains in training mode because
putting a frozen Detect module into eval changes its public output from training dictionaries to inference tuples.

## Fixed phases

| Phase | Epochs | Trainable scope | LR groups | Warmup | Schedule | Patience |
|---|---:|---|---|---:|---|---:|
| A1 | 5 | MASF only | MASF `1.0e-3` | 0.5 | constant | fixed phase |
| A2 | ≤10 | MASF only | MASF `3.8e-4` | 0 | cosine to 50% | 4 |
| B | ≤10 | MASF + actual Neck/Detect, attention frozen | MASF `3.8e-4`; Neck/Detect `1.9e-4` | 1 | cosine to 50% | 4 |
| C | ≤55 | full model | MASF `3.8e-4`; Neck/Detect `1.9e-4`; Backbone `3.8e-5`; attention `5e-6` | 1 | cosine to 10% | 8 |

Momentum is 0.948 and Conv weight decay is 0.00027. Alpha, biases, and BatchNorm parameters have no decay.
Every epoch's best/early-stop metric is measured on a deep-copied, actually converted Bit-True PWL EMA model;
Float-PWL remains the differentiable training model.

A2 and B stop after four consecutive epochs without a higher Bit-True mAP50-95, then roll back when the child loses
more than 0.001 to its accepted parent. Phase C retains its parent unless it strictly improves it. A1 remains a
fixed five-epoch phase.

## Winner tuning and selection

A1 versus A2 is selected by Bit-True COCO2017 mAP50-95. Within 0.001, lower same-GPU FP16 p50 latency wins,
followed by lower GFLOPs and Params. Peak VRAM is recorded only as a capacity diagnostic and never ranks candidates,
because development checks may run on an RTX 4080 SUPER while formal training and final profiling run on an RTX
5090. The architecture winner then starts T1/T2/T3 from the exact same Float winner checkpoint with MASF LR 0.0002,
0.00038, or 0.0006. Other LR groups preserve Phase-C ratios. Each continuation runs at most ten epochs with patience
five and a fresh optimizer.

Formal candidates are A0, accepted A1/A2 best, and T1/T2/T3 best. Reports must include mAP50-95, mAP50, mAP75,
AP_S/M/L, recall, class-32 sports-ball AP, class-34 baseball-bat AP, alpha, Params, GFLOPs, FP16 p50 latency, peak
VRAM, epoch time, curves, best/stop epoch, and stop reason.

## Causal limitation

A0 is not retrained. Any A1/A2 difference from A0 combines the MASF change with staged retraining and cannot be
claimed as a pure architectural causal gain.
