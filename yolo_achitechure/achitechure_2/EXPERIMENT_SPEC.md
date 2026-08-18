# YOLO26m Stage-Aware Lite-C3k2 experiment contract

## Scope and dependency gates

This workspace implements the engineering surface only. The PDF's Day 1–7 labels are dependency gates, not a
calendar deadline. Formal C0–C3 training is blocked until `intake --execute` accepts a final `achitechure_1` handoff
containing a retrainable Float-PWL checkpoint, its Bit-True counterpart, hashes, metrics, versions, and a hash-verified
model-selection manifest. The current `achitechure_1/inputs/parent/best.pt` is a development fixture: it has both inherited attention
sites but no P3 MASF, so it cannot become formal C0.

The inherited system is YOLO26m, Ultralytics 8.4.90, P3 MASF (`full35` or `partial75`), BinaryQK, Float/Bit-True PWL,
Detect inputs `[16, 19, 22]`, strides `[8, 16, 32]`, and `end2end=True`. No source under `yolo_p2`,
`yolo_attention_final`, or `achitechure_1` is copied or modified.

## Architecture factors

`LiteC3k2Config` exposes exactly `e`, `inner_n`, `kernel_mode`, and `use_rep`. Every candidate is built independently
from the same accepted Float checkpoint. Shape-compatible tensors are copied exactly; missing, unexpected, and
shape-mismatched tensors are reported. New or incompatible tensors retain deterministic seed-0 upstream-style
initialization.

| ID | Unique change | Layers |
|---|---|---|
| C0 | none | none |
| C1 | `e=0.375` | 6, 8, 13, 19 |
| C2 | `inner_n=1` | 6, 8, 13, 19 |
| C3 | `kernel_mode=1x1_3x3` | 6, 8, 13, 19 |
| C3-P5 | C3 fallback | 8 only |
| R1 | C2 plus `RepBottleneck` | 6, 8, 13, 19 |

Layers 2, 4, and 16 remain standard C3k2. Layer 22 is a P5 Detect input but contains inherited attention and is
excluded. P3 MASF and both attention roots remain eval-mode frozen, including parameters and buffers.

## Training and selection

All main candidates use COCO2017, 640 pixels, batch 16, seed 0, MuSGD, and identical augmentation/scheduler/validation
arguments. A 3–5 epoch smoke stage checks execution only. Formal A runs at most 100 epochs with patience 20. An
extension resumes `last.pt` through epoch 140 with patience 15 only when A did not early-stop and either the best epoch
is 85–100 or the epoch 81–100 rolling best exceeds epoch 61–80 by at least 0.001.

Float EMA is deep-copied and converted to Bit-True PWL at every validation. Bit-True mAP50-95 drives best checkpoint
selection and early stopping. Every delivered checkpoint must be reloaded and validated in a new process.

C0-relative rules are: drop at most 0.005 is PASS; drop above 0.005 through 0.008 is CONDITIONAL and eligible only
with at least 8% measured latency or GFLOPs improvement; drop above 0.008 is REJECT. C3 rejection triggers only
C3-P5. Eligible conditional C2 triggers independent R1. R1 requires fused/eval `max_abs_diff <= 1e-4`.
`C_best` ranks eligible single-factor candidates by decision state, mAP drop, measured latency, GFLOPs, then Params.
If none is eligible, the quantization mainline stops without inventing a winner.

## W8A8 robustness

Only C0 and recorded `C_best` run Q0 fused FP32, Q1 calibrated W8A8 PTQ fake-quant evaluation, and Q2 10–15 epoch
fake-quant fine-tuning (patience 5). Observers update in epochs 1–3 and freeze afterward. Standard Conv weights are
per-channel symmetric INT8; activations are per-tensor affine INT8. Standard Conv inside MASF is included.
Hardware-friendly attention—including Binary score/PWL custom arithmetic—is excluded and listed in the unquantized
manifest.

This first implementation uses PyTorch observer and fake-quant primitives around Conv2d because the public TorchAO
QAT workflow does not establish an automatic YOLO custom-Conv graph conversion. Every quantization result is
`simulation_only=true`; it is not evidence of real INT8 deployment. Report `Q0-Q1` PTQ gap, `Q0-Q2` QAT gap, and
`Q2-Q1` recovery. QAT gap at most 0.005 is robust, above 0.005 through 0.008 is acceptable but sensitive, and above
0.008 is quantization-sensitive.

## Required final report

After formal execution, `REPORT.md` must answer whether C1 warrants further e=0.25 work; compare C2 and C3; report
C3-P5 if triggered; report R1 recovery/fusion if triggered; compare C0/C_best PTQ and QAT gaps; and derive only the
next experiment supported by results. It must not claim that engineering thresholds are universal research standards.
