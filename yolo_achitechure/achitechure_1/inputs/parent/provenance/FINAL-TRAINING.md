# Algorithm and training method

## Attention algorithm

The delivered model replaces exactly two YOLO26m Attention sites:

- `model.10.m.0.attn`
- `model.22.m.0.1.attn`

Q and K are converted to deterministic binary signs, evaluated in identity and Walsh-Hadamard bases, and
combined with fixed power-of-two coefficients. A decomposed two-dimensional relative-position bias is added.
Normalization centers each row, rounds and saturates the score to signed Q8.8 in `[-10, 0]`, indexes 20
half-unit segments, and linearly interpolates between 21 UQ1.15 endpoints. The numerator path is Bit-True;
the denominator remains an exact floating-point row sum and reciprocal.

The executable blocks are under `yolo_attention/`: `attention.py` is the dataflow, `binary_basis.py`
implements Binary Q/K and Hadamard scoring, `normalization.py` implements Float and Bit-True PWL,
`projection.py` splits Q/K/V, and `relative_bias.py` implements the decomposed bias.

## Training method

Training cannot differentiate through Bit-True rounding and integer indexing, so optimization uses Float-PWL
with the same `[-10, 0]` range and 20 segments. Every candidate `best.pt` is then separately reconfigured
and evaluated as an actual Bit-True checkpoint on all 5,000 COCO2017 validation images.

The queue performs:

1. A same-parent block LR sweep at x1/x2/x4. Attention LRs are `5e-6`, `1e-5`, `2e-5`; adjacent
   block LRs are `1e-6`, `2e-6`, `4e-6`. Maximum 8 epochs, patience 3.
2. Neck/Detect recovery with Attention `5e-6`, adjacent block `1e-6`, Neck/Detect `5e-7`.
   Maximum 16 epochs, patience 5.
3. Backbone-last recovery, adding Backbone LR `1e-7`. Maximum 16 epochs, patience 5.
4. Full-model recovery at the same discriminative LRs. Maximum 20 epochs, patience 6.

All stages use seed 0, AdamW, batch 16, imgsz 640, workers 8, AMP, deterministic execution, weight decay
`5e-4`, constant LR, and no warmup. BatchNorm affine parameters may train when in scope, but every running
mean, variance, and counter is locked. Each phase compares the real Bit-True child with its direct parent and
rolls back when the loss exceeds `0.001` mAP50-95. The final selector compares every Bit-True candidate.

## How to run

From this directory:

```bash
# Validate the only delivered checkpoint.
python run.py check

# Print the complete recipe without changing state.
python run.py recipe
python run.py train

# Initialize/resume a new immutable queue and execute GPU jobs.
python run.py train --execute

# Inspect that reproduction queue.
python run.py status

# Run inference.
python run.py predict path/to/images --device 0 --save
```

The training wrapper deliberately defaults to dry-run. It invokes the tested production engine in the parent
repository and checks every required input before queue creation. Exact reproduction also needs the retained
historical Phase-B parent in `../artifacts/runs/`; it is not duplicated here because this delivery is required
to contain only one `.pt` file.
