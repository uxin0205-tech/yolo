# YOLO26m PWL Attention Final

This project contains the completed Float-PWL training and Bit-True evaluation workflow for the retained V1-BR
YOLO26m parent.

Fixed contract: official `yolo26m.yaml`, scale `m`, COCO 80 classes, Attention sites
`model.10.m.0.attn` and `model.22.m.0.1.attn`, Q8.8 scores in `[-10, 0]`, 20 half-unit segments,
and 21 UQ1.15 endpoints (336 bits per site). The denominator remains an exact floating reference.

## Final result

The authoritative LR sweep and staged-recovery queue completed 19/19 jobs. Block x1 produced the highest
measured Bit-True mAP50-95, `0.5069387`, versus audited parent `0.5067442`. Its `+0.0001944`
gain is below the `+0.001` robustness gate and only seed 0 was run, so the formal winner remains the
verified zero-train fallback at `0.5067369`.

- Clean delivery with the only formal checkpoint and one command surface: [`final/`](final/README.md)
- Algorithm and executable training method: [`final/TRAINING.md`](final/TRAINING.md)
- Authoritative measured-results report: [`final/RESULTS.md`](final/RESULTS.md)
- Machine-readable summary: [`final/results.json`](final/results.json)
- Method rationale and primary sources: [`docs/research/pwl-final-training-rationale.md`](docs/research/pwl-final-training-rationale.md)

## Use the delivery

```bash
cd final
python run.py check
python run.py predict path/to/images --device 0 --save
python run.py recipe
python run.py train                 # dry-run
python run.py train --execute       # explicit queue/GPU execution
python run.py status
```

## Inspect the completed research queue

```bash
PYTHONPATH=src python -m yolo_attention.cli queue validate --queue-root artifacts/lr-sweep-queue
PYTHONPATH=src python -m yolo_attention.cli queue status --queue-root artifacts/lr-sweep-queue
```

Training used AdamW, AMP, deterministic seed 0, constant discriminative learning rates, no warmup, locked
BatchNorm running statistics, early stopping, and a `0.001` phase rollback tolerance. Full COCO2017 metrics
use the Ultralytics internal evaluator; canonical COCO API was not a required gate.
