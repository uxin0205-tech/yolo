# Final YOLO26m Bit-True PWL delivery

This folder keeps one model checkpoint and one command surface.

- `pwl-final-best.pt`: the only `.pt`; formal zero-train winner, mAP50-95 `0.5067369`.
- `run.py`: check, predict, print the recipe, launch/resume queued training, and show queue status.
- `training_recipe.py`: executable LR, unfreezing, early-stop, Bit-True gate, and rollback recipe.
- `yolo_attention/`: minimal executable Attention algorithm used by the checkpoint.
- `TRAINING.md`: algorithm, training method, and exact commands.
- `RESULTS.md`, `results.json`, `metrics.csv`: measured results and provenance.

## Quick start

```bash
python run.py check
python run.py predict path/to/images --device 0 --save
python run.py recipe
python run.py train                 # dry-run only
python run.py train --execute       # explicit GPU queue execution
python run.py status
```

The single checkpoint is official YOLO26m scale m with 80 classes and two Bit-True PWL Attention sites.
It uses Hadamard Binary Q/K, power-of-two fixed scale, decomposed 2D bias, Q8.8 scores in `[-10, 0]`,
20 segments, and 21 UQ1.15 endpoints. The denominator remains an exact floating reference.

The trained block-x1 candidate measured `0.5069387`, but only seed 0 was run and its gain was below the
formal `+0.001` gate. It remains preserved in the parent project's immutable artifacts, not duplicated here.
