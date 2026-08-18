# YOLO26m PWL Final Training

The final workflow trains a differentiable Float-PWL surrogate and evaluates separately materialized Bit-True
PWL checkpoints. The completed staged-recovery queue progressively tested the containing blocks, Neck/Detect,
the final Backbone stage, and the full model with discriminative learning rates and locked BatchNorm running
statistics.

## Vocabulary

- **Parent**: immutable retained `v1-br` checkpoint.
- **Float-PWL**: differentiable PWL exponential surrogate used only during training.
- **Bit-True PWL**: Q8.8/UQ1.15 integer numerator reference used for gates and delivery.
- **Phase gate**: accepts a child unless it loses more than 0.001 mAP50-95 to its direct parent.
- **Recovery winner**: highest Bit-True result from one staged trajectory; exported as best-observed.
- **Formal winner**: trained winner only when the three-seed mean improves on the zero-train parent by 0.001.

## Completed decision

The LR sweep/recovery queue finished 19/19 jobs. Block x1 was the best observed candidate at `0.5069387`
mAP50-95, but its gain over the audited parent was only `0.0001944`. Seed 1/2 were intentionally not run, so
the formal winner remains the zero-train fallback `final/pwl-final-best.pt`. The portable delivery is in `final/`.
