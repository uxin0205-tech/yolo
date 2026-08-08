# MASF-YOLO Phase 1

This work item implements the static MASF-YOLO ablation described in
`MFAM_plan.md`: B1 (YOLO11m + P2), M0, M1, M2, M3, and one validation-selected
BEST_PARTIAL. It deliberately excludes S0 and Temporal Motion.

The implementation is repository-local and does not patch Ultralytics. Source
data and `../original/weight/yolo11m.pt` are read-only; generated manifests,
runs, checkpoints, evaluations, and reports are written beneath `artifacts/`.

```bash
../.venv/bin/python -m masf_yolo.cli audit --config configs/static-phase1.yaml
../.venv/bin/python -m masf_yolo.cli verify --config configs/static-phase1.yaml
../.venv/bin/python -m masf_yolo.cli pipeline start --config configs/static-phase1.yaml
../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml
../.venv/bin/python -m masf_yolo.cli report --config configs/static-phase1.yaml
```

See `OPERATIONS.md` for failure and recovery procedures and
`docs/design/phase1-best-partial.md` for the locked design.
