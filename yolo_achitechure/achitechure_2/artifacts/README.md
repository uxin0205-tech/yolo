# Artifact contract

Generated artifacts are intentionally ignored. Formal execution creates:

- `intake/accepted.json`: immutable handoff gate with both checkpoint hashes and fresh-process graph reports.
- `candidates/<id>/`: candidate checkpoint, lineage, graph, and tensor transfer report.
- `runs/<run-id>/`: resolved arguments, git/config/checkpoint hashes, best/last checkpoints, metrics, best epoch,
  stop reason, latency/GFLOPs/Params, and peak CUDA memory when available.
- `selection.json`: C0-relative decisions, conditional triggers, and `C_best` (or `null`).
- `quant/<id>/`: Q0/Q1/Q2 metrics and `quant-scope.json`; every first-version result says
  `simulation_only=true`.

Do not copy datasets, checkpoints, Ultralytics source, or generated runs into `src/`.
