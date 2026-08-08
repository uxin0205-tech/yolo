# Phase 1 Operations

Use the pinned interpreter at `../.venv/bin/python`. `pipeline start` performs
read-only environment, data, Git, CUDA, hash, and systemd checks before it
creates one user service. A second invocation for the same pipeline ID must
report the existing run rather than launch a duplicate.

Transient CUDA OOM or process interruption may resume the same native
`last.pt` at most three times. Data leakage, malformed labels, hash mismatch,
architecture mismatch, non-finite loss, or a failed research gate is permanent
and requires an operator correction; the workflow never changes batch size,
hyperparameters, or architecture in response.

`state.json` is written atomically. A completed stage is reusable only when all
input/output hashes match and its strict fresh-process checkpoint gate passed.
The canonical best EMA is a CPU float32 state dict, not a pickled live module.

Useful commands:

```bash
../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml
systemctl --user status masf-yolo-phase1-<pipeline-id>.service
journalctl --user -u masf-yolo-phase1-<pipeline-id>.service -f
../.venv/bin/python -m masf_yolo.cli report --config configs/static-phase1.yaml
```

Phase 1 ends after test evaluation, profiling, final audit, and report. It never
starts S0 or Temporal Motion. Closing Codex does not terminate the user service.
