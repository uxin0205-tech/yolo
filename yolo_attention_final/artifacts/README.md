# Artifacts

Runs are immutable and preserve the evidence behind the reported metrics.

- `lr-sweep-queue/`: completed authoritative queue, revision 95, 19/19 jobs succeeded.
- `runs/lr-*/`: LR pilots, Bit-True evaluations, staged recovery, gates, and final observed selection.
- `runs/s0-phase-b-bittrue/`: the only run directory included in Git; required immutable parent for
  `final/run.py train --execute` after a fresh clone.
- `runs/epoch0-*/`, `runs/pilot-*/`, `runs/s0-*/`: retained parent lineage and negative controls.
- `queue/`: stopped attention-only multi-seed history; retained as provenance and cannot start automatically.

The excluded preliminary `recovery-queue/`, its duplicate parent evaluation, and its interrupted block run were
removed during final cleanup. The human-facing report is `../final/RESULTS.md`; `final-selection.json`
records why the formal checkpoint in `../final/` was retained while block x1 remains best-observed.
