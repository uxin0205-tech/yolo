---
status: accepted
---

# Share the Full35 trunk while retaining independent Detect and Pose heads

F1 uses one live copy of the Full35 layers 0–22 feature trunk to feed the existing Detect head and an independent ball/bat Pose26 head. This removes the duplicate Pose-side trunk without averaging, pruning, quantizing, or overwriting the Authoritative Full35 Source; F0.5 retains both complete models as the accuracy baseline and rollback path.

## Evidence and scope

- The Independent Task Pair contains 45,580,762 parameters; F1 contains 26,529,701. The removed duplicate trunk accounts for 19,051,061 parameters, a 41.796% reduction.
- Detect-to-Pose trunk initialization is strict: 587/587 tensors match by name and shape before copying.
- In evaluation mode on the same GPU input, both Detect and Pose nested outputs pass `rtol=0, atol=0` against independent task graphs whose trunks contain the same 587/587 tensors. F1 can also be materialized back to official task graphs with exact outputs.
- P1 trains only the Pose head, so it preserves this precondition. P2/P3 train the independent Pose neck/trunk; after those weights diverge from Detect, no single shared trunk can reproduce both independent graphs exactly.
- **Initialization Equivalence** is therefore not a permanent accuracy claim. Joint training intentionally updates one shared trunk using both task losses, and every trained F1 must pass the Accuracy Gate.

## Consequences

- The accepted Full35 attention/PWL/MASF structure and both task-specific heads remain present; source bundle files and independent checkpoints remain unchanged.
- The standard factory creates a new F1 experiment instance and saves a separate checkpoint. The lower-level constructor consumes its in-memory task modules, so a frozen baseline must be loaded as a separate instance rather than aliased to the modules being trained.
- When both tasks are requested, P3/P4/P5 are computed once instead of running two complete trunks. The 41.796% parameter reduction is real, but it must not be reported as the same percentage of latency, MAC, energy, or FPGA resource savings until those are measured.
- Every trained F1 candidate must be compared with the corresponding independent Detect and Pose baselines. A drop greater than 0.08 mAP50-95 in any tracked metric rejects that candidate; F0.5 remains available if hard sharing cannot pass the gate.
