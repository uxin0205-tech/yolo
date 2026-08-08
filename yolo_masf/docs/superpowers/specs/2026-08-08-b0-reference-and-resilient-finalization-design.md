# B0 Reference and Resilient Training Finalization Design

## Goal

Resume Phase 1 from the completed B1-B training output without repeating its 90 epochs, prevent completed runs from being resumed as unfinished runs, add the existing pose-derived detection checkpoint as a fully measured B0 reference, and prioritize a new 3/5/7-equivalent MFAM ablation.

## Confirmed Context

- B1-A completed 10 epochs and has a canonical strict checkpoint.
- B1-B completed all 90 epochs. Its `results.csv` has 90 data rows and its loadable `best.pt` and `last.pt` are each approximately 40 MB.
- B1-B was killed during Ultralytics' duplicate final validation, after weights were stripped and after epoch 90 validation had already completed.
- Kernel evidence records a global OOM kill, a 13.1 GB service memory peak, 2.7 GB swap peak, and 88 leaked multiprocessing semaphores.
- On restart, the pipeline treated the completed stripped `last.pt` as resumable. Ultralytics correctly rejected it with “training to 90 epochs is finished, nothing to resume.”
- B0 is `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`, SHA-256 `9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d`.
- B0 metadata shows Ultralytics 8.4.90, classes `ball` and `bat`, strides `[8, 16, 32]`, and a detection model produced from a BBT5 pose-trained checkpoint by replacing/removing the pose head for detection use.

## Research Contract

B0 is an external reference, not a Phase 1 training variant. It is evaluated and profiled with exactly the same generated val/test manifests, image size, confidence handling, COCO evaluator, class metrics, ball size bins, blur proxy, and hardware profiler used for B1/M7/M0/M1/M2/M3.

Because its pose-derived source was trained on BBT5 before the leak-free 80/10/10 union split existed, B0 is labelled `pose-derived, data-exposed reference`. It must not:

- participate in BEST_PARTIAL selection;
- initialize B1 or any MFAM model;
- satisfy a trained-variant checkpoint gate;
- support a claim of leak-free improvement over B1/M7/M0–M3.

Its val and test numbers remain useful as an operational detection reference and are displayed in all comparison tables with the exposure warning.

## M7 Priority Variant

M7 is a new full-channel P2-slot training variant. Its parallel sum is:

`X + DWConv3(X) + DWConv5(X) + DWConv7x1(DWConv1x7(X))`

The sum is followed by the same Ultralytics 1x1 Conv fusion used by M0 and M1. M7 has no 9-equivalent branch, partial-channel bypass, shuffle, gate, or extra projection. It initializes from the same B1-B canonical EMA as every other MFAM variant; only its MFAM-owned tensors start randomly.

M7 runs first after B1-B recovery: a 3-epoch engineering smoke followed by a 100-epoch formal run. Only after M7 passes its canonical strict-reload gate does the pipeline continue through M0, M1, M2, and M3. M7 receives the same val/test metrics and hardware profile as every trained variant. It is an ablation result and does not change the M2/M3-only BEST_PARTIAL rule.

The resulting formal order is:

`B1-B recovery -> smoke M7 -> formal M7 -> smoke M0/M1/M2/M3 -> formal M0/M1/M2/M3`

The complete planned training matrix is 615 epochs: B1-A 10, B1-B 90, M7 smoke/formal 3+100, and M0-M3 smoke/formal 4x(3+100).

## B0 Definition and Pipeline Placement

A separate `configs/b0-reference.yaml` stores the reference ID, checkpoint path, exact SHA-256, expected class names, expected strides, provenance, and exposure status. Keeping this outside `configs/static-phase1.yaml` preserves the existing Phase 1 config hash and pipeline ID, so completed B1 work remains reusable.

A new `baseline_b0` DAG stage is inserted after `formal_m3` and before `val_all`. It performs a fresh-process checkpoint inspection and writes `artifacts/static-phase1/references/b0.json`. The stage fails closed if the hash, task, class order, stride count, or checkpoint load differs from the reference definition.

`val_all`, `test_all`, `profile_all`, final audit, and the human-readable report use an evaluated-model set of `B0, B1, M7, M0, M1, M2, M3`. Training, preflight, common-batch probing, transfer, and canonical checkpoint requirements use `B1, M7, M0, M1, M2, M3`. Selection candidates remain exactly `M2, M3`.

## Completed-Run Recovery

Before constructing a model or invoking Ultralytics, each training stage classifies its output directory as one of:

1. `absent`: no usable `last.pt`; start a new run.
2. `incomplete`: a loadable `last.pt` plus fewer than the configured number of `results.csv` rows; use native resume.
3. `complete`: loadable `best.pt` and `last.pt`, exactly the configured number of result rows, matching checkpoint training arguments, and finite final metrics; skip training and proceed directly to canonical finalization.
4. `invalid`: contradictory, unreadable, overlong, non-finite, or mismatched artifacts; fail permanently without deleting or modifying them.

This makes the existing B1-B output a recoverable completed run. The recovery report records why it was accepted and hashes both weights and `results.csv`.

## Memory-Safe Stage Boundaries

Training and finalization are separated:

- Ultralytics still performs validation every epoch with the locked training arguments.
- The repository trainer overrides only Ultralytics' redundant `final_eval`: it retains official optimizer stripping but does not run the second best-checkpoint validation inside the memory-heavy training process.
- Canonical checkpoint construction runs in a fresh subprocess after training objects and dataloader workers have exited.
- Existing strict reload and fresh val run in another subprocess and remain mandatory.
- The systemd service uses `OOMPolicy=continue`; a killed worker subprocess becomes a bounded transient stage failure instead of killing the lightweight orchestrator.
- Completed-output detection runs again after any interrupted subprocess, so a kill after epoch completion can advance to finalization rather than attempt an invalid resume.

No loss, augmentation, optimizer, batch, image-size, or epoch policy changes are introduced.

## Artifacts and Audit

For every canonical training stage, `run.json` additionally records the training-output classification, results hash, best hash, and last hash. B0 writes provenance and exposure metadata but never a canonical training manifest.

Final audit requires:

- canonical strict artifacts for B1-A/B and formal M7/M0–M3;
- val/test metrics and profiles for all seven evaluated models;
- a B0 reference record with the expected checkpoint hash and `data_exposed=true`;
- an immutable M2/M3-only selection;
- zero audit errors.

The report places B0 in the result tables and adds a prominent warning explaining why its metrics are not a leak-free comparison.

## Testing and Restart

Tests are written first for:

- absent, incomplete, complete, and invalid run classification;
- completion detection for stripped Ultralytics checkpoints;
- no resume call for a completed 90-epoch run;
- native resume for an incomplete run;
- final-eval stripping without duplicate validation;
- subprocess finalization and failure propagation;
- B0 hash/provenance/stride/class gates;
- M7 branch shape, numerical sum, gradient flow, transfer ownership, hash, profile equivalence, and priority DAG order;
- separation of trained variants, evaluated models, and selection candidates;
- final audit and reporting requirements for B0;
- systemd `OOMPolicy=continue`.

After the focused and full test suites pass, the current B1-B artifacts are classified and finalized in place. The pipeline is restarted only after tracked files are committed and pushed. Acceptance for the restart is: B1-B becomes `completed` without adding a 91st result row, strict fresh validation passes, and the service advances into `smoke_m7` with no B1 retraining.
