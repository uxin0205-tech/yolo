# BinaryQK README and History Consolidation Design

## Goal

Turn `yolo_binaryqk/README.md` into the self-contained entry point for the
completed BinaryAttention study, then replace every commit after `d16a898`
with one commit named `first research` whose tree matches the latest `main`.

## Documentation design

The README uses progressive disclosure:

1. Explain the research question, scope, and the distinction between zero-shot
   validation, attention-only QAT fine-tuning, KD, bias, and magnitude studies.
2. State the reproducibility contract: YOLO11m, COCO2017, 640 px, 10 epochs,
   attention-only training, seed 0, and the evaluator distinction.
3. Summarize the complete 26-variant matrix and the principal numeric results.
4. Map every top-level directory, Python module, script, report, and canonical
   weight bundle to its maintenance responsibility.
5. Provide verified commands for testing, audit, report regeneration, area AP,
   weight verification, and full-matrix reruns.
6. Link to the full generated report and operations manual for details that
   should not be duplicated manually.

Generated result files remain untouched. The README reports values from the
current canonical report and explicitly distinguishes Ultralytics `mAP50-95`
from official COCOeval metrics.

## Git history design

- Preserve `d16a898` as the initial repository commit.
- Preserve the pre-rewrite `main` tip in a local backup branch.
- Build one replacement commit from the final verified working tree, with
  parent `d16a898` and subject exactly `first research`.
- Update `main` using `git push --force-with-lease`, protecting against an
  unexpected remote change.
- Exclude unrelated untracked datasets `original/pose/bbt5.v1i.yolov8/` and
  `pose_dataset/` from the replacement tree.

## Verification

- Run all BinaryQK tests.
- Run the formal audit and require `ok=true`, 26 verified variants, 26 archived
  weights, 26 area-metric records, and zero errors.
- Check Markdown links in the README resolve locally.
- Run `git diff --check` before committing.
- After rewriting and pushing, verify local `main` and `origin/main` point to
  the same commit and the visible history is `first research` over `d16a898`.
