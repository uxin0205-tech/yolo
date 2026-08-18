# Parent lineage

- Original copy date: 2026-08-17.
- Completed delivery reconciled: 2026-08-18.
- Formal source: `/home/uxin/yolo/yolo_attention_final/final/pwl-final-best.pt`.
- Local immutable copy: `inputs/parent/best.pt`.
- Checkpoint SHA256: `c989aeed09de7663ad093d32d098e5fc889cf04924fa1162efaf886869de0123`.
- Source project revision: `3e4c5dc983ce5fe52e64e45e277883572b0648fe`.
- Ultralytics: `8.4.90`, source revision `ea920761b1c6d531c2231d0033714301690cf67d`.
- Python: `3.12.3`; PyTorch: `2.11.0+cu128`; CUDA runtime: `12.8`.
- Source GPU: NVIDIA GeForce RTX 5090.

The final source checkpoint and the pre-existing local parent are byte-identical, so the local `.pt` was not
reserialized or replaced. A fresh process successfully loaded it as YOLO26m scale `m`, `end2end=True`, with three
Detect inputs and the two expected hardware-friendly attention sites.

## Completed attention study decision

The source study completed revision 95 with 19/19 queue jobs successful. The formal selector retained the verified
zero-train Bit-True checkpoint at mAP50-95 `0.5067368995935831`, mAP50 `0.677047718215406`, and mAP75
`0.5551035937976228`.

The best trained seed-0 observation was `lr-block-x1-bittrue`, SHA256
`c70cd1d0315518d61b6ac6b5936173f04dc5b0eadcb86b288699655434dcf9fb`, at mAP50-95
`0.5069386684559859`. Its `+0.0002017688624028` gain was below the `+0.001` formal gate, and no three-seed mean was
run. It is therefore documented but deliberately not copied or used as the MASF parent.

`provenance/` contains exact snapshots of the final manifest, metrics, training description and recipe,
requirements, complete queue state, and event history. `final-selection.json`, `epoch0-float-manifest.json`, and
`epoch0-bittrue-manifest.json` preserve the corresponding selector and evaluation records.
