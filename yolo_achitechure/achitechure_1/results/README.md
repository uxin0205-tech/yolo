# Results

`results-template.csv` defines the required A0/A1/A2/T1–T3 report. A0 is populated from the immutable supplied
Bit-True parent. A1/A2/T1–T3 remain unclaimed until formal MASF training and COCO validation complete.

The historical 2026-08-17 smoke attempt stopped because another attention process occupied about 15.5 GiB VRAM.
With that queue complete, the unchanged batch-16/640 official-loss probes passed on 2026-08-18: Full35 peaked at
`18,397,301,760` bytes and Partial75 at `17,761,890,304` bytes. No smaller batch, accumulation, or P3 MASF MACs run
was used. This is historical capacity evidence only; the GPU is currently considered externally blocked and no
formal run may start without explicit clearance.
