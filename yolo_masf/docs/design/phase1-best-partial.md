# Phase 1 BEST_PARTIAL Design

The workflow owns one immutable dependency DAG: dataset audit, environment and
model verification, five-variant GPU preflight, common batch probing, B1-A,
B1-B, four engineering smokes, four formal variant runs, validation,
BEST_PARTIAL selection, test evaluation, profiling, final audit, and reporting.
Every node declares hashes and success gates; only a strict-valid completed node
may be reused.

The fixed YOLO11m P2 template retains the official backbone and shared P3–P5
neck/head semantics while adding a P2 top-down/bottom-up path and four-scale
Detect head. Repository-owned slots are installed before the first forward. B1
uses Identity; M0 sums identity with 3/5/7/9 depthwise branches before 1×1
fusion; M1 retains only 3/5; M2 and M3 apply M1 to the leading half or quarter
of channels and concatenate an untouched bypass.

Dataset records from the existing train/valid union are grouped by normalized
Roboflow origin, source/video identity, and content hash. Seed-42 deterministic
greedy assignment targets 80/10/10 unique frames, class instances, and ball
size bins. Any group/hash overlap, missing class, malformed label, or fewer than
50 val/test balls stops the pipeline before GPU use.

All dynamic decisions use validation. `selection.json` is frozen before any
test call. M3 wins the efficiency-equivalent boundary only when M2/M3 differ by
at most 0.002 mAP50-95 and strictly less than 0.01 Ball Recall; otherwise the
documented quality then hardware ranking is applied. Test metrics and profiles
cannot influence selection.
