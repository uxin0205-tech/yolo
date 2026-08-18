# achitechure_1

Isolated YOLO26m Attention + P3 MASF experiment implementation. Read `EXPERIMENT_SPEC.md` for the fixed contract
and `RUNBOOK.md` for executable commands.

Current state: the completed attention study has been imported and validated as the immutable parent. Model
construction, pretrained transfer, Float/Bit-True conversion, phase scopes, MuSGD groups, Bit-True early stopping,
profiling, validation, and export are implemented and tested. Full35 and Partial75 completed one fixed
batch-16/640 official-loss smoke check, but the GPU is currently treated as externally blocked per the workspace
owner. Formal MASF training has not started, and no accuracy improvement is claimed.
