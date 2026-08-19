# achitechure_1

Isolated YOLO26m Attention + P3 MASF experiment implementation. Read `EXPERIMENT_SPEC.md` for the fixed contract
and `RUNBOOK.md` for executable commands.

Current state: the completed attention study has been imported and validated as the immutable parent. Model
construction, pretrained transfer, Float/Bit-True conversion, phase scopes, MuSGD groups, Bit-True early stopping,
profiling, validation, and export are implemented and tested. Full35 and Partial75 completed one fixed
batch-16/640 official-loss smoke check on an RTX 5090. The current RTX 4080 SUPER is a development/smoke host;
formal MASF training and final comparable profiling are reserved for the RTX 5090. Peak VRAM is reported only as a
capacity diagnostic and does not rank candidates. Formal training has not completed, and no accuracy improvement is
claimed.
