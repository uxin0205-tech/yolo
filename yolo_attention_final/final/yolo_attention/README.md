# Minimal checkpoint runtime

This package contains only the source modules required to unpickle and execute the delivered Bit-True checkpoints.
It is intentionally separate from the training and queue implementation.

- `attention.py`: complete Attention dataflow.
- `binary_basis.py`: Hadamard Binary Q/K score block.
- `normalization.py`: Bit-True Q8.8/UQ1.15 PWL normalization block.
- `projection.py` and `relative_bias.py`: Q/K/V projection and decomposed 2D bias blocks.
- `config.py`: serialized model contract.
- `quantization.py`, `schedule.py`, and `bdcn.py`: import-compatible support blocks.

Each module starts with a short block-level docstring. These files are byte-for-byte copies of the corresponding
production modules under `src/yolo_attention/`; use `../run.py` as the single public entrypoint.
