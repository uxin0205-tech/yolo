# Repository Guidelines

## Project Structure & Module Organization

This directory is currently a planning-stage YOLO26m research workspace. Treat `plan.md` as the authoritative experiment and implementation specification. Place model modules under `ultralytics/nn/modules/` (for example, `p3_mfam.py`), model and training YAML files under a clearly named configuration directory, and automated checks under `tests/`. Keep generated runs, checkpoints, datasets, and benchmark artifacts out of source directories. When implementation begins, document executable experiment workflows in `RUNBOOK.md`.

Before changing terminology or architecture, read `../CONTEXT.md` and relevant records in `../docs/adr/` when they exist. Repository-wide agent guidance is in `../docs/agents/`.

## Build, Test, and Development Commands

- `python -m pip install -r ../requirements.txt` installs the pinned PyTorch, Ultralytics, and supporting packages.
- `python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"` verifies the active environment before experiments.
- `python -m pytest` runs the test suite once `tests/` is added.
- `git status --short --branch` checks that experiments have not introduced unintended files.

Do not invent or copy stale training commands. Add the validated A0/A1/A2 training, validation, and metrics-export commands to `RUNBOOK.md`, as required by `plan.md`.

## Coding Style & Naming Conventions

Use Python with four-space indentation, PEP 8 naming, type hints for public interfaces, and short docstrings for research-specific behavior. Name classes descriptively, such as `P3MFAMFull35` and `P3MFAMPartial75`; use `snake_case` for functions, files, and configuration keys. Prefer inheritance or composition over copying upstream Ultralytics modules. Keep architecture changes isolated and avoid unrelated loss, optimizer, augmentation, P2, P4, or P5 modifications.

## Testing Guidelines

Use `pytest`; name files `test_*.py` and tests `test_<behavior>`. Cover shape preservation, identity-safe initialization, finite gradients, model construction, three-scale Detect inputs, pretrained-weight transfer, smoke inference, and a short training step. Record parameter count, FLOPs when available, latency, and peak CUDA memory for A0/A1/A2.

## Commit & Pull Request Guidelines

Recent history favors concise imperative subjects such as `Add historical experiment records index`. Follow that style and avoid vague timestamp-only messages. Track specs and work in GitHub Issues using `gh`; link the issue in the pull request. PRs should summarize scope, list validation commands, report weight-transfer results and experiment metrics, and include screenshots only for visual artifacts. Explicitly disclose assumptions, unresolved conflicts, hardware, seed, batch size, and configuration changes.
