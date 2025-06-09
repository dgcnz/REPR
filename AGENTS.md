# AGENTS Instructions

This repository is a Python project organised as a package inside `src/`. It uses
Hydra configuration, Lightning, and various utilities for training and
benchmarking vision models. Tests live in `tests/` and rely on PyTorch.

## Repository structure

- `fabric_configs/` contains configuration files.
- `src/models/components/partmae_v6/` hosts the latest model version.
- `scripts/` provides standalone single-use scripts.
- `tests/` holds unit tests.

## When modifying the code

* Prefer jaxtyping annotations for type hints.
* Use docstrings with the ``:param name:`` style.
* Place new modules under `src/`.
* Avoid modifying anything under `third-party/` or `artifacts/`.
* Do not commit large binaries or model checkpoints (e.g. ``*.ckpt``, ``*.pt``).
* Only add tests when explicitly asked to do so.

## Linting

Run `ruff` to lint the code and apply fixes with `ruff --fix` when needed.

```bash
ruff .
```

## Testing

Run tests only when a specific file is requested. For example:

```bash
pytest tests/test_feat.py
```
