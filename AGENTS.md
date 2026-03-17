# Repository Guidelines

## Project Structure & Module Organization
- `training/` — Public training entry points (`train.py`, `train_pipeline.py`, `train_support.py`).
- `models/` — JSCC encoders/decoders, vocoder modules, and quantization blocks.
- `utils/` — Data loaders, channel simulation, visualization, audio I/O.
- `tests/` — `unittest`-based checks; designed to run locally without large datasets.
- `scripts/` — Public inference helpers (`infer_wav.py`, `infer_features.py`).
- `configs/` — Public JSON configs for single-run resume and multistage training.
- `docs/` — Public training guide and upload manifest.
- Data/artifacts: `data/` (dataset root placeholder), `vocoder_pt/` (pretrained vocoder), `artifacts/`, `checkpoints/`.

## Build, Test, and Development Commands
- Single-GPU training:  
  `python training/train.py --data_root ./data --out_dir ./runs/debug_run --vocoder_ckpt ./vocoder_pt/vocoder_sq1Ab_adv_50.pth`
- Multi-stage training:  
  `python training/train_pipeline.py --config configs/multistage_training.json`
- Run tests:  
  `python tests/run_tests.py` or `python -m unittest discover -s tests -p "test_*.py"`
- CUDA check:  
  `python -c "import torch; print(torch.cuda.is_available())"`

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, PEP 8 compliant.
- Naming: `snake_case` (modules/functions), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Files: public entry points use direct names such as `train.py`; tests use `tests/test_*.py`.
- Prefer type hints, docstrings for public APIs, and f-strings for formatting.

## Testing Guidelines
- Framework: `unittest`; keep tests deterministic and CPU-friendly.
- Naming: place tests under `tests/` with `test_*.py`; include a `main()` guard if directly runnable.
- Coverage focus: data loaders, channel simulation, loss functions, model forward passes, and training entry points.

## Commit & Pull Request Guidelines
- Commits: imperative, concise titles (e.g., "Fix DDP init on rank 0"); add context when changing training defaults or I/O.
- PRs: describe purpose, key changes, how to run (commands + minimal args), and expected outputs (sample logs/metrics). Link issues and attach screenshots for visualizations.

## Security & Configuration Tips
- Use a dedicated conda env with CUDA-enabled PyTorch; set `CUDA_VISIBLE_DEVICES` for multi-GPU and prefer `torchrun`.
- Keep large datasets out of Git; store generated checkpoints under `runs/` or `checkpoints/` and ignore them with `.gitignore`.
