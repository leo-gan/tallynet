# Experiments

One directory per study. Studies do **not** import each other. Shared code lives in `tallynet/`. Shared on-disk data lives in repo `data/` (gitignored). Default `pytest` does not collect this tree.

```
experiments/
  <name>/
    PROTOCOL.md     # question, grid, how to read the result
    README.md       # how to run / reproduce this study only
    train.py        # only entry point
    test_*.py       # smoke for this study
    analysis.py     # optional, local to this study
    results/        # gitignored outputs (this study only)
```

| Dir | Question |
|-----|----------|
| [mnist_matched_size](mnist_matched_size/) | Iso-memory MNIST: does Tally beat float? |
| [cifar_scale_gap](cifar_scale_gap/) | Iso-shape CIFAR: does the Tally–float gap shrink with width? |

## Rules

- **No cross-imports.** `cifar_scale_gap` must not import `mnist_matched_size` (or the reverse), including via `sys.path`.
- **No shared results folder.** Default `--out-dir` is `experiments/<name>/results/`.
- **Reproduce from the folder.** `train.py` writes `latest.csv` plus a `*.manifest.json` (argv, resolved knobs, git commit, torch / tallynet versions, native kernel).
- Cite another study in `PROTOCOL.md` if you need to. Do not call its Python.

```bash
uv run --extra train python experiments/mnist_matched_size/train.py --quick
uv run --extra train python experiments/cifar_scale_gap/train.py --quick
uv run pytest -q experiments/mnist_matched_size experiments/cifar_scale_gap
```
