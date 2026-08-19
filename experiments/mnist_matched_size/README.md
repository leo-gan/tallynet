# mnist_matched_size

Iso-memory MNIST: float MLP vs TallyMLP under a matched training footprint.

This folder is standalone. Outputs go to `results/` here, not under another experiment.

```bash
uv run --extra train python experiments/mnist_matched_size/train.py --quick
uv run --extra train python experiments/mnist_matched_size/train.py --scale
uv run pytest -q experiments/mnist_matched_size
```

Protocol: [PROTOCOL.md](PROTOCOL.md). After a run, `results/latest.csv` and `results/latest.manifest.json` are enough to see what was executed (commit, knobs, versions).
