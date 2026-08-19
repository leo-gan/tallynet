# cifar_scale_gap

Iso-shape CIFAR-10: does the Tally–float accuracy gap shrink as width grows?

This folder is standalone. Outputs go to `results/` here, not under another experiment.

```bash
uv run --extra train python experiments/cifar_scale_gap/train.py --quick
uv run --extra train python experiments/cifar_scale_gap/train.py --scale
uv run --extra train python experiments/cifar_scale_gap/train.py --expand-h
uv run --extra train python experiments/cifar_scale_gap/train.py --expand-s
# after a live --scale process: wait, then apply EXPAND H then EXPAND S
./experiments/cifar_scale_gap/run_campaign.sh --wait-pid <pid>
uv run pytest -q experiments/cifar_scale_gap
```

Protocol: [PROTOCOL.md](PROTOCOL.md). After a run, `results/latest.csv` and `results/latest.manifest.json` record the exact knobs, commit, and library versions. `--expand-*` resumes from this folder’s `results/latest.csv` only.
