"""Each study lives in experiments/<name>/ and must not import another study."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments"
REQUIRED = ("PROTOCOL.md", "README.md", "train.py")


def _experiment_dirs() -> list[Path]:
    return sorted(
        p
        for p in EXP_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
    )


def test_experiments_exist_and_are_named():
    names = [p.name for p in _experiment_dirs()]
    assert "mnist_matched_size" in names
    assert "cifar_scale_gap" in names


def test_each_experiment_has_the_standard_files():
    dirs = _experiment_dirs()
    assert dirs, "expected at least one experiments/<name>/"
    for exp in dirs:
        for name in REQUIRED:
            assert (exp / name).is_file(), f"{exp.name} missing {name}"


def test_experiments_do_not_import_each_other():
    names = [p.name for p in _experiment_dirs()]
    for exp in _experiment_dirs():
        others = [n for n in names if n != exp.name]
        for py in exp.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            for line in py.read_text().splitlines():
                stripped = line.strip()
                if not stripped.startswith(("import ", "from ")):
                    continue
                for other in others:
                    assert f"experiments.{other}" not in stripped, f"{py} imports {other}"
                    assert f"experiments/{other}" not in stripped, f"{py} imports {other}"


def test_default_out_dir_is_local_results():
    for exp in _experiment_dirs():
        text = (exp / "train.py").read_text()
        assert "EXP_DIR / \"results\"" in text or "EXP_DIR / 'results'" in text
        assert "artifacts" not in text
