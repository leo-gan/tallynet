# TallyNet: Python package + C++ popcount kernels.
#   make install     venv + editable install + native kernel
#   make native      compile kernels only
#   make test        pytest (builds kernel first)
#   make wheel       portable wheel + kernel artifact
#   make clean

PYTHON ?= $(CURDIR)/.venv/bin/python
export PYTHON

.PHONY: install install-cpu native test lint bench wheel clean

install:
	./scripts/install.sh

install-cpu:
	./scripts/install.sh --cpu-torch

install-train:
	./scripts/install.sh --train

native:
	./scripts/build_native.sh

test: native
	$(PYTHON) -m pytest -q --junitxml=artifacts/junit.xml

lint:
	$(PYTHON) -m ruff check tallynet tests scripts

bench: native
	$(PYTHON) scripts/bench_decode.py

wheel:
	./scripts/package.sh

clean:
	rm -rf .kernel_cache build dist artifacts .pytest_cache
	rm -f tallynet/lib/*.so
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
