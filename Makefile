# Shortcuts for the things you run often. `make` on its own lists them.
#
# Everything goes through `poetry run`, so these work without activating the
# environment first.

.DEFAULT_GOAL := help
.PHONY: help setup test lint format data train analyze figures all quick clean clean-runs

RUN  ?= baseline
DATA ?= main

help:  ## show this list
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## create the virtualenv and install everything from poetry.lock
	poetry install
	@echo
	@echo "interpreter: $$(poetry env info --path)/bin/python"

test:  ## run the property tests
	poetry run pytest

lint:  ## check style and imports
	poetry run ruff check .

format:  ## fix what ruff can fix automatically
	poetry run ruff check --fix .

data:  ## generate the main dataset and the always-fuse twin
	poetry run python scripts/01_generate_data.py --name $(DATA)
	poetry run python scripts/01_generate_data.py --name twin --head fused

train:  ## train on DATA into results/RUN
	poetry run python scripts/02_train.py --data $(DATA) --run $(RUN)

analyze:  ## compare RUN against the analytical observer
	poetry run python scripts/03_analyze.py --run $(RUN)

figures:  ## render every figure for RUN
	poetry run python scripts/04_figures.py --run $(RUN)

all:  ## the whole pipeline, including the control twin
	poetry run bash scripts/run_all.sh

quick:  ## the whole pipeline, small and fast
	poetry run bash scripts/run_all.sh quick

clean:  ## remove caches and compiled files
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

clean-runs:  ## delete every generated dataset and run (keeps the folders)
	rm -rf results/* data/*.npz
	touch results/.gitkeep data/.gitkeep
