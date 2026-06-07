.PHONY: setup lint test generate-data train analyze visualize format

setup:
	poetry install

lint:
	poetry run ruff check src tests scripts
	poetry run black --check src tests scripts
	poetry run mypy src

format:
	poetry run ruff check --fix src tests scripts
	poetry run black src tests scripts

test:
	poetry run pytest

generate-data:
	poetry run generate-data --config configs/default.yaml --out data/dataset.npz

train:
	poetry run train --config configs/default.yaml --head-type causal --dataset-path data/dataset.npz

analyze:
	poetry run run-analysis integration --checkpoint checkpoints/model_seed0.pth --config configs/default.yaml --dataset data/dataset.npz

visualize:
	poetry run python scripts/visualize.py --config configs/default.yaml --results-dir results
