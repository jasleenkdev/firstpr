DATA ?= ml1m
MODEL ?= popularity
ARGS ?=

.PHONY: data test lint tune run leaderboard

data:
	uv run python scripts/prepare_data.py --config configs/data/$(DATA).yaml $(ARGS)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

tune:
	uv run python scripts/run_experiment.py --model $(MODEL) --data $(DATA) --mode val $(ARGS)

run:
	uv run python scripts/run_experiment.py --model $(MODEL) --data $(DATA) --mode test $(ARGS)

leaderboard:
	uv run python scripts/leaderboard.py
