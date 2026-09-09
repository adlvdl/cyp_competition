.PHONY: help setup data data-diff lab test lint baseline check-activity check-tdi

help:
	@echo "setup          uv sync (creates .venv with all deps)"
	@echo "data           download challenge CSVs into a dated data/raw/YYYYMMDD/ snapshot"
	@echo "data-diff      compare the two most recent snapshots (no download)"
	@echo "lab            launch marimo"
	@echo "baseline       run notebooks/01_baseline.py as a script"
	@echo "test           run pytest"
	@echo "lint           ruff check + format"
	@echo "check-activity FILE=path/to.csv   validate a direct-inhibition submission"
	@echo "check-tdi      FILE=path/to.csv   validate a TDI submission"

setup:
	uv sync --all-extras

data:
	uv run python -m cyp.download

data-diff:
	uv run python -m cyp.download --compare-only

lab:
	uv run marimo edit notebooks

baseline:
	uv run python notebooks/01_baseline.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format src tests

check-activity:
	uv run python -m cyp.submission $(FILE) --track activity

check-tdi:
	uv run python -m cyp.submission $(FILE) --track tdi
