.PHONY: help setup data data-diff external-data external-diff lab test lint baseline check-activity check-tdi watch-contention

help:
	@echo "setup             uv sync (creates .venv with all deps)"
	@echo "data              download challenge CSVs into a dated data/raw/YYYYMMDD/ snapshot"
	@echo "data-diff         compare the two most recent snapshots (no download)"
	@echo "external-data     download the PubChem qHTS panel into data/external/YYYYMMDD/"
	@echo "external-diff     compare the two most recent external snapshots (no download)"
	@echo "lab               launch marimo"
	@echo "baseline          run notebooks/01_baseline.py as a script"
	@echo "test              run pytest"
	@echo "lint              ruff check + format"
	@echo "check-activity FILE=path/to.csv   validate a direct-inhibition submission"
	@echo "check-tdi      FILE=path/to.csv   validate a TDI submission"
	@echo "watch-contention TIMING=path/to/timings.csv   live-monitor a long CV run for"
	@echo "                  machine contention (Spotlight/thermal), snapshotting system"
	@echo "                  state the moment a fold looks anomalous. Run this alongside"
	@echo "                  a notebook, not instead of it -- e.g. in a second terminal"
	@echo "                  while notebooks/07_placement.py's CV cells are executing."

setup:
	uv sync --all-extras

data:
	uv run python -m cyp.download

data-diff:
	uv run python -m cyp.download --compare-only

external-data:
	uv run python -m cyp.external

external-diff:
	uv run python -m cyp.external --compare-only

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

watch-contention:
	@test -n "$(TIMING)" || (echo "usage: make watch-contention TIMING=experiments/<notebook>/timings.csv"; exit 1)
	bash scripts/watch_contention.sh $(TIMING)
