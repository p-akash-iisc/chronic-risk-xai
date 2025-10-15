SHELL := /bin/bash
PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip

CONFIG ?= configs/default.yaml

.PHONY: setup data train evaluate explain dashboard api test fmt lint clean

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PYTHON) -m src.cli.prepare_data --config $(CONFIG)

train:
	$(PYTHON) -m src.cli.run_experiment --config $(CONFIG)

evaluate:
	$(PYTHON) -m src.chronic_risk.evaluate --config $(CONFIG)

explain:
	$(PYTHON) -m src.cli.run_explain --config $(CONFIG)

dashboard:
	$(PYTHON) -m src.cli.launch_dashboard --config $(CONFIG)

api:
	$(PYTHON) -m src.cli.serve_api --config $(CONFIG)

test:
	./.venv/bin/pytest --cov=src --cov-report=term-missing

fmt:
	./.venv/bin/black .
	./.venv/bin/ruff check . --fix

lint:
	./.venv/bin/ruff check .
	./.venv/bin/black --check .

clean:
	rm -rf .venv mlruns models/artifacts/* models/reports/* data/processed/*

