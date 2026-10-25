# Financial Analysis Agent Crew — developer shortcuts
.PHONY: help install dev test lint format run docker-build docker-run clean

PY ?= python3
VENV ?= .venv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create the virtualenv and install runtime dependencies
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

dev: install  ## Install runtime + development dependencies
	$(VENV)/bin/pip install -r requirements-dev.txt

run:  ## Start the Streamlit app on http://localhost:8501
	$(VENV)/bin/streamlit run app.py

test:  ## Run the test-suite (offline, no API key required)
	$(VENV)/bin/python -m pytest tests -q

lint:  ## Lint with ruff
	$(VENV)/bin/ruff check .

format:  ## Format with ruff
	$(VENV)/bin/ruff format .

demo:  ## Generate an offline demo report from the CLI
	FORCE_DEMO_DATA=1 $(VENV)/bin/python scripts/daily_report.py --tickers AAPL --days 180 --no-llm

docker-build:  ## Build the Docker image
	docker build -t financial-crew .

docker-run:  ## Run the container with .env and persisted outputs
	docker compose up --build

clean:  ## Remove generated reports, caches and build artefacts
	rm -rf reports/output/charts/* reports/output/pdfs/* reports/output/*.md \
	       reports/output/*.json logs/*.log .pytest_cache .ruff_cache __pycache__ \
	       */__pycache__ */*/__pycache__
