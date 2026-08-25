.PHONY: install db-up db-down dashboard api mlflow lint test clean

install:
	pip install -e ".[dev]"

db-up:
	docker compose up -d postgres pgadmin

db-down:
	docker compose down

dashboard:
	streamlit run dashboard/app.py

api:
	uvicorn src.api.main:app --reload --port 8000

mlflow:
	mlflow server --port 5000

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
