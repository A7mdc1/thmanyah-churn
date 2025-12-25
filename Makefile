install:
	pip install uv && uv sync

lint:
	uv run ruff check src && uv run black --check src

format:
	uv run ruff check --fix src && uv run black src

test:
	uv run pytest tests/ -v

run:
	uv run uvicorn src.api:app --reload --port 8000

train:
	uv run python -m src.model

monitor:
	uv run python -m src.monitor_job

docker-build:
	docker build -t thmanyah-churn .

docker-run:
	docker run -p 8000:8000 thmanyah-churn