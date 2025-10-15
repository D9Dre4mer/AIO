.PHONY: help up down train serve test clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## Start development stack
	docker-compose -f infra/docker-compose.dev.yml up -d --build

down: ## Stop development stack
	docker-compose -f infra/docker-compose.dev.yml down

train: ## Run training pipeline with MLflow logging
	python -m src.train.main

ui: ## Start Streamlit UI
	python -m streamlit run app.py --server.port 8501

serve: ## Start FastAPI server
	uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

test: ## Run tests
	python -m pytest tests/ -v

clean: ## Clean up generated files
	if exist data\processed rmdir /s /q data\processed
	if exist data\features rmdir /s /q data\features
	if exist models rmdir /s /q models
	if exist artifacts rmdir /s /q artifacts
	if exist metrics rmdir /s /q metrics
	if exist cache rmdir /s /q cache
	docker-compose -f infra/docker-compose.dev.yml down -v

dvc-repro: ## Run DVC pipeline
	dvc repro

dvc-push: ## Push data to DVC remote
	dvc push

dvc-pull: ## Pull data from DVC remote
	dvc pull

mlflow-ui: ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

conda-lock: ## Generate conda lock file
	conda-lock -f environment.yml -p linux-64 -p win-64 -p osx-64
