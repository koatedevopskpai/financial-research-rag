.PHONY: dev test lint typecheck clean build docker docker-run

# ── Development ──────────────────────────────────────────────
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-level info

test:
	python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

lint:
	ruff check app/ tests/

lint-fix:
	ruff check --fix app/ tests/

typecheck:
	mypy app/ tests/

check: lint typecheck test

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# ── Docker ───────────────────────────────────────────────────
build:
	docker build -t financial-research-rag -f infrastructure/docker/Dockerfile .

docker-run:
	docker compose -f infrastructure/docker/docker-compose.yml up --build

docker-down:
	docker compose -f infrastructure/docker/docker-compose.yml down

# ── Infrastructure ───────────────────────────────────────────
terraform-init:
	cd infrastructure/terraform && terraform init

terraform-plan:
	cd infrastructure/terraform && terraform plan

terraform-apply:
	cd infrastructure/terraform && terraform apply

terraform-destroy:
	cd infrastructure/terraform && terraform destroy
