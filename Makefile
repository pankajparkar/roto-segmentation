# Roto-Seg Makefile
# Simple task runner for development and deployment

.PHONY: help dev dev-backend dev-frontend build test lint clean install docker-up docker-down docker-build

# Default target
help:
	@echo "Roto-Seg Development Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install        Install all dependencies"
	@echo "  make dev            Start full development environment (Docker)"
	@echo "  make dev-backend    Start backend only (local Python)"
	@echo "  make dev-frontend   Start frontend only (local Node)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-backend   Run backend tests"
	@echo "  make test-frontend  Run frontend tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Lint all code"
	@echo "  make lint-backend   Lint backend code"
	@echo "  make lint-frontend  Lint frontend code"
	@echo "  make format         Format all code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-build   Build all images"
	@echo "  make docker-logs    View logs"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Clean build artifacts"
	@echo "  make db-migrate     Run database migrations"
	@echo "  make db-reset       Reset database"

# =============================================================================
# Installation
# =============================================================================

install: install-backend install-frontend
	@echo "All dependencies installed!"

install-backend:
	@echo "Installing backend dependencies..."
	cd backend && pip install -e ".[dev]"

install-frontend:
	@echo "Installing frontend dependencies..."
	cd frontend && pnpm install

# =============================================================================
# Development
# =============================================================================

dev:
	@echo "Starting development environment..."
	docker-compose up

dev-backend:
	@echo "Starting backend server..."
	cd backend && uvicorn src.roto_seg.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	@echo "Starting frontend server..."
	cd frontend && pnpm dev

dev-worker:
	@echo "Starting Celery worker..."
	cd backend && celery -A roto_seg.worker worker --loglevel=info

# =============================================================================
# Testing
# =============================================================================

test: test-backend test-frontend
	@echo "All tests passed!"

test-backend:
	@echo "Running backend tests..."
	cd backend && pytest

test-frontend:
	@echo "Running frontend tests..."
	cd frontend && pnpm test

test-coverage:
	@echo "Running tests with coverage..."
	cd backend && pytest --cov=src/roto_seg --cov-report=html
	cd frontend && pnpm test:coverage

# =============================================================================
# Code Quality
# =============================================================================

lint: lint-backend lint-frontend
	@echo "All linting passed!"

lint-backend:
	@echo "Linting backend..."
	cd backend && ruff check src tests
	cd backend && mypy src

lint-frontend:
	@echo "Linting frontend..."
	cd frontend && pnpm lint

format: format-backend format-frontend
	@echo "All code formatted!"

format-backend:
	@echo "Formatting backend..."
	cd backend && ruff check --fix src tests
	cd backend && black src tests

format-frontend:
	@echo "Formatting frontend..."
	cd frontend && pnpm lint --fix

# =============================================================================
# Docker
# =============================================================================

docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-logs:
	docker-compose logs -f

docker-clean:
	@echo "Removing Docker volumes..."
	docker-compose down -v

docker-prod:
	@echo "Starting production environment..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# =============================================================================
# Database
# =============================================================================

db-migrate:
	@echo "Running database migrations..."
	cd backend && alembic upgrade head

db-rollback:
	@echo "Rolling back last migration..."
	cd backend && alembic downgrade -1

db-reset:
	@echo "Resetting database..."
	cd backend && alembic downgrade base
	cd backend && alembic upgrade head

db-shell:
	@echo "Opening database shell..."
	docker-compose exec db psql -U postgres -d rotoseg

# =============================================================================
# Utilities
# =============================================================================

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"

# Download AI models
download-models:
	@echo "Downloading AI models..."
	mkdir -p models
	@echo "SAM 2 model download requires manual setup. See docs."
	@echo "Visit: https://github.com/facebookresearch/segment-anything-2"

# Generate API documentation
docs:
	@echo "Generating API documentation..."
	cd backend && python -m mkdocs build

# Show project structure
tree:
	@echo "Project structure:"
	@find . -type f -name "*.py" -o -name "*.ts" -o -name "*.tsx" | grep -v node_modules | grep -v __pycache__ | head -50
