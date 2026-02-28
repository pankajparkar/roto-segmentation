# Roto-Seg Backend

FastAPI backend for the AI rotoscoping automation system.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env

# Run the server
uvicorn src.roto_seg.main:app --reload
```

## Project Structure

```
backend/
├── src/roto_seg/
│   ├── api/           # API endpoints
│   ├── core/          # Configuration
│   ├── models/        # Database models
│   ├── services/      # Business logic
│   └── main.py        # Application entry
├── tests/             # Test files
├── pyproject.toml     # Dependencies
└── .env.example       # Environment template
```

## Running Tests

```bash
pytest
pytest --cov=src/roto_seg  # With coverage
```

## API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
