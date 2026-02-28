# Roto-Seg

AI-powered rotoscoping automation system that dramatically reduces manual labor while maintaining broadcast/theatrical quality output.

## Overview

Roto-Seg uses state-of-the-art AI models (SAM 2, Cutie, ViTMatte) to automate video object segmentation and exports results to industry-standard formats (Silhouette, Nuke, After Effects).

**Key Features:**
- Single-click object segmentation using SAM 2
- Automatic mask propagation across video frames
- Export to Silhouette (.fxs), Nuke (.nk), and image sequences
- Artist-friendly Bezier splines (fully editable)
- Batch processing with queue management

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- Redis 7+
- Docker (optional)

### Platform Support

| Platform | GPU Acceleration | Recommended |
|----------|-----------------|-------------|
| **macOS (Apple Silicon)** | MPS (Metal) | Local dev |
| **Linux (NVIDIA)** | CUDA | Production |
| **Windows (NVIDIA)** | CUDA | Local dev |

### Apple Silicon (M1/M2/M3) - Recommended Setup

For M1 Mac with 16GB RAM, run locally to use Metal GPU acceleration:

```bash
# 1. Install dependencies
brew install python@3.11 node postgresql@15 redis ffmpeg

# 2. Start services
brew services start postgresql@15
brew services start redis
createdb rotoseg

# 3. Setup backend
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio  # MPS-enabled PyTorch
pip install -e ".[dev]"
cp .env.example .env  # Edit: DEVICE=mps

# 4. Start backend
uvicorn src.roto_seg.main:app --reload

# 5. In another terminal, start frontend
cd frontend
pnpm install && pnpm dev
```

See [docs/M1-SETUP.md](./docs/M1-SETUP.md) for detailed M1/M2/M3 setup.

### Using Docker

```bash
# Start all services (uses CPU for AI - no MPS in Docker)
make dev

# Or directly
docker-compose up
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Project Structure

```
roto-segmentation/
├── backend/                 # Python FastAPI backend
│   ├── src/roto_seg/       # Application code
│   │   ├── api/            # REST API endpoints
│   │   ├── core/           # Configuration
│   │   ├── models/         # Database models
│   │   └── services/       # Business logic
│   └── tests/              # Backend tests
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── pages/          # Page components
│   │   ├── hooks/          # Custom hooks
│   │   ├── lib/            # Utilities
│   │   └── store/          # State management
│   └── tests/              # Frontend tests
├── plugins/                # VFX software plugins
│   ├── nuke/              # Nuke Python plugin
│   └── silhouette/        # Silhouette integration
├── docker/                 # Docker configurations
├── plans/                  # Planning documentation
├── docs/                   # Additional documentation
│   └── M1-SETUP.md        # Apple Silicon setup guide
├── docker-compose.yml      # Development environment
└── Makefile               # Task automation
```

## Available Commands

```bash
# Development
make dev              # Start full dev environment (Docker)
make dev-backend      # Start backend only (local Python)
make dev-frontend     # Start frontend only (local Node)

# Testing
make test             # Run all tests
make test-backend     # Run backend tests
make test-frontend    # Run frontend tests

# Code Quality
make lint             # Lint all code
make format           # Format all code

# Docker
make docker-up        # Start Docker services
make docker-down      # Stop Docker services
make docker-build     # Build Docker images

# Database
make db-migrate       # Run migrations
make db-reset         # Reset database
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/projects` | Create a project |
| `GET` | `/api/v1/projects` | List all projects |
| `POST` | `/api/v1/shots/upload` | Upload a shot |
| `POST` | `/api/v1/shots/{id}/segment` | Start segmentation |
| `GET` | `/api/v1/shots/{id}/masks` | Get mask data |
| `POST` | `/api/v1/exports` | Export to format |

See full API documentation at http://localhost:8000/docs

## Technology Stack

**Backend:**
- Python 3.11+
- FastAPI
- Celery + Redis
- PostgreSQL
- PyTorch + SAM 2

**Frontend:**
- React 18
- TypeScript
- Vite
- Tailwind CSS
- TanStack Query

**AI/ML:**
- SAM 2 (segmentation)
- MPS (Apple Silicon) / CUDA (NVIDIA)

## Documentation

- [M1/M2/M3 Setup Guide](./docs/M1-SETUP.md)
- [Executive Summary](./plans/01-executive-summary.md)
- [Technical Architecture](./plans/02-technical-architecture.md)
- [Silhouette Integration](./plans/03-silhouette-integration.md)
- [Technology Stack](./plans/04-technology-stack.md)
- [Implementation Roadmap](./plans/05-implementation-roadmap.md)
- [AI Models Guide](./plans/06-ai-models-guide.md)

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
