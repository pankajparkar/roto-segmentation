# Technology Stack

## Overview

This document details all technologies, frameworks, and tools used in the AI Roto Automation system.

---

## Backend Stack

### Core Framework

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Runtime | Python | 3.11+ | Primary language |
| API Framework | FastAPI | 0.100+ | REST API server |
| ASGI Server | Uvicorn | 0.23+ | Production server |
| Task Queue | Celery | 5.3+ | Async job processing |
| Message Broker | Redis | 7.0+ | Queue backend + caching |

### Database

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Primary DB | PostgreSQL | 15+ | Relational data |
| ORM | SQLAlchemy | 2.0+ | Database abstraction |
| Migrations | Alembic | 1.12+ | Schema versioning |
| Cache | Redis | 7.0+ | Session/result caching |

### Storage

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Object Storage | MinIO / AWS S3 | Frame & mask storage |
| File System | NFS / EFS | Shared mount for workers |
| Temp Storage | Local SSD | GPU worker scratch |

---

## AI/ML Stack

### Deep Learning Framework

```
PyTorch 2.1+
├── torchvision
├── torchaudio (optional)
└── CUDA 12.1+
```

### Segmentation Models

| Model | Source | Purpose | VRAM |
|-------|--------|---------|------|
| **SAM 2** | Meta | Foundation segmentation | 8-16 GB |
| **Grounding DINO** | IDEA Research | Text-prompted detection | 4-8 GB |
| **Mobile SAM** | MIT | Lightweight alternative | 2-4 GB |

```bash
# Installation
pip install segment-anything-2
pip install groundingdino
```

### Video Object Segmentation

| Model | Source | Purpose | VRAM |
|-------|--------|---------|------|
| **Cutie** | HKUST | Modern VOS | 6-10 GB |
| **XMem** | HKUST | Video propagation | 8-12 GB |
| **DEVA** | HKUST | Tracking + segmentation | 8-12 GB |

```bash
# Installation
pip install cutie-video-segmentation
# or clone from GitHub for latest
```

### Matting Models

| Model | Source | Purpose | VRAM |
|-------|--------|---------|------|
| **ViTMatte** | Microsoft | Transformer matting | 6-10 GB |
| **Matte Anything** | Various | Combined pipeline | 8-12 GB |
| **MODNet** | ZHKKKe | Real-time matting | 2-4 GB |

### Supporting Libraries

```python
# Computer Vision
opencv-python>=4.8.0        # Image processing
scikit-image>=0.21.0        # Additional algorithms
Pillow>=10.0.0              # Image I/O

# Scientific Computing
numpy>=1.24.0               # Array operations
scipy>=1.11.0               # Bezier fitting, interpolation

# Video Processing
ffmpeg-python>=0.2.0        # Video I/O wrapper
av>=10.0.0                  # PyAV for frame extraction

# Image Formats
OpenEXR>=3.2.0              # EXR support
imageio>=2.31.0             # Multi-format I/O
```

---

## Frontend Stack

### Web Application

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Framework | React | 18+ | UI framework |
| Language | TypeScript | 5.0+ | Type safety |
| Build Tool | Vite | 5.0+ | Fast bundling |
| State | Zustand | 4.4+ | State management |
| Styling | Tailwind CSS | 3.3+ | Utility CSS |

### Canvas & Visualization

```javascript
// Interactive canvas
fabric.js >= 6.0.0          // Shape manipulation
// OR
konva >= 9.0.0              // React-friendly canvas

// Video playback
video.js >= 8.0.0           // Player framework
// Custom frame-accurate seeking implementation

// UI Components
@radix-ui/react-* >= 1.0.0  // Accessible primitives
lucide-react >= 0.290.0     // Icons
```

### Desktop Application (Optional)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | Electron 27+ | Cross-platform desktop |
| Alternative | Tauri 1.5+ | Lighter weight, Rust-based |

---

## Infrastructure

### Containerization

```yaml
# Docker
docker >= 24.0
docker-compose >= 2.20

# Container Images
python:3.11-slim            # Base API image
nvidia/cuda:12.1-runtime    # GPU worker base
node:20-alpine              # Frontend build
```

### Orchestration

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Container Orchestration | Kubernetes 1.28+ | Production deployment |
| GPU Scheduling | NVIDIA Device Plugin | GPU allocation |
| Ingress | NGINX Ingress | Load balancing |
| Secrets | External Secrets | Credentials management |

### Monitoring

```yaml
# Observability Stack
Prometheus              # Metrics collection
Grafana                 # Visualization
Loki                    # Log aggregation
Jaeger                  # Distributed tracing

# Application Monitoring
Sentry                  # Error tracking
OpenTelemetry           # Instrumentation
```

---

## GPU Infrastructure

### Recommended Hardware

| Tier | GPU | VRAM | Use Case |
|------|-----|------|----------|
| Development | RTX 4080/4090 | 16-24 GB | Local testing |
| Production | A100 40GB | 40 GB | Full pipeline |
| Production (budget) | A10 | 24 GB | Cost-effective |
| Production (high-end) | H100 | 80 GB | Maximum throughput |

### Cloud Options

| Provider | Instance | GPU | Cost/hr (approx) |
|----------|----------|-----|------------------|
| AWS | g5.xlarge | A10G 24GB | $1.00 |
| AWS | p4d.24xlarge | 8x A100 | $32.00 |
| GCP | a2-highgpu-1g | A100 40GB | $3.67 |
| Azure | NC24ads A100 | A100 80GB | $3.67 |
| Lambda Labs | 1x A100 | A100 40GB | $1.10 |
| RunPod | 1x A100 | A100 40GB | $1.44 |

### Multi-GPU Setup

```python
# Model distribution across GPUs
GPU 0: SAM 2 (segmentation)
GPU 1: Cutie/XMem (propagation)
GPU 2: ViTMatte (matting)
GPU 3: Reserve / batch processing
```

---

## Development Tools

### Code Quality

```yaml
# Python
ruff >= 0.1.0               # Fast linter
black >= 23.0.0             # Formatter
mypy >= 1.6.0               # Type checking
pytest >= 7.4.0             # Testing

# JavaScript/TypeScript
eslint >= 8.50.0            # Linting
prettier >= 3.0.0           # Formatting
vitest >= 0.34.0            # Testing
playwright >= 1.39.0        # E2E testing
```

### Documentation

```yaml
mkdocs >= 1.5.0             # Documentation site
mkdocs-material >= 9.4.0    # Theme
mkdocstrings >= 0.23.0      # API docs generation
```

### CI/CD

| Stage | Tool | Purpose |
|-------|------|---------|
| Version Control | GitHub | Repository hosting |
| CI | GitHub Actions | Build & test |
| Container Registry | GitHub Container Registry | Image storage |
| CD | ArgoCD | GitOps deployment |

---

## External Integrations

### VFX Software SDKs

```python
# Nuke
nuke >= 14.0                # Nuke Python API (when running inside Nuke)

# Silhouette
# No external SDK - generate FXS files directly

# After Effects
# CEP/UXP extensions communicate via localhost API
```

### File Format Libraries

```python
# OpenEXR
import OpenEXR
import Imath

# DPX
import imageio

# Video containers
import av  # PyAV for MOV, MP4, MXF
```

---

## Dependency Management

### Python

```toml
# pyproject.toml
[project]
name = "roto-segmentation"
version = "0.1.0"
requires-python = ">=3.11"

[tool.poetry]
# OR
[tool.pdm]
# OR
[project.dependencies]
# Using pip with requirements.txt
```

### Node.js

```json
{
  "packageManager": "pnpm@8.10.0",
  "engines": {
    "node": ">=20.0.0"
  }
}
```

---

## Version Compatibility Matrix

| Component | Min Version | Recommended | Notes |
|-----------|-------------|-------------|-------|
| Python | 3.10 | 3.11 | 3.12 has some ML lib issues |
| PyTorch | 2.0 | 2.1+ | For SAM 2 compatibility |
| CUDA | 11.8 | 12.1 | Match PyTorch build |
| Node.js | 18 | 20 LTS | For frontend build |
| PostgreSQL | 14 | 15+ | JSONB performance |
| Redis | 6.2 | 7.0+ | Streams support |

---

## Quick Start Dependencies

### Minimal Development Setup

```bash
# System dependencies (macOS)
brew install python@3.11 node@20 postgresql@15 redis ffmpeg

# System dependencies (Ubuntu)
sudo apt install python3.11 nodejs postgresql redis-server ffmpeg

# Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn celery redis sqlalchemy
pip install opencv-python numpy scipy pillow
pip install segment-anything-2

# Frontend
cd frontend
pnpm install
```

### Production Docker Images

```dockerfile
# API Server
FROM python:3.11-slim
# ... dependencies

# GPU Worker
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04
# ... PyTorch + models

# Frontend
FROM node:20-alpine AS builder
# ... build React app
FROM nginx:alpine
# ... serve static files
```
