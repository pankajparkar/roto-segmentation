# Apple Silicon (M1/M2/M3) Setup Guide

This guide covers setting up roto-segmentation on Apple Silicon Macs.

## Hardware Considerations

| Model | RAM | Recommended SAM2 Model | Notes |
|-------|-----|----------------------|-------|
| M1 (16GB) | 16GB | `sam2_hiera_small` | Your setup - works well |
| M1 Pro/Max | 32GB+ | `sam2_hiera_base_plus` | Good performance |
| M2/M3 Pro/Max | 32GB+ | `sam2_hiera_large` | Best quality |

With 16GB RAM, use the **small** model to leave room for video frames and other processing.

## Quick Start (Recommended)

### Option 1: Local Development (Best for M1)

Running locally gives you access to the **Metal Performance Shaders (MPS)** GPU acceleration, which is not available inside Docker containers.

```bash
# 1. Install system dependencies
brew install python@3.11 node postgresql@15 redis ffmpeg

# 2. Start services
brew services start postgresql@15
brew services start redis

# 3. Create database
createdb rotoseg

# 4. Setup Python environment
cd backend
python3.11 -m venv venv
source venv/bin/activate

# 5. Install PyTorch with MPS support
pip install torch torchvision torchaudio

# 6. Install project dependencies
pip install -e ".[dev]"

# 7. Copy and edit environment
cp .env.example .env
# Edit .env and set DEVICE=mps

# 8. Start backend
uvicorn src.roto_seg.main:app --reload

# 9. In another terminal, start frontend
cd frontend
pnpm install
pnpm dev
```

### Option 2: Docker (Simpler but No GPU)

Docker on Mac runs in a Linux VM, so MPS is not available. Use this for quick testing or if you don't need GPU acceleration.

```bash
# Start everything
make dev

# Or directly with docker-compose
docker-compose up
```

## PyTorch MPS Verification

Verify MPS is working:

```python
import torch

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Test tensor on MPS
if torch.backends.mps.is_available():
    x = torch.ones(5, device="mps")
    print(f"Tensor on MPS: {x}")
```

## Model Memory Requirements

| Model | Parameters | Approx. Memory | Recommended RAM |
|-------|-----------|----------------|-----------------|
| SAM2 Tiny | 39M | ~2 GB | 8GB+ |
| SAM2 Small | 46M | ~3 GB | 16GB (your setup) |
| SAM2 Base+ | 81M | ~5 GB | 24GB+ |
| SAM2 Large | 224M | ~10 GB | 32GB+ |

## Performance Tips for 16GB M1

### 1. Use Smaller Model

```python
# In .env
SAM2_MODEL=sam2_hiera_small.pt
```

### 2. Process Fewer Frames at Once

```python
# Reduce batch size for video processing
BATCH_SIZE=4  # Instead of default 8-16
```

### 3. Clear Memory Between Operations

```python
import torch
import gc

# After processing a shot
torch.mps.empty_cache()
gc.collect()
```

### 4. Monitor Memory Usage

```bash
# Watch memory in real-time
sudo powermetrics --samplers gpu_power -i 1000
```

### 5. Close Other Apps

For best performance, close memory-heavy apps (Chrome, Slack, etc.) when running AI inference.

## Troubleshooting

### "MPS backend out of memory"

```python
# Reduce image resolution temporarily
MAX_INFERENCE_SIZE = 1024  # Instead of original resolution

# Or switch to CPU for large images
DEVICE = "cpu"
```

### Slow First Inference

MPS compilation happens on first run. Subsequent inferences are faster.

```python
# Warm up the model
dummy_input = torch.randn(1, 3, 1024, 1024, device="mps")
with torch.no_grad():
    model(dummy_input)
```

### PyTorch MPS Bugs

MPS is relatively new. If you hit issues:

```bash
# Update PyTorch to latest
pip install --upgrade torch torchvision torchaudio

# Or fall back to CPU
export DEVICE=cpu
```

## Recommended .env for M1 16GB

```bash
# backend/.env
DEBUG=true
ENV=development

DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rotoseg
REDIS_URL=redis://localhost:6379/0

STORAGE_PATH=./storage
MODEL_PATH=./models

# Optimized for M1 16GB
SAM2_MODEL=sam2_hiera_small.pt
DEVICE=mps

# Memory-conscious settings
MAX_UPLOAD_SIZE=209715200  # 200MB limit
```

## Expected Performance

On M1 Mac with 16GB RAM using MPS:

| Operation | Time (approx) |
|-----------|--------------|
| Single image segmentation | 200-400ms |
| Video propagation (per frame) | 50-100ms |
| FXS export (100 frames) | <2s |

CPU fallback is ~3-5x slower but still usable for testing.
