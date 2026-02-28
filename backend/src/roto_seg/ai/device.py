"""
Device detection and management for ML inference.
Supports MPS (Apple Silicon), CUDA (NVIDIA), and CPU.
"""

import platform
from enum import Enum
from typing import Optional

import torch


class DeviceType(str, Enum):
    """Supported device types."""
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Get the best available device for ML inference.

    Args:
        preferred: Preferred device ("mps", "cuda", "cpu").
                   If None or unavailable, auto-detects best option.

    Returns:
        torch.device: The selected device.
    """
    if preferred:
        preferred = preferred.lower()

        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred == "cpu":
            return torch.device("cpu")

    # Auto-detect best available
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        dict: Device availability and specs.
    """
    info = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "devices": {
            "mps": {
                "available": torch.backends.mps.is_available(),
                "built": torch.backends.mps.is_built(),
            },
            "cuda": {
                "available": torch.cuda.is_available(),
            },
            "cpu": {
                "available": True,
            },
        },
        "recommended": str(get_device()),
    }

    # Add CUDA details if available
    if torch.cuda.is_available():
        info["devices"]["cuda"]["device_count"] = torch.cuda.device_count()
        info["devices"]["cuda"]["device_name"] = torch.cuda.get_device_name(0)
        info["devices"]["cuda"]["memory_total"] = torch.cuda.get_device_properties(0).total_memory

    return info


def clear_memory(device: torch.device):
    """Clear GPU memory cache."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
