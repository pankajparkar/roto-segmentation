"""AI models and inference."""

from roto_seg.ai.segmentation import SegmentationService
from roto_seg.ai.device import get_device, DeviceType

__all__ = ["SegmentationService", "get_device", "DeviceType"]
