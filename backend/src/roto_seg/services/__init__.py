"""Business logic services."""

from roto_seg.services.mask_to_bezier import (
    MaskToBezierConverter,
    BezierPoint,
    BezierShape,
    mask_to_shapes,
    optimize_keyframes,
)
from roto_seg.services.video import (
    VideoReader,
    ImageSequenceReader,
    VideoInfo,
    get_reader,
)

# Lazy import to avoid circular dependency
def get_roto_pipeline():
    from roto_seg.services.roto_pipeline import RotoPipeline
    return RotoPipeline

__all__ = [
    "MaskToBezierConverter",
    "BezierPoint",
    "BezierShape",
    "mask_to_shapes",
    "optimize_keyframes",
    "VideoReader",
    "ImageSequenceReader",
    "VideoInfo",
    "get_reader",
    "get_roto_pipeline",
]
