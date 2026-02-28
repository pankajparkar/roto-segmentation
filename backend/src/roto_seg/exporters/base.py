"""Base exporter interface."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from roto_seg.services.mask_to_bezier import BezierShape


class ExportFormat(str, Enum):
    """Supported export formats."""
    SILHOUETTE = "silhouette"  # .fxs
    NUKE = "nuke"  # .nk
    PNG = "png"  # PNG sequence
    EXR = "exr"  # EXR sequence
    JSON = "json"  # Raw shape data


class BaseExporter(ABC):
    """Base class for all exporters."""

    format: ExportFormat

    def __init__(
        self,
        width: int,
        height: int,
        start_frame: int = 1001,
        end_frame: int = 1100,
        fps: float = 24.0,
    ):
        """
        Initialize exporter.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            start_frame: First frame number
            end_frame: Last frame number
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.fps = fps

    @abstractmethod
    def export(
        self,
        shapes_by_frame: Dict[int, List[BezierShape]],
        output_path: str,
        layer_name: Optional[str] = None,
    ) -> str:
        """
        Export shapes to file.

        Args:
            shapes_by_frame: Dictionary mapping frame numbers to shape lists
            output_path: Path for output file
            layer_name: Optional name for the layer

        Returns:
            Path to the created file
        """
        pass

    def validate_output_path(self, output_path: str) -> Path:
        """Validate and create output directory if needed."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
