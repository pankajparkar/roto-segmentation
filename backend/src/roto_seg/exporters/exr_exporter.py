"""
EXR Matte Exporter - Export segmentation masks as EXR sequences.

This matches the industry-standard workflow:
- Source footage (color EXR) + Matte (grayscale EXR) → Compositing
"""

import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

# Fallback to imageio for EXR writing
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


class EXRExporter:
    """
    Export masks as EXR sequence.

    Usage:
        exporter = EXRExporter(
            output_dir="./output/mattes",
            prefix="Final",
            start_frame=1001
        )
        exporter.export_masks(masks_by_frame)
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str = "Matte",
        start_frame: int = 1001,
        compression: str = "ZIP",  # ZIP, PIZ, RLE, NONE
    ):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.start_frame = start_frame
        self.compression = compression

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_masks(
        self,
        masks: Dict[int, np.ndarray],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """
        Export masks as EXR sequence.

        Args:
            masks: Dict of {frame_index: mask_array}
            width: Optional output width (resize if different)
            height: Optional output height

        Returns:
            Output directory path
        """
        if not masks:
            raise ValueError("No masks to export")

        for frame_idx, mask in masks.items():
            output_path = self.output_dir / f"{self.prefix}.{frame_idx:04d}.exr"
            self._write_exr(mask, output_path, width, height)

        return str(self.output_dir)

    def export_single(
        self,
        mask: np.ndarray,
        frame_idx: int,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """Export a single mask as EXR."""
        output_path = self.output_dir / f"{self.prefix}.{frame_idx:04d}.exr"
        self._write_exr(mask, output_path, width, height)
        return str(output_path)

    def _write_exr(
        self,
        mask: np.ndarray,
        output_path: Path,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """Write mask to EXR file."""
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[2] > 0 else mask.squeeze()

        # Normalize to 0-1 float
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255.0
        elif mask.dtype == bool:
            mask = mask.astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        # Resize if needed
        if width and height and (mask.shape[1] != width or mask.shape[0] != height):
            import cv2
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

        h, w = mask.shape

        if HAS_OPENEXR:
            self._write_exr_openexr(mask, output_path, w, h)
        elif HAS_IMAGEIO:
            self._write_exr_imageio(mask, output_path)
        else:
            raise RuntimeError("No EXR library available. Install OpenEXR or imageio.")

    def _write_exr_openexr(self, mask: np.ndarray, output_path: Path, w: int, h: int):
        """Write EXR using OpenEXR library."""
        # Convert to half float (float16) for efficiency
        half = Imath.PixelType(Imath.PixelType.HALF)

        # Create header
        header = OpenEXR.Header(w, h)
        header['compression'] = Imath.Compression(
            getattr(Imath.Compression, f'{self.compression}_COMPRESSION',
                    Imath.Compression.ZIP_COMPRESSION)
        )

        # For matte, we typically store in R, G, B, A or just Y (luminance)
        # Using RGB with same value for compatibility
        mask_bytes = mask.astype(np.float16).tobytes()

        header['channels'] = {
            'R': Imath.Channel(half),
            'G': Imath.Channel(half),
            'B': Imath.Channel(half),
            'A': Imath.Channel(half),
        }

        # Write file
        out = OpenEXR.OutputFile(str(output_path), header)
        out.writePixels({
            'R': mask_bytes,
            'G': mask_bytes,
            'B': mask_bytes,
            'A': mask_bytes,
        })
        out.close()

    def _write_exr_imageio(self, mask: np.ndarray, output_path: Path):
        """Write EXR using imageio (simpler but less control)."""
        # Stack to RGBA
        rgba = np.stack([mask, mask, mask, mask], axis=-1)
        iio.imwrite(str(output_path), rgba.astype(np.float32))


def export_masks_as_exr(
    masks: Dict[int, np.ndarray],
    output_dir: str,
    prefix: str = "Matte",
    start_frame: int = 1001,
) -> str:
    """
    Convenience function to export masks as EXR sequence.

    Args:
        masks: Dict of {frame_index: mask_array}
        output_dir: Directory for output files
        prefix: Filename prefix (e.g., "Final" → Final.1001.exr)
        start_frame: Starting frame number

    Returns:
        Output directory path
    """
    exporter = EXRExporter(
        output_dir=output_dir,
        prefix=prefix,
        start_frame=start_frame,
    )
    return exporter.export_masks(masks)
