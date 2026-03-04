"""
Mask matting refinement utilities.

This module provides a production-friendly refinement stage that turns
hard binary masks into soft alpha mattes. It uses trimap generation and
distance-based alpha estimation, with optional temporal smoothing.
"""

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class MattingOptions:
    """Matting options used by the roto pipeline."""

    enabled: bool = False
    model: str = "vitmatte"
    temporal_smooth: float = 0.0  # 0.0 = off, 1.0 = strongest smoothing
    erode_size: int = 5
    dilate_size: int = 9


class MattingRefiner:
    """
    Refine binary masks into soft alpha mattes.

    Notes:
    - `model="vitmatte"` is accepted as an API contract.
    - Current implementation uses trimap + distance transform refinement.
      This keeps runtime lightweight and deterministic on local setups.
    """

    def __init__(self, options: MattingOptions):
        self.options = options

    def refine_sequence(self, masks_by_frame: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Refine a frame-indexed mask sequence into alpha mattes."""
        refined: Dict[int, np.ndarray] = {}
        for frame_idx in sorted(masks_by_frame.keys()):
            refined[frame_idx] = self.refine_mask(masks_by_frame[frame_idx])

        if self.options.temporal_smooth > 0:
            refined = self._temporal_smooth(refined, self.options.temporal_smooth)
        return refined

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert a binary-like mask into a soft alpha matte [0..1]."""
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        binary = (mask > 0).astype(np.uint8)
        if binary.max() == 0:
            return binary.astype(np.float32)

        trimap = self._build_trimap(binary)
        alpha = self._trimap_to_alpha(trimap)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def _build_trimap(self, binary_mask: np.ndarray) -> np.ndarray:
        """Generate trimap with values {0, 0.5, 1}."""
        erode_k = max(1, self.options.erode_size)
        dilate_k = max(erode_k + 2, self.options.dilate_size)

        k_erode = np.ones((erode_k, erode_k), np.uint8)
        k_dilate = np.ones((dilate_k, dilate_k), np.uint8)

        sure_fg = cv2.erode(binary_mask, k_erode, iterations=1)
        sure_bg = 1 - cv2.dilate(binary_mask, k_dilate, iterations=1)

        trimap = np.full(binary_mask.shape, 0.5, dtype=np.float32)
        trimap[sure_bg == 1] = 0.0
        trimap[sure_fg == 1] = 1.0
        return trimap

    def _trimap_to_alpha(self, trimap: np.ndarray) -> np.ndarray:
        """
        Build soft alpha from trimap using distance fields.

        This gives smoother edges than binary masks and is stable for EXR matte export.
        """
        fg = (trimap == 1.0).astype(np.uint8)
        bg = (trimap == 0.0).astype(np.uint8)
        unknown = (trimap == 0.5)

        # Distances to known foreground and background.
        dist_to_fg = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
        dist_to_bg = cv2.distanceTransform(1 - bg, cv2.DIST_L2, 5)

        denom = dist_to_fg + dist_to_bg + 1e-6
        alpha_unknown = dist_to_bg / denom

        alpha = np.zeros(trimap.shape, dtype=np.float32)
        alpha[fg == 1] = 1.0
        alpha[bg == 1] = 0.0
        alpha[unknown] = alpha_unknown[unknown]
        return alpha

    def _temporal_smooth(
        self,
        alpha_by_frame: Dict[int, np.ndarray],
        smooth: float,
    ) -> Dict[int, np.ndarray]:
        """Apply exponential moving average across frames."""
        smooth = float(np.clip(smooth, 0.0, 1.0))
        if smooth <= 0:
            return alpha_by_frame

        keep_current = 1.0 - smooth
        smoothed: Dict[int, np.ndarray] = {}
        prev_alpha = None

        for frame_idx in sorted(alpha_by_frame.keys()):
            curr = alpha_by_frame[frame_idx]
            if prev_alpha is None:
                out = curr
            else:
                out = keep_current * curr + smooth * prev_alpha
            smoothed[frame_idx] = out.astype(np.float32)
            prev_alpha = out

        return smoothed
