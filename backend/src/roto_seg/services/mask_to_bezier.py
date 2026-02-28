"""
Mask to Bezier Curve Conversion.

Converts binary segmentation masks to smooth Bezier curves
suitable for export to Silhouette, Nuke, and other VFX tools.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2


@dataclass
class BezierPoint:
    """
    A single Bezier control point with tangent handles.

    Attributes:
        center: (x, y) position of the control point
        right_tangent: (dx, dy) offset for outgoing handle (relative to center)
        left_tangent: (dx, dy) offset for incoming handle (relative to center)
    """
    center: Tuple[float, float]
    right_tangent: Tuple[float, float]
    left_tangent: Tuple[float, float]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "center": list(self.center),
            "right_tangent": list(self.right_tangent),
            "left_tangent": list(self.left_tangent),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BezierPoint":
        """Create from dictionary."""
        return cls(
            center=tuple(data["center"]),
            right_tangent=tuple(data["right_tangent"]),
            left_tangent=tuple(data["left_tangent"]),
        )

    def to_fxs_string(self) -> str:
        """Format point for Silhouette FXS XML."""
        cx, cy = self.center
        rx, ry = self.right_tangent
        lx, ly = self.left_tangent
        return f"({cx:.6f},{cy:.6f}),({rx:.6f},{ry:.6f}),({lx:.6f},{ly:.6f})"


@dataclass
class BezierShape:
    """
    A complete Bezier shape (closed or open path).

    Attributes:
        points: List of BezierPoint objects
        closed: Whether the shape is closed
        label: Optional label/name for the shape
    """
    points: List[BezierPoint]
    closed: bool = True
    label: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "points": [p.to_dict() for p in self.points],
            "closed": self.closed,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BezierShape":
        """Create from dictionary."""
        return cls(
            points=[BezierPoint.from_dict(p) for p in data["points"]],
            closed=data.get("closed", True),
            label=data.get("label"),
        )


class MaskToBezierConverter:
    """
    Converts binary masks to Bezier curves.

    Features:
    - Contour extraction with configurable simplification
    - Automatic tangent calculation for smooth curves
    - Point count optimization for artist-friendly shapes
    - Support for multiple contours (holes, separate objects)
    """

    def __init__(
        self,
        target_points: int = 40,
        smoothing: float = 0.5,
        min_area: int = 100,
        tangent_scale: float = 0.33,
    ):
        """
        Initialize converter.

        Args:
            target_points: Target number of points per shape
            smoothing: Contour simplification factor (0-1, higher = smoother)
            min_area: Minimum contour area to include (filters noise)
            tangent_scale: Tangent handle length as fraction of segment length
        """
        self.target_points = target_points
        self.smoothing = smoothing
        self.min_area = min_area
        self.tangent_scale = tangent_scale

    def convert(
        self,
        mask: np.ndarray,
        label: Optional[str] = None,
    ) -> List[BezierShape]:
        """
        Convert binary mask to Bezier shapes.

        Args:
            mask: Binary mask (H, W) with values 0 or 255 (or bool)
            label: Optional label for shapes

        Returns:
            List of BezierShape objects (one per contour)
        """
        # Ensure correct format
        if mask.dtype == bool:
            mask = (mask * 255).astype(np.uint8)
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        # Find contours
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,  # Only outer contours
            cv2.CHAIN_APPROX_NONE,  # All points
        )

        shapes = []
        for i, contour in enumerate(contours):
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # Convert contour to Bezier shape
            shape = self._contour_to_bezier(contour, label or f"shape_{i}")
            if shape is not None:
                shapes.append(shape)

        return shapes

    def _contour_to_bezier(
        self,
        contour: np.ndarray,
        label: str,
    ) -> Optional[BezierShape]:
        """Convert single contour to Bezier shape."""
        # Squeeze to (N, 2)
        contour = contour.squeeze()
        if len(contour) < 3:
            return None

        # Simplify contour
        simplified = self._simplify_contour(contour)
        if len(simplified) < 3:
            return None

        # Resample to target point count
        resampled = self._resample_contour(simplified, self.target_points)

        # Calculate Bezier points with tangents
        bezier_points = self._calculate_bezier_points(resampled)

        return BezierShape(
            points=bezier_points,
            closed=True,
            label=label,
        )

    def _simplify_contour(self, contour: np.ndarray) -> np.ndarray:
        """Simplify contour using Douglas-Peucker algorithm."""
        # Ensure contour is in correct shape for OpenCV
        contour_cv = contour.reshape(-1, 1, 2).astype(np.float32)

        # Calculate epsilon based on perimeter and smoothing factor
        perimeter = cv2.arcLength(contour_cv, True)
        epsilon = self.smoothing * perimeter / 100

        # Simplify
        simplified = cv2.approxPolyDP(contour_cv, epsilon, True)

        return simplified.squeeze()

    def _resample_contour(
        self,
        contour: np.ndarray,
        target_points: int,
    ) -> np.ndarray:
        """Resample contour to have target number of points."""
        n = len(contour)

        if n <= target_points:
            return contour

        # Select evenly spaced indices
        indices = np.linspace(0, n - 1, target_points, dtype=int)
        return contour[indices]

    def _calculate_bezier_points(
        self,
        contour: np.ndarray,
    ) -> List[BezierPoint]:
        """
        Calculate Bezier points with smooth tangent handles.

        Tangent direction is based on the line between neighboring points.
        Tangent length is proportional to distance to neighbors.
        """
        n = len(contour)
        bezier_points = []

        for i in range(n):
            # Current point
            curr = contour[i]

            # Previous and next points (wrap around for closed shape)
            prev_pt = contour[(i - 1) % n]
            next_pt = contour[(i + 1) % n]

            # Calculate tangent direction (average of incoming and outgoing vectors)
            tangent = next_pt - prev_pt
            tangent_len = np.linalg.norm(tangent)

            if tangent_len > 0:
                tangent = tangent / tangent_len
            else:
                tangent = np.array([1.0, 0.0])

            # Calculate handle lengths based on distance to neighbors
            dist_prev = np.linalg.norm(curr - prev_pt)
            dist_next = np.linalg.norm(curr - next_pt)

            # Tangent handles (scaled by distance)
            right_handle = tangent * dist_next * self.tangent_scale
            left_handle = -tangent * dist_prev * self.tangent_scale

            # Detect corners (sharp angles) and reduce tangent length
            if n > 2:
                vec_in = curr - prev_pt
                vec_out = next_pt - curr

                len_in = np.linalg.norm(vec_in)
                len_out = np.linalg.norm(vec_out)

                if len_in > 0 and len_out > 0:
                    cos_angle = np.dot(vec_in, vec_out) / (len_in * len_out)
                    cos_angle = np.clip(cos_angle, -1, 1)

                    # If angle is sharp (< 90 degrees), reduce tangent
                    if cos_angle < 0:
                        corner_factor = (1 + cos_angle) / 2  # 0 at 180°, 0.5 at 90°
                        right_handle = right_handle * corner_factor
                        left_handle = left_handle * corner_factor

            bezier_points.append(BezierPoint(
                center=(float(curr[0]), float(curr[1])),
                right_tangent=(float(right_handle[0]), float(right_handle[1])),
                left_tangent=(float(left_handle[0]), float(left_handle[1])),
            ))

        return bezier_points


def mask_to_shapes(
    mask: np.ndarray,
    target_points: int = 40,
    smoothing: float = 0.5,
    label: Optional[str] = None,
) -> List[BezierShape]:
    """
    Convenience function to convert mask to Bezier shapes.

    Args:
        mask: Binary mask (H, W)
        target_points: Target number of points per shape
        smoothing: Simplification factor (0-1)
        label: Optional label for shapes

    Returns:
        List of BezierShape objects
    """
    converter = MaskToBezierConverter(
        target_points=target_points,
        smoothing=smoothing,
    )
    return converter.convert(mask, label)


def optimize_keyframes(
    shapes_by_frame: Dict[int, List[BezierShape]],
    tolerance: float = 1.0,
) -> Dict[int, List[BezierShape]]:
    """
    Remove redundant keyframes where shapes haven't changed significantly.

    This reduces file size and makes artist cleanup easier.

    Args:
        shapes_by_frame: Dictionary mapping frame numbers to shape lists
        tolerance: Maximum pixel deviation to consider "same"

    Returns:
        Optimized dictionary with redundant frames removed
    """
    frames = sorted(shapes_by_frame.keys())
    if len(frames) <= 2:
        return shapes_by_frame

    optimized = {frames[0]: shapes_by_frame[frames[0]]}

    for i in range(1, len(frames)):
        curr_frame = frames[i]
        prev_frame = frames[i - 1]

        curr_shapes = shapes_by_frame[curr_frame]
        prev_shapes = shapes_by_frame[prev_frame]

        # Check if shapes are significantly different
        if len(curr_shapes) != len(prev_shapes):
            optimized[curr_frame] = curr_shapes
            continue

        is_different = False
        for cs, ps in zip(curr_shapes, prev_shapes):
            if len(cs.points) != len(ps.points):
                is_different = True
                break

            for cp, pp in zip(cs.points, ps.points):
                deviation = np.sqrt(
                    (cp.center[0] - pp.center[0])**2 +
                    (cp.center[1] - pp.center[1])**2
                )
                if deviation > tolerance:
                    is_different = True
                    break

            if is_different:
                break

        if is_different:
            optimized[curr_frame] = curr_shapes

    # Always include last frame
    optimized[frames[-1]] = shapes_by_frame[frames[-1]]

    return optimized
