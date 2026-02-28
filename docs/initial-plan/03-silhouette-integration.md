# Silhouette Integration Specification

## Overview

This document details the technical implementation for exporting AI-generated rotoscoping data to SilhouetteFX's native `.fxs` format, enabling seamless artist workflow integration.

---

## FXS File Format Specification

### Root Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Silhouette
    width="1920"
    height="1080"
    workRangeStart="1001"
    workRangeEnd="1100"
    sessionStartFrame="1001">

    <!-- Layers contain shapes -->
    <Layer>...</Layer>

</Silhouette>
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `width` | int | Frame width in pixels |
| `height` | int | Frame height in pixels |
| `workRangeStart` | int | First frame number |
| `workRangeEnd` | int | Last frame number |
| `sessionStartFrame` | int | Session start (usually same as workRangeStart) |

---

## Layer Structure

```xml
<Layer type="Layer" label="AI_Roto_Layer" expanded="True">
    <Properties>
        <!-- Layer color for UI -->
        <Property constant="True" id="color">
            <Value>(1.000000, 0.500000, 0.000000)</Value>
        </Property>

        <!-- Invert layer output -->
        <Property constant="True" id="invert">
            <Value>false</Value>
        </Property>

        <!-- Blend mode: Add, Subtract, Difference, Max, Inside -->
        <Property constant="True" id="mode">
            <Value>Add</Value>
        </Property>

        <!-- Layer transform (4x4 matrix per frame) -->
        <Property id="transform.matrix">
            <Key frame="1001" interp="linear">
                (1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)
            </Key>
        </Property>

        <!-- Child objects (shapes and sub-layers) -->
        <Property id="objects" constant="True" expanded="True">
            <Object type="Shape">...</Object>
            <Object type="Shape">...</Object>
        </Property>
    </Properties>
</Layer>
```

---

## Shape Structure

### Complete Shape Example

```xml
<Object type="Shape" label="person_body" shape_type="Bezier"
        hidden="False" locked="False">
    <Properties>
        <!-- Opacity: 0-100, can be animated -->
        <Property id="opacity">
            <Value>100</Value>
        </Property>

        <!-- Or animated opacity -->
        <Property id="opacity">
            <Key frame="1001" interp="linear">100</Key>
            <Key frame="1050" interp="linear">50</Key>
            <Key frame="1100" interp="linear">100</Key>
        </Property>

        <!-- Motion blur toggle -->
        <Property constant="True" id="motionBlur">
            <Value>true</Value>
        </Property>

        <!-- Shape outline color in UI -->
        <Property constant="True" id="outlineColor">
            <Value>(1.0, 0.0, 0.0)</Value>
        </Property>

        <!-- Blend mode -->
        <Property constant="True" id="mode">
            <Value>Add</Value>
        </Property>

        <!-- Invert shape -->
        <Property constant="True" id="invert">
            <Value>false</Value>
        </Property>

        <!-- PATH DATA - The actual shape points -->
        <Property id="path">
            <!-- Keyframe for frame 1001 -->
            <Key frame="1001" interp="linear">
                <Path closed="true" type="Bezier">
                    <!-- Each point: (cx,cy),(rt_x,rt_y),(lt_x,lt_y) -->
                    <Point>(100.000,200.000),(5.000,0.000),(-5.000,0.000)</Point>
                    <Point>(150.000,180.000),(3.000,2.000),(-3.000,-2.000)</Point>
                    <Point>(200.000,220.000),(0.000,5.000),(0.000,-5.000)</Point>
                </Path>
            </Key>
            <!-- Keyframe for frame 1002 -->
            <Key frame="1002" interp="linear">
                <Path closed="true" type="Bezier">
                    <Point>(102.000,201.000),(5.000,0.000),(-5.000,0.000)</Point>
                    <Point>(152.000,181.000),(3.000,2.000),(-3.000,-2.000)</Point>
                    <Point>(202.000,221.000),(0.000,5.000),(0.000,-5.000)</Point>
                </Path>
            </Key>
        </Property>
    </Properties>
</Object>
```

### Point Format

#### Bezier Points
```
(center_x, center_y), (right_tangent_x, right_tangent_y), (left_tangent_x, left_tangent_y)
```

- **Center**: The control point position in pixel coordinates
- **Right Tangent**: Offset from center for outgoing curve handle
- **Left Tangent**: Offset from center for incoming curve handle

#### B-Spline Points (Alternative)
```
(x, y)
```
Simple point positions; curve smoothness is automatic.

---

## Python Implementation

### Core Exporter Class

```python
"""
fxs_exporter.py - Export AI masks to Silhouette FXS format
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import cv2


@dataclass
class BezierPoint:
    """Single Bezier control point with tangent handles"""
    center: Tuple[float, float]
    right_tangent: Tuple[float, float]  # Relative to center
    left_tangent: Tuple[float, float]   # Relative to center

    def to_fxs_string(self) -> str:
        """Format point for FXS XML"""
        return (
            f"({self.center[0]:.6f},{self.center[1]:.6f}),"
            f"({self.right_tangent[0]:.6f},{self.right_tangent[1]:.6f}),"
            f"({self.left_tangent[0]:.6f},{self.left_tangent[1]:.6f})"
        )


@dataclass
class Shape:
    """Roto shape with per-frame Bezier paths"""
    label: str
    color: Tuple[float, float, float]
    frames: Dict[int, List[BezierPoint]]  # frame_num -> points
    closed: bool = True
    opacity: float = 100.0
    mode: str = "Add"


class FXSExporter:
    """Export shapes to Silhouette FXS format"""

    def __init__(self, width: int, height: int,
                 start_frame: int, end_frame: int):
        self.width = width
        self.height = height
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.shapes: List[Shape] = []

    def add_shape(self, shape: Shape):
        """Add a shape to export"""
        self.shapes.append(shape)

    def export(self, output_path: str):
        """Generate FXS file"""
        root = self._create_root()
        layer = self._create_layer(root, "AI_Generated_Roto")

        for shape in self.shapes:
            self._add_shape(layer, shape)

        self._write_xml(root, output_path)

    def _create_root(self) -> ET.Element:
        """Create Silhouette root element"""
        return ET.Element("Silhouette", {
            "width": str(self.width),
            "height": str(self.height),
            "workRangeStart": str(self.start_frame),
            "workRangeEnd": str(self.end_frame),
            "sessionStartFrame": str(self.start_frame)
        })

    def _create_layer(self, parent: ET.Element, label: str) -> ET.Element:
        """Create a layer element"""
        layer = ET.SubElement(parent, "Layer", {
            "type": "Layer",
            "label": label,
            "expanded": "True"
        })

        props = ET.SubElement(layer, "Properties")

        # Layer color
        self._add_constant_property(props, "color", "(1.0,0.5,0.0)")

        # Layer mode
        self._add_constant_property(props, "mode", "Add")

        # Objects container (will hold shapes)
        objects_prop = ET.SubElement(props, "Property", {
            "id": "objects",
            "constant": "True",
            "expanded": "True"
        })

        return objects_prop  # Return objects container for adding shapes

    def _add_shape(self, parent: ET.Element, shape: Shape):
        """Add shape to layer"""
        obj = ET.SubElement(parent, "Object", {
            "type": "Shape",
            "label": shape.label,
            "shape_type": "Bezier",
            "hidden": "False",
            "locked": "False"
        })

        props = ET.SubElement(obj, "Properties")

        # Opacity
        self._add_constant_property(props, "opacity", str(shape.opacity))

        # Motion blur
        self._add_constant_property(props, "motionBlur", "true")

        # Outline color
        color_str = f"({shape.color[0]},{shape.color[1]},{shape.color[2]})"
        self._add_constant_property(props, "outlineColor", color_str)

        # Mode
        self._add_constant_property(props, "mode", shape.mode)

        # Path data (animated)
        path_prop = ET.SubElement(props, "Property", {"id": "path"})

        for frame_num in sorted(shape.frames.keys()):
            points = shape.frames[frame_num]
            self._add_path_keyframe(path_prop, frame_num, points, shape.closed)

    def _add_path_keyframe(self, parent: ET.Element, frame: int,
                           points: List[BezierPoint], closed: bool):
        """Add a path keyframe"""
        key = ET.SubElement(parent, "Key", {
            "frame": str(frame),
            "interp": "linear"
        })

        path = ET.SubElement(key, "Path", {
            "closed": "true" if closed else "false",
            "type": "Bezier"
        })

        for point in points:
            pt_elem = ET.SubElement(path, "Point")
            pt_elem.text = point.to_fxs_string()

    def _add_constant_property(self, parent: ET.Element,
                                prop_id: str, value: str):
        """Add a constant (non-animated) property"""
        prop = ET.SubElement(parent, "Property", {
            "constant": "True",
            "id": prop_id
        })
        val = ET.SubElement(prop, "Value")
        val.text = value

    def _write_xml(self, root: ET.Element, output_path: str):
        """Write formatted XML to file"""
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
```

### Mask to Bezier Conversion

```python
"""
mask_to_bezier.py - Convert binary masks to Bezier curves
"""

import cv2
import numpy as np
from typing import List, Tuple
from scipy import interpolate


def mask_to_bezier_points(
    binary_mask: np.ndarray,
    num_points: int = 50,
    smoothing: float = 0.5
) -> List[BezierPoint]:
    """
    Convert binary mask to Bezier points suitable for FXS export.

    Args:
        binary_mask: HxW numpy array (0 or 255)
        num_points: Target number of control points
        smoothing: Contour smoothing factor (0-1)

    Returns:
        List of BezierPoint objects
    """
    # Find contours
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return []

    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()

    if len(contour) < 3:
        return []

    # Simplify contour
    epsilon = smoothing * cv2.arcLength(contour.reshape(-1, 1, 2), True) / 100
    simplified = cv2.approxPolyDP(
        contour.reshape(-1, 1, 2), epsilon, True
    ).squeeze()

    # Resample to target point count
    if len(simplified) > num_points:
        indices = np.linspace(0, len(simplified) - 1, num_points, dtype=int)
        simplified = simplified[indices]

    # Calculate tangents for Bezier handles
    bezier_points = []
    n = len(simplified)

    for i in range(n):
        # Current point
        curr = simplified[i]

        # Previous and next points (wrap around for closed shape)
        prev_pt = simplified[(i - 1) % n]
        next_pt = simplified[(i + 1) % n]

        # Calculate tangent direction
        tangent = next_pt - prev_pt
        tangent_len = np.linalg.norm(tangent)

        if tangent_len > 0:
            tangent = tangent / tangent_len
        else:
            tangent = np.array([1.0, 0.0])

        # Scale tangent based on distance to neighbors
        dist_prev = np.linalg.norm(curr - prev_pt)
        dist_next = np.linalg.norm(curr - next_pt)

        # Tangent handle length (typically 1/3 of distance to neighbor)
        handle_scale = 0.33

        right_handle = tangent * dist_next * handle_scale
        left_handle = -tangent * dist_prev * handle_scale

        bezier_points.append(BezierPoint(
            center=(float(curr[0]), float(curr[1])),
            right_tangent=(float(right_handle[0]), float(right_handle[1])),
            left_tangent=(float(left_handle[0]), float(left_handle[1]))
        ))

    return bezier_points


def optimize_keyframes(
    shapes_by_frame: Dict[int, List[BezierPoint]],
    tolerance: float = 0.5
) -> Dict[int, List[BezierPoint]]:
    """
    Remove redundant keyframes where shapes haven't changed significantly.

    Args:
        shapes_by_frame: Dictionary mapping frame numbers to point lists
        tolerance: Maximum allowed point deviation to consider "same"

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

        curr_points = shapes_by_frame[curr_frame]
        prev_points = shapes_by_frame[prev_frame]

        # Check if points are significantly different
        if len(curr_points) != len(prev_points):
            optimized[curr_frame] = curr_points
            continue

        max_deviation = 0.0
        for cp, pp in zip(curr_points, prev_points):
            deviation = np.sqrt(
                (cp.center[0] - pp.center[0])**2 +
                (cp.center[1] - pp.center[1])**2
            )
            max_deviation = max(max_deviation, deviation)

        if max_deviation > tolerance:
            optimized[curr_frame] = curr_points

    # Always include last frame
    optimized[frames[-1]] = shapes_by_frame[frames[-1]]

    return optimized
```

### Usage Example

```python
"""
example_usage.py - Complete workflow example
"""

from fxs_exporter import FXSExporter, Shape, BezierPoint
from mask_to_bezier import mask_to_bezier_points, optimize_keyframes
import cv2

def process_shot(
    mask_folder: str,
    output_fxs: str,
    width: int = 1920,
    height: int = 1080,
    start_frame: int = 1001
):
    """
    Process a folder of mask images and export to FXS.
    """
    # Collect masks
    import glob
    mask_files = sorted(glob.glob(f"{mask_folder}/*.png"))

    # Convert each mask to Bezier points
    shapes_by_frame = {}
    for i, mask_file in enumerate(mask_files):
        frame_num = start_frame + i

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        points = mask_to_bezier_points(mask, num_points=40)

        if points:
            shapes_by_frame[frame_num] = points

    # Optimize keyframes
    optimized = optimize_keyframes(shapes_by_frame, tolerance=1.0)

    # Create shape
    shape = Shape(
        label="AI_Segmented_Object",
        color=(1.0, 0.0, 0.0),
        frames=optimized,
        closed=True,
        opacity=100.0
    )

    # Export
    exporter = FXSExporter(
        width=width,
        height=height,
        start_frame=start_frame,
        end_frame=start_frame + len(mask_files) - 1
    )
    exporter.add_shape(shape)
    exporter.export(output_fxs)

    print(f"Exported {len(optimized)} keyframes to {output_fxs}")


# Run
process_shot(
    mask_folder="./output/masks/person_01",
    output_fxs="./exports/shot_001.fxs"
)
```

---

## Quality Considerations

### Point Count Guidelines

| Object Type | Recommended Points | Notes |
|-------------|-------------------|-------|
| Simple shape (box, circle) | 8-12 | Minimal complexity |
| Body silhouette | 30-50 | Good balance |
| Complex outline | 60-100 | Detailed edges |
| Hair/fur detail | Use matting instead | Binary mask insufficient |

### Tangent Calculation

For smooth Bezier curves that artists can easily adjust:

1. **Tangent Direction**: Aligned with curve flow (average of vectors to neighbors)
2. **Tangent Length**: ~1/3 of distance to neighboring points
3. **Corner Detection**: Reduce tangent length at sharp corners

### Temporal Consistency

1. **Point Correspondence**: Maintain same point count across frames
2. **Point Ordering**: Consistent winding order (clockwise/counter-clockwise)
3. **Keyframe Optimization**: Remove redundant frames where shape is static

---

## Testing Checklist

- [ ] FXS file opens in Silhouette without errors
- [ ] Shapes appear at correct position and scale
- [ ] Animation plays back smoothly
- [ ] Points are editable in Silhouette
- [ ] Tangent handles behave as expected
- [ ] Export works for various frame ranges (1001-based, 0-based)
- [ ] Multiple shapes in single file work correctly
- [ ] Layer hierarchy is preserved
