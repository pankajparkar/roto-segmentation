"""
Silhouette FXS Exporter.

Exports Bezier shapes to Silhouette's native .fxs format.
Shapes can be imported directly into Silhouette for artist refinement.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom

from roto_seg.exporters.base import BaseExporter, ExportFormat
from roto_seg.services.mask_to_bezier import BezierShape, BezierPoint


class FXSExporter(BaseExporter):
    """
    Export shapes to Silhouette FXS format.

    Creates XML files that can be imported into SilhouetteFX
    with full editing capabilities (Bezier points, tangent handles).
    """

    format = ExportFormat.SILHOUETTE

    # Default colors for shapes (RGB, 0-1 range)
    SHAPE_COLORS = [
        (1.0, 0.0, 0.0),    # Red
        (0.0, 1.0, 0.0),    # Green
        (0.0, 0.0, 1.0),    # Blue
        (1.0, 1.0, 0.0),    # Yellow
        (1.0, 0.0, 1.0),    # Magenta
        (0.0, 1.0, 1.0),    # Cyan
        (1.0, 0.5, 0.0),    # Orange
        (0.5, 0.0, 1.0),    # Purple
    ]

    def export(
        self,
        shapes_by_frame: Dict[int, List[BezierShape]],
        output_path: str,
        layer_name: Optional[str] = None,
    ) -> str:
        """
        Export shapes to FXS file.

        Args:
            shapes_by_frame: Dictionary mapping frame numbers to shape lists
            output_path: Path for output .fxs file
            layer_name: Name for the layer (default: "AI_Generated_Roto")

        Returns:
            Path to created file
        """
        output_path = self.validate_output_path(output_path)

        # Ensure .fxs extension
        if not str(output_path).endswith('.fxs'):
            output_path = Path(str(output_path) + '.fxs')

        # Determine frame range from data
        frames = sorted(shapes_by_frame.keys())
        if frames:
            self.start_frame = min(self.start_frame, frames[0])
            self.end_frame = max(self.end_frame, frames[-1])

        # Build XML structure
        root = self._create_root()
        layer = self._create_layer(root, layer_name or "AI_Generated_Roto")

        # Group shapes by label for consistent IDs
        shape_ids = self._get_shape_ids(shapes_by_frame)

        # Create shape elements
        for shape_label, shape_id in shape_ids.items():
            self._add_shape(
                layer,
                shape_label,
                shape_id,
                shapes_by_frame,
            )

        # Write formatted XML
        self._write_xml(root, str(output_path))

        return str(output_path)

    def _create_root(self) -> ET.Element:
        """Create the Silhouette root element."""
        return ET.Element("Silhouette", {
            "width": str(self.width),
            "height": str(self.height),
            "workRangeStart": str(self.start_frame),
            "workRangeEnd": str(self.end_frame),
            "sessionStartFrame": str(self.start_frame),
        })

    def _create_layer(
        self,
        parent: ET.Element,
        label: str,
    ) -> ET.Element:
        """Create a layer element that will contain shapes."""
        layer = ET.SubElement(parent, "Layer", {
            "type": "Layer",
            "label": label,
            "expanded": "True",
        })

        props = ET.SubElement(layer, "Properties")

        # Layer color
        self._add_constant_property(props, "color", "(1.0, 0.5, 0.0)")

        # Layer invert
        self._add_constant_property(props, "invert", "false")

        # Layer mode
        self._add_constant_property(props, "mode", "Add")

        # Identity transform matrix
        transform_prop = ET.SubElement(props, "Property", {"id": "transform.matrix"})
        key = ET.SubElement(transform_prop, "Key", {
            "frame": str(self.start_frame),
            "interp": "linear",
        })
        key.text = "(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)"

        # Objects container (will hold shapes)
        objects_prop = ET.SubElement(props, "Property", {
            "id": "objects",
            "constant": "True",
            "expanded": "True",
        })

        return objects_prop

    def _get_shape_ids(
        self,
        shapes_by_frame: Dict[int, List[BezierShape]],
    ) -> Dict[str, int]:
        """
        Assign consistent IDs to shapes based on their labels.

        Returns:
            Dictionary mapping shape labels to IDs
        """
        labels = set()
        for frame_shapes in shapes_by_frame.values():
            for shape in frame_shapes:
                labels.add(shape.label or "shape")

        return {label: i for i, label in enumerate(sorted(labels))}

    def _add_shape(
        self,
        parent: ET.Element,
        shape_label: str,
        shape_id: int,
        shapes_by_frame: Dict[int, List[BezierShape]],
    ):
        """Add a shape element with all its keyframes."""
        # Get color for this shape
        color = self.SHAPE_COLORS[shape_id % len(self.SHAPE_COLORS)]

        obj = ET.SubElement(parent, "Object", {
            "type": "Shape",
            "label": shape_label,
            "shape_type": "Bezier",
            "hidden": "False",
            "locked": "False",
        })

        props = ET.SubElement(obj, "Properties")

        # Opacity
        self._add_constant_property(props, "opacity", "100")

        # Motion blur
        self._add_constant_property(props, "motionBlur", "true")

        # Outline color
        color_str = f"({color[0]},{color[1]},{color[2]})"
        self._add_constant_property(props, "outlineColor", color_str)

        # Mode
        self._add_constant_property(props, "mode", "Add")

        # Invert
        self._add_constant_property(props, "invert", "false")

        # Path data (animated)
        path_prop = ET.SubElement(props, "Property", {"id": "path"})

        # Add keyframes for this shape
        for frame_num in sorted(shapes_by_frame.keys()):
            frame_shapes = shapes_by_frame[frame_num]

            # Find shape with matching label
            matching_shapes = [
                s for s in frame_shapes
                if (s.label or "shape") == shape_label
            ]

            if matching_shapes:
                shape = matching_shapes[0]
                self._add_path_keyframe(
                    path_prop,
                    frame_num,
                    shape.points,
                    shape.closed,
                )

    def _add_path_keyframe(
        self,
        parent: ET.Element,
        frame: int,
        points: List[BezierPoint],
        closed: bool,
    ):
        """Add a path keyframe."""
        key = ET.SubElement(parent, "Key", {
            "frame": str(frame),
            "interp": "linear",
        })

        path = ET.SubElement(key, "Path", {
            "closed": "true" if closed else "false",
            "type": "Bezier",
        })

        for point in points:
            pt_elem = ET.SubElement(path, "Point")
            pt_elem.text = point.to_fxs_string()

    def _add_constant_property(
        self,
        parent: ET.Element,
        prop_id: str,
        value: str,
    ):
        """Add a constant (non-animated) property."""
        prop = ET.SubElement(parent, "Property", {
            "constant": "True",
            "id": prop_id,
        })
        val = ET.SubElement(prop, "Value")
        val.text = value

    def _write_xml(self, root: ET.Element, output_path: str):
        """Write formatted XML to file."""
        # Convert to string
        rough_string = ET.tostring(root, encoding='unicode')

        # Pretty print
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Remove extra blank lines (minidom adds them)
        lines = [
            line for line in pretty_xml.split('\n')
            if line.strip()
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def export_masks_to_fxs(
    masks_by_frame: Dict[int, "np.ndarray"],
    output_path: str,
    width: int,
    height: int,
    start_frame: int = 1001,
    layer_name: str = "AI_Roto",
    target_points: int = 40,
) -> str:
    """
    Convenience function to export masks directly to FXS.

    Args:
        masks_by_frame: Dictionary mapping frame numbers to binary masks
        output_path: Path for output .fxs file
        width: Frame width
        height: Frame height
        start_frame: Starting frame number
        layer_name: Name for the layer
        target_points: Target number of Bezier points per shape

    Returns:
        Path to created file
    """
    from roto_seg.services.mask_to_bezier import mask_to_shapes, optimize_keyframes

    # Convert masks to shapes
    shapes_by_frame = {}
    for frame_num, mask in masks_by_frame.items():
        shapes = mask_to_shapes(
            mask,
            target_points=target_points,
            label=layer_name,
        )
        if shapes:
            shapes_by_frame[frame_num] = shapes

    # Optimize keyframes
    shapes_by_frame = optimize_keyframes(shapes_by_frame)

    # Export
    exporter = FXSExporter(
        width=width,
        height=height,
        start_frame=start_frame,
        end_frame=start_frame + len(masks_by_frame) - 1,
    )

    return exporter.export(shapes_by_frame, output_path, layer_name)
