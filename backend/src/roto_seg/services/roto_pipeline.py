"""
Roto Pipeline - End-to-end rotoscoping automation.

Coordinates AI segmentation, mask processing, and export
into a single easy-to-use pipeline.
"""

import gc
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
import numpy as np

from roto_seg.ai.segmentation import SegmentationService, get_segmentation_service
from roto_seg.services.mask_to_bezier import (
    mask_to_shapes,
    optimize_keyframes,
    BezierShape,
)
from roto_seg.services.video import get_reader, VideoInfo
from roto_seg.exporters.fxs_exporter import FXSExporter
from roto_seg.exporters.exr_exporter import EXRExporter


@dataclass
class SegmentationPrompt:
    """Prompt for segmentation."""
    frame_idx: int
    points: Optional[np.ndarray] = None  # (N, 2) array of [x, y]
    point_labels: Optional[np.ndarray] = None  # (N,) array, 1=foreground, 0=background
    box: Optional[np.ndarray] = None  # [x1, y1, x2, y2]
    object_id: int = 1
    label: str = "object"


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    success: bool
    message: str
    output_path: Optional[str] = None
    frame_count: int = 0
    object_count: int = 0
    video_info: Optional[VideoInfo] = None
    errors: List[str] = field(default_factory=list)


class RotoPipeline:
    """
    End-to-end rotoscoping automation pipeline.

    Usage:
        pipeline = RotoPipeline()
        result = pipeline.process(
            video_path="shot.mp4",
            prompts=[
                SegmentationPrompt(
                    frame_idx=0,
                    points=np.array([[500, 300]]),
                    point_labels=np.array([1]),
                    label="person"
                )
            ],
            output_path="output.fxs",
        )
    """

    def __init__(
        self,
        segmentation_service: Optional[SegmentationService] = None,
        target_points: int = 40,
        smoothing: float = 0.5,
    ):
        """
        Initialize pipeline.

        Args:
            segmentation_service: Segmentation service to use (creates default if None)
            target_points: Target Bezier points per shape
            smoothing: Contour smoothing factor (0-1)
        """
        self.segmentation_service = segmentation_service or get_segmentation_service()
        self.target_points = target_points
        self.smoothing = smoothing

    def process(
        self,
        video_path: str,
        prompts: List[SegmentationPrompt],
        output_path: str,
        output_format: str = "silhouette",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        frame_step: int = 1,
        propagate: bool = True,
    ) -> PipelineResult:
        """
        Process video with AI segmentation and export results.

        Args:
            video_path: Path to input video or image sequence
            prompts: List of segmentation prompts (one per object)
            output_path: Path for output file
            output_format: Export format ("silhouette", "nuke", "png")
            progress_callback: Optional callback(current, total, message)
            frame_step: Process every Nth frame (1 = all frames)
            propagate: Whether to propagate masks through video

        Returns:
            PipelineResult with status and output info
        """
        errors = []

        try:
            # Get video reader
            reader = get_reader(video_path)
            video_info = reader.info

            # Debug: Log video dimensions
            print(f"[DEBUG] Video Info: width={video_info.width}, height={video_info.height}, frames={video_info.frame_count}, fps={video_info.fps}")

            if progress_callback:
                progress_callback(0, video_info.frame_count, "Loading video...")

            # Process each object
            all_masks: Dict[int, Dict[int, np.ndarray]] = {}  # object_id -> {frame -> mask}

            for prompt in prompts:
                if progress_callback:
                    progress_callback(
                        0, video_info.frame_count,
                        f"Segmenting {prompt.label}..."
                    )

                try:
                    masks = self._segment_object(
                        reader, prompt, video_info,
                        propagate, frame_step, progress_callback
                    )
                    all_masks[prompt.object_id] = masks
                except Exception as e:
                    errors.append(f"Failed to segment {prompt.label}: {str(e)}")

            if not all_masks:
                return PipelineResult(
                    success=False,
                    message="No objects were successfully segmented",
                    errors=errors,
                )

            # Export based on format
            if progress_callback:
                progress_callback(
                    video_info.frame_count,
                    video_info.frame_count,
                    "Exporting..."
                )

            # For EXR export, use raw masks directly
            if output_format == "exr":
                output_file = self._export_exr(
                    all_masks,
                    output_path,
                    video_info,
                    prompts,
                )
                # Count frames from masks
                frame_count = sum(len(masks) for masks in all_masks.values())

                return PipelineResult(
                    success=True,
                    message=f"Exported {len(prompts)} objects to {output_file}",
                    output_path=output_file,
                    frame_count=frame_count,
                    object_count=len(prompts),
                    video_info=video_info,
                    errors=errors,
                )
            else:
                # Convert masks to shapes for vector formats
                if progress_callback:
                    progress_callback(
                        video_info.frame_count - 1,
                        video_info.frame_count,
                        "Converting to shapes..."
                    )

                shapes_by_frame = self._masks_to_shapes(all_masks, prompts)
                shapes_by_frame = optimize_keyframes(shapes_by_frame)

                output_file = self._export(
                    shapes_by_frame,
                    output_path,
                    output_format,
                    video_info,
                )

                return PipelineResult(
                    success=True,
                    message=f"Exported {len(prompts)} objects to {output_file}",
                    output_path=output_file,
                    frame_count=len(shapes_by_frame),
                    object_count=len(prompts),
                    video_info=video_info,
                    errors=errors,
                )

        except Exception as e:
            return PipelineResult(
                success=False,
                message=f"Pipeline failed: {str(e)}",
                errors=errors + [str(e)],
            )
        finally:
            gc.collect()

    def _segment_object(
        self,
        reader: Union["VideoReader", "ImageSequenceReader"],
        prompt: SegmentationPrompt,
        video_info: VideoInfo,
        propagate: bool,
        frame_step: int,
        progress_callback: Optional[Callable],
    ) -> Dict[int, np.ndarray]:
        """Segment a single object through video."""
        masks = {}

        # Get the initial frame
        initial_frame = reader.read_frame(prompt.frame_idx)
        if initial_frame is None:
            raise ValueError(f"Could not read frame {prompt.frame_idx}")

        # Debug: Log frame dimensions and click point
        frame_h, frame_w = initial_frame.shape[:2]
        print(f"[DEBUG] Frame {prompt.frame_idx}: shape=({frame_w}x{frame_h}), click_point={prompt.points}, labels={prompt.point_labels}")

        # Validate click point is within frame bounds
        if prompt.points is not None:
            for pt in prompt.points:
                px, py = pt[0], pt[1]
                if px < 0 or px >= frame_w or py < 0 or py >= frame_h:
                    print(f"[WARNING] Click point ({px}, {py}) is outside frame bounds ({frame_w}x{frame_h})")

        # Segment initial frame
        mask_result, scores, _ = self.segmentation_service.segment_image(
            initial_frame,
            points=prompt.points,
            point_labels=prompt.point_labels,
            box=prompt.box,
            multimask_output=True,
        )

        # Use best mask
        best_idx = scores.argmax()
        initial_mask = mask_result[best_idx]
        masks[prompt.frame_idx] = initial_mask

        if not propagate:
            return masks

        # Propagate through video
        # For now, use simple mask propagation (SAM2 video mode or frame-by-frame)
        # In production, this would use SAM2's video propagation

        # Simple frame-by-frame propagation using previous mask as prompt
        prev_mask = initial_mask

        # Forward propagation
        for frame_idx, frame in reader.iter_frames(
            start=prompt.frame_idx + frame_step,
            step=frame_step,
        ):
            if progress_callback:
                progress_callback(
                    frame_idx,
                    video_info.frame_count,
                    f"Frame {frame_idx}/{video_info.frame_count}"
                )

            # Use center of previous mask as prompt
            points = self._get_mask_center(prev_mask)
            if points is not None:
                mask_result, scores, _ = self.segmentation_service.segment_image(
                    frame,
                    points=points,
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                best_idx = scores.argmax()
                masks[frame_idx] = mask_result[best_idx]
                prev_mask = mask_result[best_idx]
            else:
                # Lost the object
                break

        # Backward propagation (from initial frame)
        prev_mask = initial_mask
        for frame_idx in range(prompt.frame_idx - frame_step, -1, -frame_step):
            frame = reader.read_frame(frame_idx)
            if frame is None:
                break

            if progress_callback:
                progress_callback(
                    prompt.frame_idx - frame_idx,
                    video_info.frame_count,
                    f"Frame {frame_idx}/{video_info.frame_count} (backward)"
                )

            points = self._get_mask_center(prev_mask)
            if points is not None:
                mask_result, scores, _ = self.segmentation_service.segment_image(
                    frame,
                    points=points,
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                best_idx = scores.argmax()
                masks[frame_idx] = mask_result[best_idx]
                prev_mask = mask_result[best_idx]
            else:
                break

        return masks

    def _get_mask_center(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Get center point of mask for propagation."""
        if mask.sum() == 0:
            return None

        # Find center of mass
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0:
            return None

        cx = x_indices.mean()
        cy = y_indices.mean()

        return np.array([[cx, cy]])

    def _masks_to_shapes(
        self,
        all_masks: Dict[int, Dict[int, np.ndarray]],
        prompts: List[SegmentationPrompt],
    ) -> Dict[int, List[BezierShape]]:
        """Convert all masks to Bezier shapes."""
        # Create label lookup
        labels = {p.object_id: p.label for p in prompts}

        # Group shapes by frame
        shapes_by_frame: Dict[int, List[BezierShape]] = {}

        for object_id, masks in all_masks.items():
            label = labels.get(object_id, f"object_{object_id}")

            for frame_idx, mask in masks.items():
                shapes = mask_to_shapes(
                    mask,
                    target_points=self.target_points,
                    smoothing=self.smoothing,
                    label=label,
                )

                if frame_idx not in shapes_by_frame:
                    shapes_by_frame[frame_idx] = []

                shapes_by_frame[frame_idx].extend(shapes)

        return shapes_by_frame

    def _export(
        self,
        shapes_by_frame: Dict[int, List[BezierShape]],
        output_path: str,
        output_format: str,
        video_info: VideoInfo,
    ) -> str:
        """Export shapes to file."""
        frames = sorted(shapes_by_frame.keys())
        start_frame = frames[0] if frames else 1001
        end_frame = frames[-1] if frames else 1100

        if output_format == "silhouette":
            exporter = FXSExporter(
                width=video_info.width,
                height=video_info.height,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=video_info.fps,
            )
            return exporter.export(shapes_by_frame, output_path)

        raise ValueError(f"Unsupported output format: {output_format}")

    def _export_exr(
        self,
        all_masks: Dict[int, Dict[int, np.ndarray]],
        output_path: str,
        video_info: VideoInfo,
        prompts: List[SegmentationPrompt],
    ) -> str:
        """Export masks as EXR sequence."""
        output_dir = Path(output_path)
        if output_dir.suffix:
            # If a file path was given, use its parent dir
            output_dir = output_dir.parent / output_dir.stem

        # Combine all object masks into single matte per frame
        combined_masks: Dict[int, np.ndarray] = {}

        for object_id, masks in all_masks.items():
            for frame_idx, mask in masks.items():
                if frame_idx not in combined_masks:
                    combined_masks[frame_idx] = mask.astype(np.float32)
                else:
                    # Combine masks (max for overlapping regions)
                    combined_masks[frame_idx] = np.maximum(
                        combined_masks[frame_idx],
                        mask.astype(np.float32)
                    )

        # Get label for filename
        label = prompts[0].label if prompts else "Matte"

        exporter = EXRExporter(
            output_dir=str(output_dir),
            prefix=label,
            start_frame=min(combined_masks.keys()) if combined_masks else 1001,
        )

        return exporter.export_masks(
            combined_masks,
            width=video_info.width,
            height=video_info.height,
        )


def quick_roto(
    video_path: str,
    click_x: int,
    click_y: int,
    frame_idx: int = 0,
    output_path: Optional[str] = None,
    label: str = "object",
) -> PipelineResult:
    """
    Quick rotoscoping with a single click.

    Args:
        video_path: Path to video
        click_x: X coordinate of click
        click_y: Y coordinate of click
        frame_idx: Frame where click was made
        output_path: Output path (auto-generated if None)
        label: Label for the object

    Returns:
        PipelineResult
    """
    if output_path is None:
        output_path = str(Path(video_path).with_suffix('.fxs'))

    pipeline = RotoPipeline()
    return pipeline.process(
        video_path=video_path,
        prompts=[
            SegmentationPrompt(
                frame_idx=frame_idx,
                points=np.array([[click_x, click_y]]),
                point_labels=np.array([1]),
                label=label,
            )
        ],
        output_path=output_path,
    )
