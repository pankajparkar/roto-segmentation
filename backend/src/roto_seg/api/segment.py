"""
Direct segmentation API endpoints.

Provides synchronous segmentation for testing and small jobs.
For production, use the async job-based API in shots.py.
"""

from typing import List, Optional
from pathlib import Path
import tempfile
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np

router = APIRouter()


class PointPrompt(BaseModel):
    """Point prompt for segmentation."""
    x: float
    y: float
    label: int = 1  # 1 = foreground, 0 = background


class BoxPrompt(BaseModel):
    """Box prompt for segmentation."""
    x1: float
    y1: float
    x2: float
    y2: float


class SegmentImageRequest(BaseModel):
    """Request for single image segmentation."""
    points: Optional[List[PointPrompt]] = None
    box: Optional[BoxPrompt] = None


class SegmentImageResponse(BaseModel):
    """Response from image segmentation."""
    success: bool
    message: str
    mask_shape: Optional[List[int]] = None
    num_masks: int = 0
    best_score: float = 0.0


class DeviceInfoResponse(BaseModel):
    """Device information response."""
    platform: str
    torch_version: str
    recommended_device: str
    mps_available: bool
    cuda_available: bool
    active_model: str
    sam2_available: bool
    sam2_enabled: bool


@router.get("/device-info", response_model=DeviceInfoResponse)
async def get_device_info():
    """Get information about available compute devices."""
    from roto_seg.ai.device import get_device_info
    from roto_seg.ai.segmentation import get_segmentation_service

    info = get_device_info()
    seg_info = get_segmentation_service().get_runtime_info()
    return DeviceInfoResponse(
        platform=info["platform"],
        torch_version=info["torch_version"],
        recommended_device=info["recommended"],
        mps_available=info["devices"]["mps"]["available"],
        cuda_available=info["devices"]["cuda"]["available"],
        active_model=seg_info["model_name"],
        sam2_available=seg_info["sam2_available"],
        sam2_enabled=seg_info["use_sam2"],
    )


@router.post("/segment-image")
async def segment_image(
    image: UploadFile = File(...),
    points: Optional[str] = Form(None),  # JSON: [[x, y, label], ...]
    box: Optional[str] = Form(None),  # JSON: [x1, y1, x2, y2]
):
    """
    Segment objects in a single image.

    Upload an image and provide point or box prompts.
    Returns the segmentation mask.

    Example:
        curl -X POST "http://localhost:8000/api/v1/segment/segment-image" \
            -F "image=@photo.jpg" \
            -F 'points=[[500, 300, 1]]'
    """
    import json
    import cv2

    from roto_seg.ai.segmentation import get_segmentation_service

    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Parse prompts
    point_coords = None
    point_labels = None
    box_prompt = None

    if points:
        try:
            points_data = json.loads(points)
            point_coords = np.array([[p[0], p[1]] for p in points_data])
            point_labels = np.array([p[2] if len(p) > 2 else 1 for p in points_data])
        except (json.JSONDecodeError, IndexError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid points format. Expected [[x, y, label], ...]: {e}"
            )

    if box:
        try:
            box_prompt = np.array(json.loads(box))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid box format. Expected [x1, y1, x2, y2]: {e}"
            )

    if point_coords is None and box_prompt is None:
        raise HTTPException(
            status_code=400,
            detail="Must provide either points or box prompt"
        )

    # Run segmentation
    service = get_segmentation_service()

    try:
        masks, scores, _ = service.segment_image(
            img_rgb,
            points=point_coords,
            point_labels=point_labels,
            box=box_prompt,
            multimask_output=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )

    # Smart mask selection: SAM2 returns 3 masks with multimask_output=True
    # - Mask 0: Small/tight mask
    # - Mask 1: Medium mask (usually the object)
    # - Mask 2: Large mask (often includes background)
    #
    # Instead of just picking highest score, we prefer masks that:
    # 1. Cover a reasonable portion (not too small <1%, not too large >70%)
    # 2. Have good confidence score

    print(f"[DEBUG] Segmentation: {len(masks)} masks returned")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        coverage = mask.sum() / mask.size * 100
        print(f"[DEBUG]   Mask {i}: score={score:.3f}, coverage={coverage:.1f}%")

    # Calculate coverage for each mask
    coverages = [mask.sum() / mask.size * 100 for mask in masks]

    # Find best mask: prefer medium-sized masks with good scores
    best_idx = None
    best_combined_score = -1

    for i, (score, coverage) in enumerate(zip(scores, coverages)):
        # Penalize very small masks (<1%) and very large masks (>70%)
        if coverage < 1:
            size_penalty = 0.3
        elif coverage > 70:
            size_penalty = 0.5
        elif coverage > 50:
            size_penalty = 0.8
        else:
            size_penalty = 1.0

        combined = float(score) * size_penalty
        print(f"[DEBUG]   Mask {i}: combined_score={combined:.3f} (score={score:.3f} * penalty={size_penalty})")

        if combined > best_combined_score:
            best_combined_score = combined
            best_idx = i

    best_mask = masks[best_idx]
    mask_coverage = coverages[best_idx]
    print(f"[DEBUG] Selected mask {best_idx}: score={scores[best_idx]:.3f}, coverage={mask_coverage:.1f}%")

    # Convert to PNG and return
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    _, encoded = cv2.imencode('.png', mask_uint8)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(encoded.tobytes())
        temp_path = f.name

    return FileResponse(
        temp_path,
        media_type="image/png",
        filename="mask.png",
        headers={
            "X-Mask-Score": str(float(scores[best_idx])),
            "X-Num-Masks": str(len(masks)),
        }
    )


@router.post("/quick-roto")
async def quick_roto(
    video: UploadFile = File(...),
    click_x: Optional[int] = Form(None),
    click_y: Optional[int] = Form(None),
    box: Optional[str] = Form(None),  # JSON: [x1, y1, x2, y2]
    points: Optional[str] = Form(None),  # JSON: [[x, y, label], ...] for multiple annotations
    frame_idx: int = Form(0),
    label: str = Form("object"),
    output_format: str = Form("silhouette"),
    propagation_mode: str = Form("auto"),
    matting: bool = Form(False),
    matting_model: str = Form("vitmatte"),
    temporal_smooth: float = Form(0.0),
):
    """
    Quick rotoscoping with point annotations, single click, or bounding box.

    Upload a video, annotate the object (add/remove points), get an FXS file.
    Supports multiple annotation points with labels for refinement.

    Industry-standard controls:
        - Click = Add to selection (label=1, foreground)
        - Alt+Click = Remove from selection (label=0, background)

    This is a synchronous endpoint for small videos.
    For large videos, use the async job API.

    Examples:
        # Multiple annotation points (recommended):
        curl -X POST "http://localhost:8000/api/v1/segment/quick-roto" \
            -F "video=@clip.mp4" \
            -F 'points=[[500, 300, 1], [600, 400, 1], [450, 250, 0]]' \
            -F "frame_idx=0" \
            -F "label=person"

        # Single point selection (legacy):
        curl -X POST "http://localhost:8000/api/v1/segment/quick-roto" \
            -F "video=@clip.mp4" \
            -F "click_x=500" \
            -F "click_y=300" \
            -F "frame_idx=0" \
            -F "label=person"

        # Box selection:
        curl -X POST "http://localhost:8000/api/v1/segment/quick-roto" \
            -F "video=@clip.mp4" \
            -F 'box=[100, 150, 400, 350]' \
            -F "frame_idx=0" \
            -F "label=car"
    """
    import logging
    import json
    logger = logging.getLogger(__name__)

    # Parse multiple annotation points if provided
    annotation_points = None
    annotation_labels = None
    if points:
        try:
            points_data = json.loads(points)
            annotation_points = np.array([[p[0], p[1]] for p in points_data])
            annotation_labels = np.array([p[2] if len(p) > 2 else 1 for p in points_data])
            print(f"[DEBUG] Parsed annotation points: {len(points_data)} points")
            for i, p in enumerate(points_data):
                action = "ADD" if (p[2] if len(p) > 2 else 1) == 1 else "REMOVE"
                print(f"[DEBUG]   Point {i+1}: ({p[0]}, {p[1]}) - {action}")
        except (json.JSONDecodeError, IndexError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid points format. Expected [[x, y, label], ...]: {e}"
            )

    # Parse box if provided
    box_coords = None
    if box:
        try:
            box_coords = json.loads(box)
            if len(box_coords) != 4:
                raise HTTPException(
                    status_code=400,
                    detail="Box must have 4 coordinates: [x1, y1, x2, y2]"
                )
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid box format. Expected JSON array [x1, y1, x2, y2]: {e}"
            )

    # Validate we have at least one selection method
    has_annotations = annotation_points is not None and len(annotation_points) > 0
    has_point = click_x is not None and click_y is not None
    has_box = box_coords is not None

    if not has_annotations and not has_point and not has_box:
        raise HTTPException(
            status_code=400,
            detail="Must provide annotation points, click coordinates (click_x, click_y), or box coordinates"
        )
    if temporal_smooth < 0 or temporal_smooth > 1:
        raise HTTPException(
            status_code=400,
            detail="temporal_smooth must be between 0.0 and 1.0"
        )

    # Debug log the received selection
    if has_annotations:
        fg_count = int(np.sum(annotation_labels == 1))
        bg_count = int(np.sum(annotation_labels == 0))
        logger.info(f"Quick Roto Request: {len(annotation_points)} points ({fg_count} add, {bg_count} remove), frame_idx={frame_idx}, label={label}, format={output_format}")
        print(f"[DEBUG] Quick Roto: {len(annotation_points)} annotation points ({fg_count} add, {bg_count} remove), frame={frame_idx}, label={label}, format={output_format}")
    elif has_box:
        logger.info(f"Quick Roto Request: box={box_coords}, frame_idx={frame_idx}, label={label}, format={output_format}")
        print(f"[DEBUG] Quick Roto: box={box_coords}, frame={frame_idx}, label={label}, format={output_format}")
    else:
        logger.info(f"Quick Roto Request: click_x={click_x}, click_y={click_y}, frame_idx={frame_idx}, label={label}, format={output_format}")
        print(f"[DEBUG] Quick Roto: click=({click_x}, {click_y}), frame={frame_idx}, label={label}, format={output_format}")

    from roto_seg.ai.segmentation import get_segmentation_service
    from roto_seg.services.roto_pipeline import RotoPipeline, SegmentationPrompt

    # Save uploaded video to temp file
    temp_dir = Path(tempfile.mkdtemp())
    video_path = temp_dir / video.filename

    try:
        with open(video_path, 'wb') as f:
            shutil.copyfileobj(video.file, f)

        # Determine output path based on format
        if output_format == "silhouette":
            output_ext = ".fxs"
            output_path = temp_dir / f"output{output_ext}"
        elif output_format == "exr":
            output_ext = ".exr"
            output_path = temp_dir / "exr_output"  # Directory for EXR sequence
        else:
            output_ext = ".nk"
            output_path = temp_dir / f"output{output_ext}"

        # Run pipeline
        pipeline = RotoPipeline()
        seg_runtime = get_segmentation_service().get_runtime_info()
        print(
            "[DEBUG] Quick Roto Runtime:",
            f"model={seg_runtime['model_name']}, use_sam2={seg_runtime['use_sam2']}, "
            f"sam2_available={seg_runtime['sam2_available']}, device={seg_runtime['device']}, "
            f"propagation_mode={propagation_mode}, matting={matting}, "
            f"matting_model={matting_model}, temporal_smooth={temporal_smooth}"
        )

        # Create prompt based on selection mode (priority: annotations > box > single click)
        if has_annotations:
            # Multiple annotation points with labels
            prompt = SegmentationPrompt(
                frame_idx=frame_idx,
                points=annotation_points,
                point_labels=annotation_labels,
                label=label,
            )
        elif has_box:
            prompt = SegmentationPrompt(
                frame_idx=frame_idx,
                box=np.array(box_coords),
                label=label,
            )
        else:
            # Legacy single click
            prompt = SegmentationPrompt(
                frame_idx=frame_idx,
                points=np.array([[click_x, click_y]]),
                point_labels=np.array([1]),
                label=label,
            )

        result = pipeline.process(
            video_path=str(video_path),
            prompts=[prompt],
            output_path=str(output_path),
            output_format=output_format,
            propagation_mode=propagation_mode,
            matting=matting,
            matting_model=matting_model,
            temporal_smooth=temporal_smooth,
        )

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed: {result.message}"
            )

        # For EXR format, zip the output directory
        if output_format == "exr":
            import zipfile
            zip_path = temp_dir / f"{label}_exr.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                exr_dir = Path(result.output_path)
                for exr_file in exr_dir.glob("*.exr"):
                    zf.write(exr_file, exr_file.name)

            return FileResponse(
                str(zip_path),
                media_type="application/zip",
                filename=f"{label}_exr.zip",
                headers={
                    "X-Frame-Count": str(result.frame_count),
                    "X-Object-Count": str(result.object_count),
                    "X-Propagation-Mode": result.propagation_mode,
                    "X-Active-Model": seg_runtime["model_name"],
                    "X-Matting-Enabled": str(result.matting_enabled).lower(),
                    "X-Matting-Model": result.matting_model,
                }
            )

        # Return the output file (FXS or NK)
        return FileResponse(
            result.output_path,
            media_type="application/octet-stream",
            filename=f"{label}{output_ext}",
            headers={
                "X-Frame-Count": str(result.frame_count),
                "X-Object-Count": str(result.object_count),
                "X-Propagation-Mode": result.propagation_mode,
                "X-Active-Model": seg_runtime["model_name"],
                "X-Matting-Enabled": str(result.matting_enabled).lower(),
                "X-Matting-Model": result.matting_model,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        # Cleanup will happen when temp files are garbage collected
        # In production, use proper cleanup
        pass
