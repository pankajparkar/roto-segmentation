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


@router.get("/device-info", response_model=DeviceInfoResponse)
async def get_device_info():
    """Get information about available compute devices."""
    from roto_seg.ai.device import get_device_info

    info = get_device_info()
    return DeviceInfoResponse(
        platform=info["platform"],
        torch_version=info["torch_version"],
        recommended_device=info["recommended"],
        mps_available=info["devices"]["mps"]["available"],
        cuda_available=info["devices"]["cuda"]["available"],
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

    # Get best mask
    best_idx = scores.argmax()
    best_mask = masks[best_idx]

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
    click_x: int = Form(...),
    click_y: int = Form(...),
    frame_idx: int = Form(0),
    label: str = Form("object"),
    output_format: str = Form("silhouette"),
):
    """
    Quick rotoscoping with a single click.

    Upload a video, click on an object, get an FXS file.

    This is a synchronous endpoint for small videos.
    For large videos, use the async job API.

    Example:
        curl -X POST "http://localhost:8000/api/v1/segment/quick-roto" \
            -F "video=@clip.mp4" \
            -F "click_x=500" \
            -F "click_y=300" \
            -F "frame_idx=0" \
            -F "label=person"
    """
    import logging
    logger = logging.getLogger(__name__)

    # Debug log the received coordinates
    logger.info(f"Quick Roto Request: click_x={click_x}, click_y={click_y}, frame_idx={frame_idx}, label={label}, format={output_format}")
    print(f"[DEBUG] Quick Roto: click=({click_x}, {click_y}), frame={frame_idx}, label={label}, format={output_format}")

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
        result = pipeline.process(
            video_path=str(video_path),
            prompts=[
                SegmentationPrompt(
                    frame_idx=frame_idx,
                    points=np.array([[click_x, click_y]]),
                    point_labels=np.array([1]),
                    label=label,
                )
            ],
            output_path=str(output_path),
            output_format=output_format,
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
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        # Cleanup will happen when temp files are garbage collected
        # In production, use proper cleanup
        pass
