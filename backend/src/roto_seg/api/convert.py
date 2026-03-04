"""
Video conversion API endpoints.

Provides MOV to MP4 conversion without quality loss using ffmpeg.
"""

import tempfile
import subprocess
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

router = APIRouter()


@router.post("/mov-to-mp4")
async def convert_mov_to_mp4(
    video: UploadFile = File(...),
    quality: str = Form("high"),  # "lossless", "high", "medium"
):
    """
    Convert MOV to MP4 without quality degradation.

    Args:
        video: The MOV file to convert
        quality: Conversion quality preset
            - "lossless": CRF 0, largest file size
            - "high": CRF 17, visually lossless (default)
            - "medium": CRF 23, good quality, smaller file

    Returns:
        The converted MP4 file as download

    Example:
        curl -X POST "http://localhost:8000/api/v1/convert/mov-to-mp4" \
            -F "video=@input.mov" \
            -F "quality=high" \
            --output output.mp4
    """
    # Validate file extension
    filename = video.filename or "video.mov"
    if not filename.lower().endswith(('.mov', '.MOV')):
        raise HTTPException(
            status_code=400,
            detail="File must be a MOV file"
        )

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    input_path = temp_dir / filename
    output_filename = Path(filename).stem + ".mp4"
    output_path = temp_dir / output_filename

    try:
        # Save uploaded file
        with open(input_path, 'wb') as f:
            shutil.copyfileobj(video.file, f)

        # Build ffmpeg command based on quality
        if quality == "lossless":
            # Mathematically lossless
            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-c:v", "libx264",
                "-crf", "0",
                "-preset", "veryslow",
                "-c:a", "aac",
                "-b:a", "320k",
                "-y",
                str(output_path)
            ]
        elif quality == "medium":
            # Good quality, smaller file
            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "medium",
                "-c:a", "aac",
                "-b:a", "192k",
                "-y",
                str(output_path)
            ]
        else:  # high (default)
            # Visually lossless
            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-c:v", "libx264",
                "-crf", "17",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "256k",
                "-y",
                str(output_path)
            ]

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg conversion failed: {result.stderr}"
            )

        if not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Conversion completed but output file not found"
            )

        # Get file sizes for header info
        input_size = input_path.stat().st_size
        output_size = output_path.stat().st_size

        return FileResponse(
            str(output_path),
            media_type="video/mp4",
            filename=output_filename,
            headers={
                "X-Original-Size": str(input_size),
                "X-Converted-Size": str(output_size),
                "X-Quality-Preset": quality,
            }
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Conversion timed out (>10 minutes)"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}"
        )


@router.post("/stream-copy")
async def stream_copy_to_mp4(
    video: UploadFile = File(...),
):
    """
    Convert MOV to MP4 using stream copy (fastest, no re-encoding).

    This simply repackages the video streams into MP4 container.
    Only works if the MOV codecs are MP4-compatible (most are).

    Example:
        curl -X POST "http://localhost:8000/api/v1/convert/stream-copy" \
            -F "video=@input.mov" \
            --output output.mp4
    """
    filename = video.filename or "video.mov"

    temp_dir = Path(tempfile.mkdtemp())
    input_path = temp_dir / filename
    output_filename = Path(filename).stem + ".mp4"
    output_path = temp_dir / output_filename

    try:
        # Save uploaded file
        with open(input_path, 'wb') as f:
            shutil.copyfileobj(video.file, f)

        # Stream copy - no re-encoding
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-c", "copy",
            "-y",
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Stream copy failed (codec may not be MP4 compatible): {result.stderr}"
            )

        input_size = input_path.stat().st_size
        output_size = output_path.stat().st_size

        return FileResponse(
            str(output_path),
            media_type="video/mp4",
            filename=output_filename,
            headers={
                "X-Original-Size": str(input_size),
                "X-Converted-Size": str(output_size),
                "X-Method": "stream-copy",
            }
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Conversion timed out"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}"
        )
