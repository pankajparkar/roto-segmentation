"""
Shot management and segmentation endpoints.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter()


class ShotResponse(BaseModel):
    """Response model for a shot."""

    id: UUID
    project_id: UUID
    name: str
    status: str  # pending, processing, completed, failed
    frame_start: int
    frame_end: int
    width: int
    height: int
    fps: float


class SegmentRequest(BaseModel):
    """Request model for starting segmentation."""

    prompt_type: str  # point, box, text
    prompt_data: dict  # { "points": [[x,y]], "labels": [1] } or { "box": [x1,y1,x2,y2] }
    propagate: bool = True
    matting: bool = False


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    job_id: UUID
    shot_id: UUID
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 - 1.0
    current_frame: Optional[int] = None
    message: Optional[str] = None


# In-memory store for now
_shots: dict = {}
_jobs: dict = {}


@router.post("/upload")
async def upload_shot(
    project_id: UUID = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a video or image sequence for a shot."""
    import uuid

    # TODO: Actually process the uploaded file
    shot_id = uuid.uuid4()
    _shots[shot_id] = {
        "id": shot_id,
        "project_id": project_id,
        "name": name,
        "status": "pending",
        "frame_start": 1001,
        "frame_end": 1100,
        "width": 1920,
        "height": 1080,
        "fps": 24.0,
    }

    return _shots[shot_id]


@router.get("/{shot_id}", response_model=ShotResponse)
async def get_shot(shot_id: UUID):
    """Get shot details."""
    if shot_id not in _shots:
        raise HTTPException(status_code=404, detail="Shot not found")
    return _shots[shot_id]


@router.post("/{shot_id}/segment")
async def start_segmentation(shot_id: UUID, request: SegmentRequest):
    """Start AI segmentation for a shot."""
    import uuid

    if shot_id not in _shots:
        raise HTTPException(status_code=404, detail="Shot not found")

    job_id = uuid.uuid4()
    _jobs[job_id] = {
        "job_id": job_id,
        "shot_id": shot_id,
        "status": "pending",
        "progress": 0.0,
        "current_frame": None,
        "message": "Queued for processing",
    }

    # TODO: Submit to Celery queue
    # task = segment_shot.delay(shot_id, request.dict())
    # _jobs[job_id]["celery_task_id"] = task.id

    return _jobs[job_id]


@router.get("/{shot_id}/jobs", response_model=List[JobStatusResponse])
async def list_shot_jobs(shot_id: UUID):
    """List all jobs for a shot."""
    return [job for job in _jobs.values() if job["shot_id"] == shot_id]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: UUID):
    """Get job status."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.get("/{shot_id}/masks")
async def get_shot_masks(shot_id: UUID, frame: Optional[int] = None):
    """Get mask data for a shot."""
    if shot_id not in _shots:
        raise HTTPException(status_code=404, detail="Shot not found")

    # TODO: Return actual mask data
    return {
        "shot_id": shot_id,
        "objects": [
            {
                "id": 1,
                "label": "person_01",
                "frames": [1001, 1002, 1003],  # Frames with masks
            }
        ],
    }
