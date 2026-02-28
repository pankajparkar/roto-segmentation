"""
Project management endpoints.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ProjectCreate(BaseModel):
    """Request model for creating a project."""

    name: str
    description: Optional[str] = None
    client: Optional[str] = None


class ProjectResponse(BaseModel):
    """Response model for a project."""

    id: UUID
    name: str
    description: Optional[str]
    client: Optional[str]
    shot_count: int = 0


# In-memory store for now (replace with database)
_projects: dict = {}


@router.post("/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """Create a new project."""
    import uuid

    project_id = uuid.uuid4()
    _projects[project_id] = {
        "id": project_id,
        "name": project.name,
        "description": project.description,
        "client": project.client,
        "shot_count": 0,
    }
    return _projects[project_id]


@router.get("/", response_model=List[ProjectResponse])
async def list_projects():
    """List all projects."""
    return list(_projects.values())


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: UUID):
    """Get a specific project."""
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return _projects[project_id]


@router.delete("/{project_id}")
async def delete_project(project_id: UUID):
    """Delete a project."""
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    del _projects[project_id]
    return {"status": "deleted"}
