# Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI ROTO SYSTEM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    FRONTEND     │     │    API LAYER    │     │   PROCESSING    │
│                 │     │                 │     │                 │
│  - Web UI       │────▶│  - REST API     │────▶│  - Job Queue    │
│  - Desktop App  │     │  - WebSocket    │     │  - GPU Workers  │
│  - Plugin SDK   │     │  - Auth/RBAC    │     │  - Storage      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────┐
                        │                               │               │
                        ▼                               ▼               ▼
                 ┌─────────────┐               ┌─────────────┐  ┌─────────────┐
                 │  AI MODELS  │               │   STORAGE   │  │  DATABASE   │
                 │             │               │             │  │             │
                 │  - SAM 2    │               │  - Frames   │  │  - Jobs     │
                 │  - XMem     │               │  - Masks    │  │  - Projects │
                 │  - ViTMatte │               │  - Exports  │  │  - Users    │
                 └─────────────┘               └─────────────┘  └─────────────┘
```

---

## Component Details

### 1. Frontend Layer

#### Web Application
- **Framework**: React 18+ with TypeScript
- **State Management**: Zustand or Redux Toolkit
- **Canvas Library**: Fabric.js or Konva.js for mask visualization
- **Video Player**: Custom HTML5 video with frame-accurate seeking

#### Desktop Application (Optional)
- **Framework**: Electron or Tauri
- **Benefits**: Local GPU access, offline mode, better file handling

#### Plugin SDK
- Nuke Python plugin
- After Effects CEP/UXP panel
- Silhouette Python integration

### 2. API Layer

#### REST API (FastAPI)
```
POST   /api/v1/projects                    # Create project
POST   /api/v1/projects/{id}/shots         # Upload shot
POST   /api/v1/shots/{id}/segment          # Start AI segmentation
GET    /api/v1/shots/{id}/masks            # Get mask data
POST   /api/v1/shots/{id}/export           # Export to format
GET    /api/v1/jobs/{id}/status            # Job status
```

#### WebSocket (Real-time)
- Job progress updates
- Frame-by-frame preview streaming
- Collaborative editing (future)

### 3. Processing Layer

#### Job Queue System
```
┌─────────────────────────────────────────────────────────────┐
│                      JOB QUEUE (Redis)                       │
├─────────────────────────────────────────────────────────────┤
│  HIGH PRIORITY    │  NORMAL PRIORITY   │  LOW PRIORITY      │
│  - User-initiated │  - Batch jobs      │  - Background      │
│  - Interactive    │  - Standard roto   │  - Optimization    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Worker 1 │   │ Worker 2 │   │ Worker 3 │
        │ (GPU 0)  │   │ (GPU 1)  │   │ (GPU 2)  │
        └──────────┘   └──────────┘   └──────────┘
```

#### Worker Types
| Worker | Model | GPU Memory | Purpose |
|--------|-------|------------|---------|
| Segmentation | SAM 2 | 8-16 GB | Initial mask generation |
| Propagation | XMem/Cutie | 8-12 GB | Temporal consistency |
| Matting | ViTMatte | 6-10 GB | Alpha refinement |
| Export | CPU | N/A | Format conversion |

### 4. AI Model Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI INFERENCE PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

INPUT: Video Frame + User Prompt (click/box/text)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: DETECTION (Optional)                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Grounding DINO                                              ││
│  │  - Text prompt → Bounding boxes                              ││
│  │  - "the person in red shirt" → [x1,y1,x2,y2]                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: SEGMENTATION                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  SAM 2 (Segment Anything Model 2)                            ││
│  │  - Point/Box/Mask prompt → Binary mask                       ││
│  │  - Handles first frame or keyframes                          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: PROPAGATION                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  XMem / Cutie                                                ││
│  │  - Propagate mask across frames                              ││
│  │  - Maintain temporal consistency                             ││
│  │  - Handle occlusions and reappearance                        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: MATTING (Optional - for fine detail)                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  ViTMatte / Matte Anything                                   ││
│  │  - Binary mask → Alpha matte                                 ││
│  │  - Handles hair, fur, transparency                           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: POST-PROCESSING                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  - Contour extraction (OpenCV)                               ││
│  │  - Bezier curve fitting                                      ││
│  │  - Temporal smoothing                                        ││
│  │  - Edge refinement                                           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
OUTPUT: Bezier Splines per frame → Export to FXS/NK/etc.
```

### 5. Storage Architecture

#### Frame Storage
```
/storage
├── /projects
│   └── /{project_id}
│       └── /{shot_id}
│           ├── /source          # Original frames (EXR/PNG)
│           ├── /masks           # AI-generated binary masks
│           ├── /mattes          # Alpha mattes (if matting enabled)
│           ├── /shapes          # Bezier shape data (JSON)
│           └── /exports         # Final exports (FXS/NK/etc.)
```

#### Database Schema (PostgreSQL)
```sql
-- Core entities
projects (id, name, client, created_at, settings)
shots (id, project_id, name, frame_range, resolution, status)
objects (id, shot_id, label, color, created_by)

-- AI data
segmentation_jobs (id, shot_id, status, model, params, started_at)
masks (id, object_id, frame, mask_path, confidence)
shapes (id, object_id, frame, bezier_data)

-- Export
exports (id, shot_id, format, file_path, created_at)
```

---

## Scalability Considerations

### Horizontal Scaling
```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └─────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  API Pod 1  │     │  API Pod 2  │     │  API Pod 3  │
└─────────────┘     └─────────────┘     └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    ┌─────────────────┐
                    │  Redis Cluster  │
                    └─────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ GPU Node 1  │     │ GPU Node 2  │     │ GPU Node N  │
│ (4x A100)   │     │ (4x A100)   │     │ (4x A100)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Performance Targets
| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Single frame segmentation | < 500ms | Interactive use |
| Video propagation | 10-20 fps | Batch processing |
| Export generation | < 5s per 100 frames | Background job |
| UI responsiveness | < 100ms | All interactions |

---

## Security & Access Control

### Authentication
- JWT-based authentication
- OAuth2 integration (Google, Microsoft)
- API key support for automation

### Authorization (RBAC)
| Role | Permissions |
|------|-------------|
| Admin | Full system access |
| Supervisor | Project management, QC approval |
| Artist | View/edit assigned shots |
| Viewer | Read-only access |

### Data Security
- Encryption at rest (AES-256)
- TLS 1.3 for transit
- Isolated project storage
- Audit logging

---

## Integration Points

### Input Formats
- Video: MP4, MOV, MXF, ProRes
- Image sequences: EXR, DPX, PNG, TIFF
- Projects: EDL, XML, AAF (for shot detection)

### Output Formats
- **Silhouette**: .fxs (native Bezier shapes)
- **Nuke**: .nk (Roto/RotoPaint nodes)
- **After Effects**: .json (Bodymovin format) or .jsx (ExtendScript)
- **Fusion**: .comp (Fusion composition)
- **Generic**: PNG/EXR sequences, JSON shape data
