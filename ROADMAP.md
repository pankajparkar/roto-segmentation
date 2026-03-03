# Roto-Segmentation Roadmap

## Vision
Build an industry-leading AI-powered rotoscoping tool that rivals commercial solutions like Runway, Rotobot, and Boris FX Silhouette, with seamless integration into professional VFX pipelines.

---

## Current State (v0.1.0)

### Completed Features
- [x] SAM2.1 integration (small, base_plus, large models)
- [x] Point-click object selection
- [x] Bounding box selection
- [x] Video frame scrubbing and capture
- [x] Mask preview with green overlay
- [x] Silhouette FXS export (Bezier shapes)
- [x] EXR matte sequence export
- [x] MPS (Apple Silicon) GPU acceleration
- [x] Smart mask selection algorithm

### Current Limitations
- Single point selection can be ambiguous
- No multi-point refinement
- No text-based selection
- No mask post-processing options
- No batch processing

### Workflow Coverage Analysis

| Step | Traditional (Manual) | Current Tool | Status |
|------|---------------------|--------------|--------|
| Import footage | Load into Nuke/Silhouette | Upload video | **Solved** |
| Analyze shot | Scrub, find key frames | Video scrubbing | **Solved** |
| Initial shape creation | Draw splines (30-60 min) | Click/Box AI | **Solved** |
| Refine edges | Adjust control points | Smart mask selection | **Partial** |
| Keyframe animation | Manual every 5-10 frames | SAM2 auto-tracking | **Solved** |
| Handle motion blur | Manual edge softening | Not implemented | **Not Solved** |
| Handle occlusions | Redraw shapes | Basic tracking | **Partial** |
| Fine edge work | Per-frame adjustments | No manual editing | **Not Solved** |
| Review passes | QC with supervisor | Preview overlay | **Partial** |
| Export | FXS/Nuke export | FXS + EXR | **Solved** |

**Current Coverage: ~65% of traditional roto workflow**

---

## Sprint Planning

### Sprint 1: Multi-Point Selection (P0)
**Duration: 3-4 hours | Impact: HIGH**

Allow multiple positive + negative points to refine selection.

#### Frontend Changes
```typescript
// video-roto.component.ts
positivePoints = signal<{x: number, y: number}[]>([]);
negativePoints = signal<{x: number, y: number}[]>([]);

onFrameClick(event: MouseEvent) {
  const {x, y} = this.getCanvasCoordinates(event);

  if (event.shiftKey) {
    // Negative point (exclude)
    this.negativePoints.update(pts => [...pts, {x, y}]);
  } else {
    // Positive point (include)
    this.positivePoints.update(pts => [...pts, {x, y}]);
  }
  this.requestMaskPreview();
}

requestMaskPreview() {
  const points = [
    ...this.positivePoints().map(p => [p.x, p.y, 1]),
    ...this.negativePoints().map(p => [p.x, p.y, 0])
  ];
  this.videoRotoService.getPreview(this.frameData, points).subscribe(...);
}
```

#### UI Changes
```html
<!-- Show point markers -->
@for (point of positivePoints(); track $index) {
  <div class="point-marker positive" [style.left.px]="point.x" [style.top.px]="point.y"></div>
}
@for (point of negativePoints(); track $index) {
  <div class="point-marker negative" [style.left.px]="point.x" [style.top.px]="point.y"></div>
}

<!-- Instructions -->
<p>Click = include (green) | Shift+Click = exclude (red)</p>
<span>{{ positivePoints().length }} include, {{ negativePoints().length }} exclude</span>
```

#### Backend (Already Supports!)
```python
# segment.py - existing code handles multiple points
point_coords = np.array([[p[0], p[1]] for p in points_data])
point_labels = np.array([p[2] if len(p) > 2 else 1 for p in points_data])
```

#### Files to Modify
- `frontend/src/app/features/video-roto/video-roto.component.ts`
- `frontend/src/app/features/video-roto/video-roto.component.html`
- `frontend/src/app/features/video-roto/video-roto.component.scss`
- `frontend/src/app/features/video-roto/video-roto.service.ts`

---

### Sprint 2: Edge Quality Improvements (P1)
**Duration: 4-6 hours | Impact: HIGH**

#### A. Post-Processing Pipeline
```python
# backend/src/roto_seg/utils/mask_processing.py

import cv2
import numpy as np

class MaskProcessor:
    @staticmethod
    def smooth_edges(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Gaussian blur for soft edges"""
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        return (blurred > 0.5).astype(np.uint8)

    @staticmethod
    def feather_edges(mask: np.ndarray, amount: int = 5) -> np.ndarray:
        """Feather edges with gradient falloff"""
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=amount)
        eroded = cv2.erode(mask, kernel, iterations=amount)

        # Create gradient between dilated and eroded
        dist_outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        feathered = np.clip(1 - (dist_outside / amount), 0, 1)
        feathered = np.where(mask, 1 - np.clip(dist_inside / amount, 0, 0.5), feathered)

        return feathered

    @staticmethod
    def expand_contract(mask: np.ndarray, pixels: int) -> np.ndarray:
        """Expand (positive) or contract (negative) mask"""
        kernel = np.ones((3, 3), np.uint8)
        if pixels > 0:
            return cv2.dilate(mask, kernel, iterations=pixels)
        elif pixels < 0:
            return cv2.erode(mask, kernel, iterations=abs(pixels))
        return mask

    @staticmethod
    def fill_holes(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
        """Fill small holes in mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Fill holes smaller than min_size
        for contour in contours:
            if cv2.contourArea(contour) < min_size:
                cv2.fillPoly(mask, [contour], 1)

        return mask
```

#### B. User Controls (Frontend)
```typescript
// Add to video-roto.component.ts
edgeSmoothing = signal<number>(0);      // 0-10
featherAmount = signal<number>(0);       // 0-20 pixels
expandContract = signal<number>(0);      // -10 to +10 pixels
fillHoles = signal<boolean>(false);
```

#### C. API Update
```python
# Add to segment.py
class MaskProcessingOptions(BaseModel):
    smooth_edges: bool = False
    smooth_kernel: int = 3
    feather_amount: int = 0
    expand_pixels: int = 0
    fill_holes: bool = False
    min_hole_size: int = 100

@router.post("/segment-image")
async def segment_image(
    image: UploadFile,
    points: Optional[str] = Form(None),
    box: Optional[str] = Form(None),
    processing: Optional[str] = Form(None),  # JSON MaskProcessingOptions
):
    # ... existing code ...

    # Apply post-processing
    if processing:
        opts = MaskProcessingOptions.parse_raw(processing)
        if opts.smooth_edges:
            best_mask = MaskProcessor.smooth_edges(best_mask, opts.smooth_kernel)
        if opts.feather_amount > 0:
            best_mask = MaskProcessor.feather_edges(best_mask, opts.feather_amount)
        # etc.
```

---

### Sprint 3: Manual Touch-Up Tools (P1)
**Duration: 1-2 days | Impact: HIGH**

#### A. In-App Mask Painting
```typescript
// video-roto.component.ts

brushMode = signal<'paint' | 'erase' | null>(null);
brushSize = signal<number>(20);
brushHardness = signal<number>(1.0);  // 0-1, soft to hard

// Drawing state
private isDrawing = false;
private paintCanvas: HTMLCanvasElement;
private paintCtx: CanvasRenderingContext2D;

initPaintCanvas() {
  this.paintCanvas = document.createElement('canvas');
  this.paintCanvas.width = this.frameCanvas.width;
  this.paintCanvas.height = this.frameCanvas.height;
  this.paintCtx = this.paintCanvas.getContext('2d')!;
}

onPaintStart(event: MouseEvent) {
  if (!this.brushMode()) return;
  this.isDrawing = true;
  this.paint(event);
}

onPaintMove(event: MouseEvent) {
  if (!this.isDrawing) return;
  this.paint(event);
}

onPaintEnd() {
  this.isDrawing = false;
  this.applyPaintToMask();
}

paint(event: MouseEvent) {
  const {x, y} = this.getCanvasCoordinates(event);
  const ctx = this.paintCtx;

  ctx.globalCompositeOperation = this.brushMode() === 'erase'
    ? 'destination-out'
    : 'source-over';

  ctx.beginPath();
  ctx.arc(x, y, this.brushSize() / 2, 0, Math.PI * 2);
  ctx.fillStyle = this.brushMode() === 'paint' ? 'white' : 'black';
  ctx.fill();
}
```

#### B. UI Controls
```html
<!-- Brush toolbar -->
<div class="brush-toolbar" *ngIf="isSelectMode()">
  <button [class.active]="brushMode() === 'paint'" (click)="setBrushMode('paint')">
    Paint (+)
  </button>
  <button [class.active]="brushMode() === 'erase'" (click)="setBrushMode('erase')">
    Erase (-)
  </button>
  <input type="range" min="5" max="100" [value]="brushSize()"
         (input)="brushSize.set($event.target.value)">
  <span>Size: {{ brushSize() }}px</span>
</div>
```

---

### Sprint 4: Occlusion Handling (P2)
**Duration: 1-2 days | Impact: MEDIUM**

#### A. Multi-Frame Initialization
```python
# backend/src/roto_seg/services/roto_pipeline.py

class MultiFramePrompt:
    """Allow prompts on multiple frames for better tracking through occlusions"""
    prompts: List[SegmentationPrompt]  # Each has frame_idx

def process_with_keyframes(self, video_path: str, prompts: List[SegmentationPrompt]):
    """
    Track with multiple keyframe prompts.

    Example:
        prompts = [
            SegmentationPrompt(frame_idx=0, points=...),    # Start
            SegmentationPrompt(frame_idx=50, points=...),   # After occlusion
        ]
    """
    # Sort prompts by frame
    prompts = sorted(prompts, key=lambda p: p.frame_idx)

    all_masks = {}

    for i, prompt in enumerate(prompts):
        # Track forward from this prompt
        start_frame = prompt.frame_idx
        end_frame = prompts[i+1].frame_idx if i+1 < len(prompts) else total_frames

        masks = self.track_range(video_path, prompt, start_frame, end_frame)
        all_masks.update(masks)

    return all_masks
```

#### B. Confidence Scoring & Re-Detection
```python
def track_with_recovery(self, video_path: str, initial_prompt: SegmentationPrompt):
    """Track with automatic recovery when tracking fails"""

    masks = {}
    confidence_threshold = 0.3

    for frame_idx, frame in enumerate(video_frames):
        mask, confidence = self.tracker.track_frame(frame)

        if confidence < confidence_threshold:
            # Tracking lost - try to re-detect
            logger.warning(f"Low confidence at frame {frame_idx}: {confidence}")

            # Option 1: Use Grounding DINO to re-detect
            if self.grounding_dino:
                new_box = self.grounding_dino.detect(frame, self.object_description)
                if new_box:
                    self.tracker.add_prompt(frame_idx, box=new_box)
                    mask, confidence = self.tracker.track_frame(frame)

            # Option 2: Interpolate if gap is small
            if confidence < confidence_threshold and frame_idx - last_good_frame < 10:
                mask = self.interpolate_mask(masks[last_good_frame], frame_idx - last_good_frame)

        masks[frame_idx] = mask
        if confidence >= confidence_threshold:
            last_good_frame = frame_idx

    return masks
```

#### C. Interpolation for Brief Occlusions
```python
def interpolate_masks(self, mask_before: np.ndarray, mask_after: np.ndarray,
                      num_frames: int) -> List[np.ndarray]:
    """Interpolate masks for brief occlusions (< 10 frames)"""

    # Find contours
    contours_before = cv2.findContours(mask_before, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_after = cv2.findContours(mask_after, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if not contours_before or not contours_after:
        return [mask_before] * num_frames

    # Get bounding boxes
    rect_before = cv2.boundingRect(contours_before[0])
    rect_after = cv2.boundingRect(contours_after[0])

    interpolated = []
    for i in range(num_frames):
        alpha = (i + 1) / (num_frames + 1)

        # Interpolate bounding box
        x = int(rect_before[0] + alpha * (rect_after[0] - rect_before[0]))
        y = int(rect_before[1] + alpha * (rect_after[1] - rect_before[1]))
        w = int(rect_before[2] + alpha * (rect_after[2] - rect_before[2]))
        h = int(rect_before[3] + alpha * (rect_after[3] - rect_before[3]))

        # Create interpolated mask (simple box for now)
        mask = np.zeros_like(mask_before)
        mask[y:y+h, x:x+w] = 1

        interpolated.append(mask)

    return interpolated
```

---

## Phase 2: Text-Based Selection (Grounding DINO)

### 2.1 Grounding DINO Integration
**Priority: HIGH**

Add natural language object selection.

```
Architecture:
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Text Prompt    │────▶│  Grounding DINO  │────▶│  Bounding   │
│  "white car"    │     │  (700MB model)   │     │  Box(es)    │
└─────────────────┘     └──────────────────┘     └─────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Final Mask     │◀────│     SAM2         │◀────│  Box Prompt │
│  (tracked)      │     │  Segmentation    │     │  to SAM2    │
└─────────────────┘     └──────────────────┘     └─────────────┘
```

**Installation:**
```bash
pip install groundingdino-py
# Model: GroundingDINO-T (~700MB)
```

**API Endpoint:**
```python
@router.post("/segment-by-text")
async def segment_by_text(
    image: UploadFile,
    text_prompt: str = Form(...),  # e.g., "the red car"
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
):
    # 1. Run Grounding DINO to get boxes
    # 2. If multiple boxes, return options to user
    # 3. Run SAM2 with selected box
    # 4. Return mask
```

**UI Changes:**
- Add text input field above frame
- Show detected objects as clickable options
- Highlight selected object
- Allow combining text + point refinement

### 2.2 Multi-Object Detection
**Priority: MEDIUM**

When text matches multiple objects:
```
User types: "car"
System finds: 3 cars in frame
UI shows: Numbered boxes (1, 2, 3)
User clicks: Box #2
System: Segments car #2
```

### 2.3 Semantic Understanding
**Priority: LOW**

Support complex queries:
- "the person on the left"
- "the larger of the two dogs"
- "everything except the background"

---

## Phase 3: Advanced Mask Processing

### 3.1 Alpha Matting (for hair/fine edges)
**Priority: MEDIUM**

```python
# Integration with ViTMatte or MODNet
class AlphaMatting:
    def __init__(self):
        self.model = load_vitmatte_model()

    def refine_mask(self, image: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
        """
        Input: RGB image + binary mask from SAM2
        Output: Soft alpha matte with fine edge details
        """
        # Generate trimap from coarse mask
        trimap = self.generate_trimap(coarse_mask)

        # Run matting model
        alpha = self.model.predict(image, trimap)

        return alpha  # Float 0-1
```

### 3.2 Morphological Operations
**Priority: MEDIUM**

```python
class MaskProcessingOptions(BaseModel):
    smooth_edges: bool = False
    smooth_kernel: int = 3
    feather_amount: int = 0
    expand_pixels: int = 0
    contract_pixels: int = 0
    fill_holes: bool = False
    min_hole_size: int = 100
    remove_small_regions: bool = False
    min_region_size: int = 50
```

---

## Phase 4: Performance & Scalability

### 4.1 Model Caching
**Priority: HIGH**

Keep models loaded in memory:
- Lazy load on first request
- Keep warm for subsequent requests
- Configurable memory limits

### 4.2 Batch Processing
**Priority: MEDIUM**

Process multiple objects/videos:
```
API: POST /api/v1/batch/quick-roto
{
  "videos": ["video1.mp4", "video2.mp4"],
  "prompts": [
    {"text": "car", "frame": 0},
    {"text": "person", "frame": 0}
  ]
}
```

### 4.3 GPU Memory Optimization
**Priority: MEDIUM**

- Dynamic model selection based on available VRAM
- Automatic fallback to smaller models
- Frame-by-frame processing for large videos

### 4.4 Progress Streaming
**Priority: LOW**

WebSocket for real-time progress:
```javascript
ws.onmessage = (event) => {
  const { frame, total, mask_preview } = JSON.parse(event.data);
  updateProgress(frame / total * 100);
  showPreview(mask_preview);
};
```

---

## Phase 5: Export & Integration

### 5.1 Additional Export Formats
**Priority: MEDIUM**

- **PNG Sequence**: Standard matte sequence
- **ProRes 4444**: Video with alpha channel
- **After Effects**: .jsx script with masks
- **Fusion**: .comp file with masks
- **Resolve**: .drp compatible format

### 5.2 Nuke Integration
**Priority: HIGH**

Better Nuke support:
- Roto node with animated shapes
- RotoPaint compatible output
- Gizmo for direct import

### 5.3 Round-Trip Workflow
**Priority: MEDIUM**

```
App → FXS export → Silhouette edit → Re-import FXS → Continue in app

Implementation:
├── "Export for Editing" button
├── Artist refines in Silhouette
├── "Import Corrections" button
├── Merge artist edits with AI tracking
└── Continue processing
```

### 5.4 Plugin Architecture
**Priority: LOW**

Build plugins for:
- Nuke (Python)
- After Effects (CEP)
- DaVinci Resolve (Fuse)
- Silhouette (Python)

---

## Phase 6: Advanced AI Features

### 6.1 Auto-Segmentation Mode
**Priority: MEDIUM**

Automatically detect and segment all objects:
```
1. Run Grounding DINO with "all objects"
2. Run SAM2 on each detected box
3. Present list of segmented objects
4. User selects which to track
```

### 6.2 Object Tracking Improvements
**Priority: HIGH**

- Handle occlusions better
- Re-detect when object reappears
- Interpolate through brief occlusions
- Handle scale/rotation changes

### 6.3 Instance Segmentation
**Priority: LOW**

Track multiple instances of same class:
- "Track all 3 cars independently"
- Separate mask for each instance
- Handle crossing/overlapping paths

### 6.4 Depth-Aware Segmentation
**Priority: LOW**

Use depth estimation to improve masks:
- Separate foreground/background by depth
- Handle transparent/semi-transparent objects
- Better edge detection at depth boundaries

---

## Phase 7: User Experience

### 7.1 Project Management
**Priority: MEDIUM**

- Save/load projects
- Session history
- Undo/redo for all operations
- Auto-save

### 7.2 Keyboard Shortcuts
**Priority: MEDIUM**

```
Space     - Play/Pause video
,/.       - Step frame back/forward
P         - Point selection mode
B         - Box selection mode
T         - Text selection mode
Enter     - Process video
Escape    - Cancel/Clear
Ctrl+Z    - Undo
Ctrl+Y    - Redo
[ / ]     - Decrease/Increase brush size
```

### 7.3 Preview Improvements
**Priority: MEDIUM**

- Side-by-side original/mask view
- Onion skinning for motion
- Mask opacity slider
- Different overlay colors

### 7.4 Dark/Light Theme
**Priority: LOW**

- System preference detection
- Manual toggle
- VFX-friendly dark theme (default)

---

## Priority Summary

| Feature | Impact | Effort | Priority | Sprint |
|---------|--------|--------|----------|--------|
| Multi-point selection | High | Low (3-4h) | **P0** | 1 |
| Edge post-processing | High | Medium (4-6h) | **P1** | 2 |
| Manual painting tools | High | Medium (1-2d) | **P1** | 3 |
| Occlusion recovery | Medium | High (1-2d) | **P2** | 4 |
| Grounding DINO | High | High (2-3d) | **P2** | 5 |
| Alpha matting | Medium | High (2-3d) | **P3** | 6 |
| Round-trip workflow | Low | Medium (1d) | **P3** | 7 |

---

## Technical Debt & Maintenance

### Code Quality
- [ ] Add comprehensive unit tests
- [ ] Add integration tests for API
- [ ] Set up CI/CD pipeline
- [ ] Add type hints throughout Python code
- [ ] Document all API endpoints (OpenAPI)

### Performance Monitoring
- [ ] Add logging infrastructure
- [ ] Performance metrics (response times)
- [ ] Error tracking (Sentry or similar)
- [ ] Usage analytics (privacy-respecting)

### Security
- [ ] Rate limiting on API
- [ ] File upload validation
- [ ] Input sanitization
- [ ] CORS configuration

---

## Model Comparison Reference

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| SAM2.1 Small | 176MB | Fast | Good | Quick previews, drafts |
| SAM2.1 Base+ | 309MB | Medium | Better | General use |
| SAM2.1 Large | 856MB | Slow | Best | Final output, complex objects |
| Grounding DINO | 700MB | Medium | N/A | Text-to-box detection |
| ViTMatte | ~400MB | Medium | N/A | Alpha matting for fine edges |

---

## Competitive Analysis

| Feature | Our Tool | Runway | Rotobot | Boris FX |
|---------|----------|--------|---------|----------|
| AI Segmentation | SAM2 | Proprietary | Proprietary | ML Edge |
| Text Selection | Planned | Yes | No | No |
| Point Selection | Yes | Yes | Yes | Yes |
| Box Selection | Yes | Yes | Yes | Yes |
| Multi-Point | Planned | Yes | Yes | Yes |
| Local Processing | Yes | Cloud | Cloud | Yes |
| FXS Export | Yes | No | No | Yes |
| Nuke Export | Planned | No | Yes | Yes |
| Price | Free/OSS | $15-76/mo | Custom | $999+ |

---

## Success Metrics

### Sprint 1 Success
- [ ] 90%+ mask accuracy with multi-point selection
- [ ] <2s preview generation time

### Sprint 2 Success
- [ ] Professional-quality edges (no jaggies)
- [ ] User-adjustable smoothing controls

### Sprint 3 Success
- [ ] Paint/erase brush working smoothly
- [ ] Real-time preview of edits

### Sprint 4 Success
- [ ] 80%+ tracking success through brief occlusions
- [ ] Automatic re-detection working

### Overall Success
- [ ] Used in at least 1 commercial production
- [ ] 100+ GitHub stars
- [ ] Community contributions

---

## Future Work: Industry Gap Analysis

Based on competitive analysis of industry leaders (Boris FX Silhouette, Mocha Pro, Runway ML, DaVinci Resolve, Wonder Dynamics, After Effects), the following features represent opportunities to reach feature parity with professional tools.

### Quick Wins (1-2 Days Each)

| Feature | Reference Tool | Description | Effort |
|---------|---------------|-------------|--------|
| ProRes 4444 Export | All Pro Tools | Video with embedded alpha channel via FFmpeg | Easy |
| Confidence Score Display | DaVinci Magic Mask | Show tracking confidence per frame | Easy |
| MultiPoly View | DaVinci Resolve 20 | UI to manage/view all masks in single list | Easy |
| Area Brush Refinement | Mocha Pro 2025 | Quick brush strokes to add/remove mask regions | Easy |
| Mask Invert/Combine | All Pro Tools | Boolean operations on masks | Easy |

### Edge & Matte Quality (Medium Term)

| Feature | Reference Tool | Description | Priority |
|---------|---------------|-------------|----------|
| **Matte Refine ML** | Silhouette 2025 | AI edge refinement for hair, fur, feathers with semi-transparency | HIGH |
| **Face ML / Face Detect** | Silhouette 2025.5, Mocha Pro 2025.5 | Auto-detect facial regions (eyes, lips, skin) for beauty work | HIGH |
| **Motion Blur Handling** | DaVinci Magic Mask v2 | Track through motion blur and lighting changes | MEDIUM |
| **Depth Map ML** | Silhouette 2025 | AI depth estimation for depth-based matte controls | MEDIUM |

**Implementation Notes:**
```python
# Face Detection - Use MediaPipe or dlib
import mediapipe as mp
face_mesh = mp.solutions.face_mesh.FaceMesh()

# Matte Refine - Integrate ViTMatte
# Model: ~400MB, generates soft alpha from coarse mask
```

### Advanced Tracking (Medium Term)

| Feature | Reference Tool | Description | Priority |
|---------|---------------|-------------|----------|
| **PowerMesh/Sub-planar Tracking** | Mocha Pro | Track warped surfaces & organic deformations | HIGH |
| **Unified Detection Mode** | DaVinci Magic Mask v2 | Single click auto-detects person/object/region type | MEDIUM |
| **Re-detection on Track Loss** | Mocha Pro 2025.5 | Auto-regenerate mask when tracking fails | MEDIUM |
| **Object Brush (Single Click)** | Mocha Pro 2025 | One click → editable splines → auto-track | MEDIUM |

**Key Insight from DaVinci Magic Mask v2:**
> "Previous AI tracking required separate modes for people versus objects. What changed: Single unified mode. Click once anywhere on subject. AI identifies whether it's person, object, or region."

### Professional Export Formats (Medium Term)

| Feature | Reference Tool | Description | Priority |
|---------|---------------|-------------|----------|
| **Cryptomatte Export** | Silhouette 2025.5 | Industry-standard multi-object matte with motion blur | HIGH |
| **Editable Splines Export** | Mocha Pro | AI generates vector splines that remain editable | HIGH |
| **USD/Alembic/FBX Export** | Mocha Pro, Wonder Dynamics | 3D scene/tracking data for Maya, Blender, Unreal | MEDIUM |
| **PNG Sequence with Metadata** | All Pro Tools | Frame numbers, timecode in filenames | LOW |

**Cryptomatte Technical Requirements:**
- 32-bit EXR format required
- Three types: crypto_asset, crypto_material, crypto_object
- Supports motion blur and depth of field
- Direct integration with Nuke, Fusion, Resolve

### AI Selection Modes (Medium Term)

| Feature | Reference Tool | Description | Priority |
|---------|---------------|-------------|----------|
| **Natural Language Prompts** | Silhouette 2025.5, Runway | "Select the red car" text-to-mask | HIGH |
| **Multi-Instance Detection** | Rotobot | "All 3 cars" → separate masks for each | MEDIUM |
| **Semantic Understanding** | Runway Aleph | "The person on the left", "everything except background" | LOW |

**Grounding DINO + SAM2 Pipeline:**
```
Text Prompt → Grounding DINO → Bounding Box(es) → SAM2 → Tracked Mask
"white car"     (700MB model)    [x1,y1,x2,y2]     segment    final output
```

### 3D & Scene Understanding (Long Term)

| Feature | Reference Tool | Description | Priority |
|---------|---------------|-------------|----------|
| **3D Scene Node** | Silhouette 2025 | Import Alembic/FBX, place cards in 3D space | LOW |
| **Camera Solve Integration** | Silhouette 2025.5 (SynthEyes) | Built-in camera tracking | LOW |
| **Video-to-3D Scene** | Wonder Dynamics/Autodesk Flow | Reconstruct 3D environment from video | VERY LOW |
| **Markerless MoCap** | Wonder Dynamics | Face/body/hand motion capture from single camera | VERY LOW |

### Repair & Restoration (Long Term)

| Feature | Reference Tool | Description | Priority |
|---------|---------------|-------------|----------|
| **Frame Fixer ML** | Silhouette 2025 | Remove photo flashes, artifacts, dropped frames | LOW |
| **Clean Plate Generation** | Wonder Dynamics | Auto-generate background without subject | LOW |
| **In-Video Object Add/Remove** | Runway Aleph | Text prompts to modify video post-generation | VERY LOW |

### Workflow & UX Improvements

| Feature | Reference Tool | Description | Priority |
|---------|---------------|-------------|----------|
| **AI + Manual Hybrid Mode** | DaVinci Resolve | Combine AI mask with manual spline refinement | HIGH |
| **Compound Nodes** | Silhouette 2025 | Merge multiple operations into single node | MEDIUM |
| **Real-time Preview** | Runway | Instant mask preview while editing | MEDIUM |
| **Stabilized Rotoscoping** | DaVinci Fusion | Lock footage, paint, restore motion | MEDIUM |

---

## Competitive Positioning

### Our Differentiators

| Advantage | Industry Pain Point We Solve |
|-----------|------------------------------|
| **100% Local Processing** | Runway/Wonder are cloud-only; privacy concerns for studios |
| **Open Source / Free** | Pro tools cost $1,000-2,200+/year |
| **Native FXS Export** | Direct Silhouette integration (no round-trip) |
| **SAM2.1 (State-of-Art)** | Others use older/proprietary models |
| **Apple Silicon Optimized** | MPS acceleration for M1/M2/M3 Macs |

### Feature Parity Target

| Tool | Their Strength | Our Response |
|------|---------------|--------------|
| **Runway ML** | Text prompts, real-time preview | Grounding DINO integration (Phase 2) |
| **Silhouette** | Matte Refine ML, Face ML | ViTMatte + MediaPipe (Phase 3) |
| **Mocha Pro** | PowerMesh, editable splines | Sub-planar tracking research (Phase 5) |
| **DaVinci** | Unified detection, Magic Mask v2 | Auto-detect object type (Phase 4) |
| **Wonder Dynamics** | Video-to-3D scene | Out of scope (different product category) |

### Pricing Comparison

| Tool | Pricing | Our Position |
|------|---------|--------------|
| Runway ML | $15-76/month | Free & Open Source |
| Boris FX Silhouette | $2,195 perpetual / $875/year | Free & Open Source |
| Mocha Pro | $995 perpetual / $495/year | Free & Open Source |
| Wonder Dynamics | ~$1,000/year | Free & Open Source |
| DaVinci Resolve Studio | $295 one-time | Free & Open Source |

---

## Implementation Roadmap (Extended)

### v0.2.0 - Core Improvements
- [ ] Multi-point selection (Sprint 1)
- [ ] Edge post-processing controls (Sprint 2)
- [ ] Manual paint/erase tools (Sprint 3)
- [ ] Occlusion recovery (Sprint 4)

### v0.3.0 - AI Selection
- [ ] Grounding DINO text-to-box integration
- [ ] Multi-instance detection ("all cars")
- [ ] Natural language prompts UI

### v0.4.0 - Professional Export
- [ ] Cryptomatte EXR export
- [ ] ProRes 4444 video export
- [ ] Editable spline export (bezier control points)
- [ ] PNG sequence with frame metadata

### v0.5.0 - Edge Quality Pro
- [ ] ViTMatte integration for hair/fur
- [ ] Face ML auto-detection
- [ ] Depth map generation
- [ ] Motion blur-aware tracking

### v1.0.0 - Production Ready
- [ ] AI + Manual hybrid mode
- [ ] Real-time preview
- [ ] Confidence-based re-detection
- [ ] Full Nuke/Fusion plugin support

---

## Resources & References

### Models
- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [LangSAM](https://github.com/luca-medeiros/lang-segment-anything)
- [ViTMatte](https://github.com/hustvl/ViTMatte)

### Competitors
- [Runway ML](https://runwayml.com/)
- [Rotobot](https://www.kognat.com/rotobot/)
- [Boris FX Silhouette](https://borisfx.com/products/silhouette/)
- [Wonder Studio](https://wonderdynamics.com/)

### Papers
- "Segment Anything" (Meta AI, 2023)
- "Grounding DINO" (IDEA Research, 2023)
- "Track Anything" (2023)
- "ViTMatte: Boosting Image Matting with Pretrained ViT" (2023)

---

*Last Updated: March 2026*
*Version: 0.1.0*
