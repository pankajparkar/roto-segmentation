# Annotation Feature for Rotoscoping

This document describes the annotation system implemented for refining AI-generated segmentation masks, following industry-standard rotoscoping workflows.

## Overview

The annotation feature allows users to iteratively refine segmentation masks by adding or removing regions. This mirrors the workflow used in professional rotoscoping tools like Silhouette, Nuke Roto, and After Effects.

## Industry-Standard Controls

The annotation system follows standard conventions used across VFX software:

| Action | Control | Description |
|--------|---------|-------------|
| **Add to Selection** | Left Click | Adds a foreground point (label=1) to include area in mask |
| **Remove from Selection** | Alt + Click | Adds a background point (label=0) to exclude area from mask |
| **Remove from Selection** | Right Click | Alternative to Alt+Click for removing areas |
| **Undo Last Point** | Ctrl/Cmd + Z | Removes the most recent annotation point |
| **Clear All Points** | Clear All button | Removes all annotation points and resets the selection |

## How It Works

### 1. Point-Based Refinement

When you click on the frame, the system:

1. Records the click position in video coordinates
2. Determines if it's an "add" (foreground) or "remove" (background) point based on modifier keys
3. Sends all accumulated points to SAM2 for segmentation
4. Displays the updated mask preview in real-time

### 2. Visual Feedback

- **Green markers (+)**: Indicate "add" points that include areas in the mask
- **Red markers (-)**: Indicate "remove" points that exclude areas from the mask
- **Numbered markers**: Show the order in which points were added
- **Real-time mask preview**: Updates after each annotation to show the current selection

### 3. SAM2 Multi-Point Segmentation

The annotation system leverages SAM2's ability to process multiple prompts:

```python
# Backend receives points with labels
points = np.array([[x1, y1], [x2, y2], [x3, y3]])  # coordinates
labels = np.array([1, 1, 0])  # 1=foreground, 0=background

# SAM2 uses all points to generate refined mask
masks, scores, logits = predictor.predict(
    point_coords=points,
    point_labels=labels,
    multimask_output=True
)
```

## User Interface

### Annotation Toolbar

When in Point mode, an annotation toolbar appears showing:

- **Add count**: Number of foreground (add) points
- **Remove count**: Number of background (remove) points
- **Mode indicator**: Shows current mode (Add/Remove) based on Alt key state
- **Undo button**: Removes the last annotation point
- **Clear All button**: Resets all annotations

### Selection Info Bar

Displays:
- Total annotation counts (+N / -N format)
- Current frame number
- Preview status

## API Integration

### Frontend Request

```typescript
// PointPrompt interface
interface PointPrompt {
  x: number;
  y: number;
  label: 0 | 1;  // 0=background, 1=foreground
}

// API call with multiple points
api.quickRoto(
  videoFile,
  null,  // clickX (legacy)
  null,  // clickY (legacy)
  frameIdx,
  objectLabel,
  outputFormat,
  undefined,  // box
  annotationPoints  // Array of PointPrompt
);
```

### Backend Handling

```python
# Endpoint accepts multiple points
@router.post("/quick-roto")
async def quick_roto(
    video: UploadFile = File(...),
    points: Optional[str] = Form(None),  # JSON: [[x, y, label], ...]
    # ... other params
):
    # Parse annotation points
    if points:
        points_data = json.loads(points)
        annotation_points = np.array([[p[0], p[1]] for p in points_data])
        annotation_labels = np.array([p[2] for p in points_data])

    # Create segmentation prompt with all points
    prompt = SegmentationPrompt(
        frame_idx=frame_idx,
        points=annotation_points,
        point_labels=annotation_labels,
        label=label,
    )
```

## Best Practices

### For Optimal Results

1. **Start with a clear foreground point**: Click on the most distinctive part of the object first
2. **Use remove points sparingly**: Add background points only where the mask incorrectly includes areas
3. **Work from coarse to fine**: Start with major regions, then refine edges
4. **Check the preview**: Wait for the mask preview to update before adding more points

### Common Workflows

**Workflow 1: Simple Object Selection**
1. Click once on the object center
2. If mask is correct, proceed to export

**Workflow 2: Object with Background Interference**
1. Click on the object (add point)
2. Alt+Click on incorrectly included background areas (remove points)
3. Add more foreground points if needed

**Workflow 3: Multiple Connected Regions**
1. Click on each separate region of the object (multiple add points)
2. Remove any incorrectly included areas

## Technical Implementation

### Files Modified

| File | Changes |
|------|---------|
| `frontend/src/app/features/video-roto/video-roto.component.ts` | Added annotation state, keyboard handlers, drawing functions |
| `frontend/src/app/features/video-roto/video-roto.component.html` | Added annotation toolbar and UI elements |
| `frontend/src/app/features/video-roto/video-roto.component.scss` | Added styles for annotation markers and toolbar |
| `frontend/src/app/core/services/api.service.ts` | Updated quickRoto to support multiple points |
| `backend/src/roto_seg/api/segment.py` | Added multi-point parsing and handling |

### Data Structures

```typescript
// Frontend annotation point
interface AnnotationPoint {
  x: number;       // Video coordinates
  y: number;
  label: 0 | 1;    // 0=background, 1=foreground
  displayX: number; // Display coordinates for rendering
  displayY: number;
}
```

### Keyboard Event Handling

```typescript
@HostListener('window:keydown', ['$event'])
onKeyDown(event: KeyboardEvent): void {
  if (event.key === 'Alt') {
    this.isAltPressed.set(true);
  }
  if ((event.ctrlKey || event.metaKey) && event.key === 'z') {
    event.preventDefault();
    this.undoLastPoint();
  }
}
```

## Comparison with Industry Tools

| Feature | Our Implementation | Silhouette | Nuke Roto |
|---------|-------------------|------------|-----------|
| Add points | Click | Click | Click |
| Remove points | Alt+Click | Alt+Click | Ctrl+Click |
| Undo | Ctrl+Z | Ctrl+Z | Ctrl+Z |
| Real-time preview | Yes | Yes | Yes |
| Multi-point support | Yes | Yes | Yes |

## Future Enhancements

1. **Brush/Paint mode**: Allow painting regions instead of clicking points
2. **Edge refinement**: Add edge-aware refinement for hair/fur
3. **Point dragging**: Allow repositioning existing points
4. **Save/Load annotations**: Persist annotation sessions
5. **Keyboard shortcuts**: Add more shortcuts (e.g., Tab to toggle mode)
