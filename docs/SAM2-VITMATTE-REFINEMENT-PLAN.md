# SAM2 + ViTMatte Refinement Plan

## Goal
Improve edge quality and temporal stability of roto output without integrating MAM2 yet.

## Scope
- Keep SAM2 video propagation as the tracking backbone.
- Add matting refinement with ViTMatte.
- Improve EXR output quality using refined alpha.
- Add API and UI controls for refinement options.

## Implementation Plan

### 1. Stabilize Tracking
1. Keep `SAM2 video` as default propagation (`auto -> sam2_video`).
2. Add frame-level diagnostics and clearer failure logs.

### 2. Add Edge Refinement Stage (No MAM2)
1. Generate trimap from SAM2 mask for each frame.
2. Run ViTMatte refinement to produce soft alpha.
3. Add optional temporal alpha smoothing to reduce flicker.

### 3. Upgrade Exports
1. Export EXR from refined alpha (float matte).
2. Keep FXS export from binary/vectorized masks (optional alpha threshold path).

### 4. API and Frontend Controls
1. Extend `quick-roto` options:
   - `matting=true|false`
   - `matting_model=vitmatte`
   - `temporal_smooth=0..1`
2. Add Quick Roto UI toggles for these options.

### 5. Validation
1. Compare baseline vs refined on 2-3 clips.
2. Measure:
   - Edge quality
   - Temporal flicker
   - Manual cleanup time in comp

## Pipeline Diagram

### Before

```text
Video
  -> SAM2 prompt on key frame
  -> SAM2 video propagation
  -> binary masks
  -> EXR/FXS export
```

### After

```text
Video
  -> SAM2 prompt on key frame
  -> SAM2 video propagation
  -> binary masks
  -> trimap generation
  -> ViTMatte refinement (soft alpha)
  -> temporal smoothing (optional)
  -> EXR alpha export (HQ matte)
  -> FXS export (optional thresholded/vectorized)
```

## Why This Order
1. Largest quality gain comes from matting refinement.
2. Low-risk rollout because existing SAM2 flow remains intact.
3. MAM2 can be evaluated later only if ViTMatte quality is insufficient.
