# SAM2 Video Changes Log

## Summary
This document captures the SAM2 video-related backend changes already implemented to improve tracking quality and observability.

## What Was Changed

### 1. Real SAM2 Video Propagation Added
- Introduced a true SAM2 video propagation path using `build_sam2_video_predictor`.
- Added prompt-based video segmentation method:
  - `segment_video_with_prompt(...)`
- Supports:
  - point prompts
  - box prompts
- Replaced default legacy center-point drift behavior in `auto` mode when SAM2 video is available.

### 2. Propagation Modes Added
- Added `propagation_mode` support in pipeline:
  - `auto`
  - `sam2_video`
  - `legacy`
- Resolution logic:
  - `auto` -> `sam2_video` if available, otherwise `legacy`
  - `sam2_video` can be explicitly forced
  - `legacy` remains available as fallback

### 3. Runtime Diagnostics Added
- Added model/runtime diagnostics in segmentation service:
  - active model name
  - device
  - SAM2 availability/enabled status
  - model file existence
  - video predictor readiness
- Added debug logs for quick-roto runtime and mode resolution.

### 4. API Improvements
- `/api/v1/segment/device-info` now includes:
  - `active_model`
  - `sam2_available`
  - `sam2_enabled`
- `quick-roto` accepts:
  - `propagation_mode` (form field)
- Response headers include:
  - `X-Propagation-Mode`
  - `X-Active-Model`

### 5. Error Handling Fix
- Preserved `HTTPException` flow in `quick-roto` endpoint (no double-wrapping into generic 500).

### 6. Lightweight Mask Cleanup
- Added post-processing in pipeline:
  - tiny-component removal
  - hole filling
- Applied to both SAM2 video outputs and legacy path outputs.

### 7. SAM2 Video Init Fallback (No decord Case)
- Some environments lack `decord` for direct MP4 initialization in SAM2 video predictor.
- Added automatic fallback:
  1. try `init_state(video_path=<video_file>)`
  2. on failure, extract JPG sequence with OpenCV
  3. run `init_state(video_path=<frames_dir>)`

## Files Updated
- `backend/src/roto_seg/ai/segmentation.py`
- `backend/src/roto_seg/services/roto_pipeline.py`
- `backend/src/roto_seg/api/segment.py`

## Current Behavior
1. `quick-roto` in `propagation_mode=auto` prefers SAM2 video predictor.
2. Active model is read from environment (`SAM2_MODEL`), currently configured for `sam2.1_hiera_large.pt`.
3. If direct SAM2 video init fails (for example due to missing `decord`), backend falls back to JPG sequence mode automatically.

## Verification Checklist
1. Start backend and check logs:
   - `SAM2 available: True, using SAM2: True`
   - `model=sam2.1_hiera_large.pt`
2. Call `GET /api/v1/segment/device-info` and verify:
   - `active_model` matches expected model
   - `sam2_enabled=true`
3. Run `quick-roto` and verify response headers:
   - `X-Propagation-Mode: sam2_video` (or expected mode)
   - `X-Active-Model: sam2.1_hiera_large.pt`

## Known Limitations
- `decord` may not be available on some local platforms, so JPG fallback is used.
- Processing is currently synchronous for quick-roto endpoint.
- Temporal refinement/matting stage is not yet integrated in this change set.
