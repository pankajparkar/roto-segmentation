#!/usr/bin/env python3
"""
Test script for the segmentation pipeline.

Run from the backend directory:
    python scripts/test_segmentation.py

This tests:
1. Device detection (MPS/CUDA/CPU)
2. Mask to Bezier conversion
3. FXS export
4. Full pipeline (if test image provided)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_device_detection():
    """Test device detection."""
    print("\n" + "="*50)
    print("Testing Device Detection")
    print("="*50)

    from roto_seg.ai.device import get_device, get_device_info

    info = get_device_info()
    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"PyTorch version: {info['torch_version']}")
    print(f"\nDevice availability:")
    print(f"  MPS (Apple Silicon): {info['devices']['mps']['available']}")
    print(f"  CUDA (NVIDIA): {info['devices']['cuda']['available']}")
    print(f"  CPU: {info['devices']['cpu']['available']}")
    print(f"\nRecommended device: {info['recommended']}")

    device = get_device()
    print(f"Selected device: {device}")

    return True


def test_mask_to_bezier():
    """Test mask to Bezier conversion."""
    print("\n" + "="*50)
    print("Testing Mask to Bezier Conversion")
    print("="*50)

    import numpy as np
    from roto_seg.services.mask_to_bezier import mask_to_shapes

    # Create a simple circular mask
    h, w = 512, 512
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a circle
    import cv2
    cv2.circle(mask, (256, 256), 100, 255, -1)

    print(f"Input mask shape: {mask.shape}")
    print(f"Mask area: {mask.sum() // 255} pixels")

    # Convert to shapes
    shapes = mask_to_shapes(mask, target_points=30, label="test_circle")

    print(f"Number of shapes: {len(shapes)}")
    for i, shape in enumerate(shapes):
        print(f"  Shape {i}: {len(shape.points)} points, closed={shape.closed}")

    if shapes:
        # Check points are valid
        shape = shapes[0]
        for j, point in enumerate(shape.points[:3]):
            print(f"    Point {j}: center={point.center}, "
                  f"right={point.right_tangent}, left={point.left_tangent}")

    return len(shapes) > 0


def test_fxs_export():
    """Test FXS export."""
    print("\n" + "="*50)
    print("Testing FXS Export")
    print("="*50)

    import numpy as np
    import tempfile
    from roto_seg.services.mask_to_bezier import mask_to_shapes
    from roto_seg.exporters.fxs_exporter import FXSExporter

    # Create test masks
    h, w = 1080, 1920
    masks_by_frame = {}

    import cv2
    for frame_idx in range(1001, 1011):
        mask = np.zeros((h, w), dtype=np.uint8)
        # Moving circle
        x = 500 + (frame_idx - 1001) * 50
        cv2.circle(mask, (x, 500), 100, 255, -1)

        shapes = mask_to_shapes(mask, label="moving_circle")
        if shapes:
            masks_by_frame[frame_idx] = shapes

    print(f"Created {len(masks_by_frame)} frames with shapes")

    # Export to FXS
    exporter = FXSExporter(
        width=w,
        height=h,
        start_frame=1001,
        end_frame=1010,
    )

    with tempfile.NamedTemporaryFile(suffix='.fxs', delete=False) as f:
        output_path = f.name

    result_path = exporter.export(masks_by_frame, output_path, "AI_Test")
    print(f"Exported to: {result_path}")

    # Verify file
    with open(result_path, 'r') as f:
        content = f.read()

    print(f"File size: {len(content)} bytes")
    print(f"Contains 'Silhouette': {'Silhouette' in content}")
    print(f"Contains 'Bezier': {'Bezier' in content}")
    print(f"Contains 'Path': {'Path' in content}")

    # Print first few lines
    print("\nFirst 20 lines of FXS file:")
    for line in content.split('\n')[:20]:
        print(f"  {line}")

    return True


def test_segmentation_service():
    """Test segmentation service (basic, no model)."""
    print("\n" + "="*50)
    print("Testing Segmentation Service (Fallback Mode)")
    print("="*50)

    import numpy as np
    from roto_seg.ai.segmentation import SegmentationService

    # Create service without SAM2
    service = SegmentationService(use_sam2=False)

    # Create a test image
    import cv2
    h, w = 512, 512
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw a white rectangle
    cv2.rectangle(image, (150, 150), (350, 350), (255, 255, 255), -1)

    print(f"Test image shape: {image.shape}")

    # Segment with box prompt
    box = np.array([140, 140, 360, 360])
    masks, scores, logits = service.segment_image(image, box=box)

    print(f"Output masks shape: {masks.shape}")
    print(f"Scores: {scores}")
    print(f"Mask coverage: {masks[0].sum() / (h * w) * 100:.1f}%")

    return masks.shape[0] > 0


def test_full_pipeline_mock():
    """Test full pipeline with mock data."""
    print("\n" + "="*50)
    print("Testing Full Pipeline (Mock)")
    print("="*50)

    import numpy as np
    import tempfile
    import cv2
    from roto_seg.services.roto_pipeline import RotoPipeline, SegmentationPrompt

    # Create a test video (just frames in a temp dir)
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Creating test frames in: {temp_dir}")

    # Create 10 frames with a moving circle
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = 200 + i * 30
        cv2.circle(frame, (x, 240), 50, (255, 255, 255), -1)
        cv2.imwrite(str(temp_dir / f"frame_{i:04d}.png"), frame)

    print(f"Created 10 test frames")

    # Test pipeline
    pipeline = RotoPipeline()

    # Can't run full video pipeline without actual video file
    # But we can test the components work together
    print("Pipeline components initialized successfully")
    print(f"  Segmentation service: {pipeline.segmentation_service}")
    print(f"  Target points: {pipeline.target_points}")
    print(f"  Smoothing: {pipeline.smoothing}")

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("ROTO-SEG TEST SUITE")
    print("="*60)

    tests = [
        ("Device Detection", test_device_detection),
        ("Mask to Bezier", test_mask_to_bezier),
        ("FXS Export", test_fxs_export),
        ("Segmentation Service", test_segmentation_service),
        ("Full Pipeline (Mock)", test_full_pipeline_mock),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")

    all_passed = all(r[1] == "PASS" for r in results)
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
