"""
AI Segmentation Service using SAM2 or fallback methods.

This module provides the core segmentation functionality:
- Single image segmentation with point/box prompts
- Video object segmentation with mask propagation
- Memory-efficient processing for M1 Macs
"""

import gc
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from roto_seg.ai.device import get_device, clear_memory
from roto_seg.core.config import settings

# Try to import SAM2 - gracefully handle if not installed
SAM2_AVAILABLE = False
try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    pass


class SegmentationService:
    """
    AI-powered segmentation service.

    Supports:
    - SAM2 for high-quality segmentation (if installed)
    - Fallback to basic thresholding (for testing without GPU)

    Optimized for M1 Mac with 16GB RAM:
    - Uses smaller model by default
    - Aggressive memory management
    - Batch size limits
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_sam2: bool = True,
    ):
        """
        Initialize segmentation service.

        Args:
            model_path: Path to SAM2 checkpoint. Defaults to settings.
            device: Device to use ("mps", "cuda", "cpu"). Auto-detects if None.
            use_sam2: Whether to use SAM2 (if available). Set False for testing.
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.device = get_device(device or settings.DEVICE)
        self.use_sam2 = use_sam2 and SAM2_AVAILABLE

        self._image_predictor: Optional["SAM2ImagePredictor"] = None
        self._video_predictor = None
        self._model_loaded = False

        print(f"SegmentationService initialized on device: {self.device}")
        print(f"SAM2 available: {SAM2_AVAILABLE}, using SAM2: {self.use_sam2}")

    def load_model(self):
        """Load the segmentation model into memory."""
        if self._model_loaded:
            return

        if not self.use_sam2:
            print("SAM2 not available or disabled, using fallback segmentation")
            self._model_loaded = True
            return

        try:
            checkpoint_path = Path(self.model_path) / settings.SAM2_MODEL
            config_name = self._get_config_name(settings.SAM2_MODEL)

            print(f"Loading SAM2 model: {checkpoint_path}")
            print(f"Config: {config_name}")

            # Build model
            sam2_model = build_sam2(
                config_name,
                str(checkpoint_path),
                device=self.device,
            )

            self._image_predictor = SAM2ImagePredictor(sam2_model)
            self._model_loaded = True
            print("SAM2 model loaded successfully")

        except Exception as e:
            print(f"Failed to load SAM2 model: {e}")
            print("Falling back to basic segmentation")
            self.use_sam2 = False
            self._model_loaded = True

    def _get_config_name(self, model_name: str) -> str:
        """Get SAM2 config name from model filename."""
        config_map = {
            # SAM2 original checkpoints
            "sam2_hiera_tiny.pt": "sam2_hiera_t.yaml",
            "sam2_hiera_small.pt": "sam2_hiera_s.yaml",
            "sam2_hiera_base_plus.pt": "sam2_hiera_b+.yaml",
            "sam2_hiera_large.pt": "sam2_hiera_l.yaml",
            # SAM2.1 checkpoints (compatible with sam2 package 1.1.0+)
            # Config path needs to include subdirectory
            "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
        }
        return config_map.get(model_name, "configs/sam2.1/sam2.1_hiera_s.yaml")

    def segment_image(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment objects in a single image.

        Args:
            image: RGB image as numpy array (H, W, 3)
            points: Point prompts as (N, 2) array of [x, y] coordinates
            point_labels: Labels for points (1=foreground, 0=background)
            box: Box prompt as [x1, y1, x2, y2]
            multimask_output: Whether to return multiple mask candidates

        Returns:
            Tuple of:
                - masks: Binary masks (N, H, W) where N is number of masks
                - scores: Confidence scores for each mask (N,)
                - logits: Raw logits for refinement (N, H, W)
        """
        self.load_model()

        if self.use_sam2 and self._image_predictor is not None:
            return self._segment_with_sam2(
                image, points, point_labels, box, multimask_output
            )
        else:
            return self._segment_fallback(image, points, box)

    def _segment_with_sam2(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
        multimask_output: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment using SAM2 model."""
        # Set image (computes embeddings)
        self._image_predictor.set_image(image)

        # Prepare prompts
        kwargs = {"multimask_output": multimask_output}

        if points is not None:
            kwargs["point_coords"] = points
            kwargs["point_labels"] = point_labels if point_labels is not None else np.ones(len(points))

        if box is not None:
            kwargs["box"] = box

        # Run prediction
        with torch.no_grad():
            masks, scores, logits = self._image_predictor.predict(**kwargs)

        # Clear memory after inference
        clear_memory(self.device)

        return masks, scores, logits

    def _segment_fallback(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray],
        box: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fallback segmentation using basic CV techniques.
        Used when SAM2 is not available (for testing).
        """
        import cv2

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if box is not None:
            # Use GrabCut with box
            x1, y1, x2, y2 = map(int, box)
            rect = (x1, y1, x2 - x1, y2 - y1)

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(
                image, mask, rect,
                bgd_model, fgd_model,
                5, cv2.GC_INIT_WITH_RECT
            )
            mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

        elif points is not None and len(points) > 0:
            # Simple flood fill from point
            x, y = map(int, points[0])
            cv2.floodFill(
                image.copy(), None, (x, y),
                newVal=(255, 255, 255),
                loDiff=(30, 30, 30),
                upDiff=(30, 30, 30)
            )
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Return in SAM2 format
        masks = mask[np.newaxis, :, :].astype(bool)
        scores = np.array([0.5])  # Low confidence for fallback
        logits = mask[np.newaxis, :, :].astype(np.float32)

        return masks, scores, logits

    def segment_video(
        self,
        video_path: str,
        initial_frame_idx: int,
        initial_points: np.ndarray,
        initial_labels: np.ndarray,
        object_id: int = 1,
    ):
        """
        Segment and track object through video.

        Args:
            video_path: Path to video file
            initial_frame_idx: Frame index where object is marked
            initial_points: Point prompts on initial frame
            initial_labels: Labels for points
            object_id: ID to assign to this object

        Yields:
            Tuple of (frame_idx, mask) for each frame
        """
        if not self.use_sam2:
            raise NotImplementedError(
                "Video segmentation requires SAM2. "
                "Install with: pip install segment-anything-2"
            )

        self.load_model()

        # Build video predictor if not already done
        if self._video_predictor is None:
            checkpoint_path = Path(self.model_path) / settings.SAM2_MODEL
            config_name = self._get_config_name(settings.SAM2_MODEL)
            self._video_predictor = build_sam2_video_predictor(
                config_name,
                str(checkpoint_path),
            )

        # Initialize video state
        inference_state = self._video_predictor.init_state(video_path=video_path)

        # Add initial prompts
        self._video_predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=initial_frame_idx,
            obj_id=object_id,
            points=initial_points,
            labels=initial_labels,
        )

        # Propagate through video
        for frame_idx, obj_ids, masks in self._video_predictor.propagate_in_video(
            inference_state
        ):
            if object_id in masks:
                yield frame_idx, masks[object_id].cpu().numpy()

            # Periodic memory cleanup
            if frame_idx % 50 == 0:
                clear_memory(self.device)
                gc.collect()

        # Final cleanup
        clear_memory(self.device)
        gc.collect()

    def unload_model(self):
        """Unload model to free memory."""
        self._image_predictor = None
        self._video_predictor = None
        self._model_loaded = False
        clear_memory(self.device)
        gc.collect()
        print("Model unloaded, memory cleared")


# Singleton instance for reuse
_segmentation_service: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    """Get or create the global segmentation service instance."""
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = SegmentationService()
    return _segmentation_service
