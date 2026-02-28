# AI Models Guide

## Overview

This document provides detailed information about each AI model used in the roto pipeline, including capabilities, performance characteristics, and implementation notes.

---

## Segmentation Models

### SAM 2 (Segment Anything Model 2)

**Source:** Meta AI
**Paper:** [Segment Anything in Images and Videos](https://ai.meta.com/sam2/)
**License:** Apache 2.0

#### Capabilities
- Single-click object segmentation
- Box prompt segmentation
- Multi-mask output with confidence scores
- Video object segmentation (built-in propagation)
- Handles complex object boundaries

#### Model Variants

| Variant | Parameters | VRAM | Speed | Quality |
|---------|-----------|------|-------|---------|
| SAM 2 Tiny | 39M | 4 GB | Fast | Good |
| SAM 2 Small | 46M | 6 GB | Medium | Better |
| SAM 2 Base Plus | 81M | 8 GB | Medium | High |
| SAM 2 Large | 224M | 16 GB | Slower | Best |

#### Installation

```bash
pip install segment-anything-2

# Or from source
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

#### Usage

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model
checkpoint = "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2 = build_sam2(model_cfg, checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2)

# Segment with point prompt
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),  # 1 = foreground
    multimask_output=True
)

# Select best mask
best_mask = masks[scores.argmax()]
```

#### Video Segmentation

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Initialize with video
inference_state = predictor.init_state(video_path="video.mp4")

# Add prompt on first frame
predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=np.array([[500, 375]]),
    labels=np.array([1])
)

# Propagate through video
for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
    # masks is dict: {obj_id: mask_tensor}
    process_frame(frame_idx, masks)
```

---

### Grounding DINO

**Source:** IDEA Research
**Paper:** [Grounding DINO](https://arxiv.org/abs/2303.05499)
**License:** Apache 2.0

#### Capabilities
- Text-to-bounding-box detection
- Open-vocabulary object detection
- Multi-object detection from single prompt
- Phrase grounding

#### Installation

```bash
pip install groundingdino-py

# Or from source
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

#### Usage

```python
from groundingdino.util.inference import load_model, predict

model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

# Detect with text prompt
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="person . car . dog",  # Period-separated classes
    box_threshold=0.35,
    text_threshold=0.25
)

# boxes: normalized [cx, cy, w, h] format
# Convert to pixel coordinates
h, w = image.shape[:2]
boxes_px = boxes * torch.tensor([w, h, w, h])
```

#### Combined with SAM 2

```python
def text_to_masks(image, text_prompt):
    """Text prompt → detection → segmentation"""
    # Step 1: Get bounding boxes from text
    boxes, logits, phrases = grounding_predict(model, image, text_prompt)

    # Step 2: Use boxes to prompt SAM 2
    predictor.set_image(image)

    all_masks = []
    for box in boxes:
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=False
        )
        all_masks.append(masks[0])

    return all_masks, phrases
```

---

## Video Object Segmentation (VOS)

### Cutie

**Source:** HKUST
**Paper:** [Putting the Object Back into Video Object Segmentation](https://arxiv.org/abs/2310.12982)
**License:** MIT

#### Capabilities
- State-of-the-art VOS performance
- Handles occlusions and reappearances
- Memory-efficient architecture
- Real-time capable

#### Installation

```bash
pip install cutie-video-segmentation

# Or from source
git clone https://github.com/hkchengrex/Cutie.git
cd Cutie
pip install -e .
```

#### Usage

```python
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

# Load model
cutie = get_default_model()
processor = InferenceCore(cutie, cfg=cutie.cfg)

# First frame with mask
processor.step(frame_0, mask_0, objects=[1])  # object ID = 1

# Subsequent frames (no mask needed)
for frame in frames[1:]:
    output_mask = processor.step(frame)
    # output_mask contains propagated segmentation
```

### XMem

**Source:** HKUST
**Paper:** [XMem: Long-Term Video Object Segmentation](https://arxiv.org/abs/2207.07115)
**License:** MIT

#### Capabilities
- Long-term memory for extended videos
- Handles complex multi-object scenarios
- Bi-directional propagation support

#### Installation

```bash
git clone https://github.com/hkchengrex/XMem.git
cd XMem
pip install -e .
```

#### Usage

```python
from inference.inference_core import InferenceCore
from model.network import XMem

# Load model
network = XMem(config, model_path).eval().cuda()
processor = InferenceCore(network, config)

# Process video
with torch.no_grad():
    # Initialize with first frame mask
    processor.set_all_labels(list(range(1, num_objects + 1)))

    for ti, frame in enumerate(frames):
        if ti == 0:
            # First frame: provide mask
            output = processor.step(frame, first_frame_mask)
        else:
            # Subsequent frames: propagate
            output = processor.step(frame)

        masks.append(output)
```

---

## Matting Models

### ViTMatte

**Source:** Microsoft Research
**Paper:** [ViTMatte: Boosting Image Matting with Pretrained ViT](https://arxiv.org/abs/2305.15272)
**License:** MIT

#### Capabilities
- Transformer-based architecture
- High-quality alpha matte output
- Handles fine details (hair, fur)
- Works with automatic or manual trimaps

#### Installation

```bash
git clone https://github.com/hustvl/ViTMatte.git
cd ViTMatte
pip install -e .
```

#### Usage

```python
from vitmatte import ViTMatte

model = ViTMatte.from_pretrained("vitmatte-small")

# Create trimap from binary mask
def create_trimap(mask, erosion=10, dilation=10):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=erosion)
    dilated = cv2.dilate(mask, kernel, iterations=dilation)

    trimap = np.zeros_like(mask)
    trimap[dilated == 255] = 128  # Unknown
    trimap[eroded == 255] = 255   # Foreground
    return trimap

# Generate alpha matte
trimap = create_trimap(binary_mask)
alpha = model.predict(image, trimap)

# alpha is float [0, 1] with soft edges
```

#### Automatic Trimap Generation

```python
def auto_trimap_from_mask(mask, narrow=5, wide=20):
    """
    Create trimap with narrow unknown region for sharp edges,
    wider for complex edges.
    """
    # Detect edge complexity
    edges = cv2.Canny(mask, 50, 150)
    edge_density = np.sum(edges > 0) / np.sum(mask > 0)

    # Adjust erosion/dilation based on complexity
    if edge_density > 0.1:  # Complex edges (hair, etc.)
        erosion, dilation = narrow, wide
    else:  # Simple edges
        erosion, dilation = narrow, narrow + 5

    return create_trimap(mask, erosion, dilation)
```

---

### MODNet (Real-time Alternative)

**Source:** ZHKKKe
**License:** CC BY-NC-SA 4.0

#### Capabilities
- Real-time matting (30+ fps)
- No trimap required
- Good for portraits/people
- Lower quality than ViTMatte

#### Usage

```python
from modnet import MODNet

model = MODNet(backbone_pretrained=False)
model.load_state_dict(torch.load('modnet_photographic.ckpt'))

# Direct alpha prediction (no trimap)
alpha = model(image)
```

---

## Model Comparison

### Segmentation Quality

| Model | Simple Objects | Complex Edges | Hair/Fur | Speed |
|-------|---------------|---------------|----------|-------|
| SAM 2 Large | Excellent | Very Good | Good | Medium |
| SAM 2 Tiny | Very Good | Good | Fair | Fast |
| Mobile SAM | Good | Fair | Poor | Very Fast |

### Video Propagation Quality

| Model | Temporal Consistency | Occlusion Handling | Memory | Speed |
|-------|---------------------|-------------------|--------|-------|
| Cutie | Excellent | Excellent | Low | Fast |
| XMem | Excellent | Very Good | Medium | Medium |
| SAM 2 Video | Very Good | Good | High | Medium |

### Matting Quality

| Model | Hair Detail | Transparency | Speed | Trimap Required |
|-------|------------|--------------|-------|-----------------|
| ViTMatte | Excellent | Very Good | Slow | Yes |
| MODNet | Good | Fair | Fast | No |
| Matte Anything | Very Good | Good | Medium | No |

---

## GPU Memory Management

### Model Loading Strategy

```python
class ModelManager:
    """Manage GPU memory across multiple models"""

    def __init__(self, device="cuda"):
        self.device = device
        self.loaded_models = {}

    def get_model(self, model_name):
        """Load model on demand, unload others if needed"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        # Check available memory
        free_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory -= torch.cuda.memory_allocated()

        required = self.model_memory_requirements[model_name]

        if free_memory < required:
            self.unload_least_used()

        model = self.load_model(model_name)
        self.loaded_models[model_name] = model
        return model

    def unload_least_used(self):
        """Free GPU memory by unloading inactive models"""
        if not self.loaded_models:
            return

        # Unload oldest accessed model
        oldest = min(self.loaded_models, key=lambda k: self.access_times[k])
        del self.loaded_models[oldest]
        torch.cuda.empty_cache()
```

### Inference Optimization

```python
# Enable memory-efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Use automatic mixed precision
with torch.cuda.amp.autocast():
    output = model(input)

# Clear cache between large batches
torch.cuda.empty_cache()

# Use gradient checkpointing for training
model.gradient_checkpointing_enable()
```

---

## Model Download Links

| Model | Size | Download |
|-------|------|----------|
| SAM 2 Large | 2.4 GB | [HuggingFace](https://huggingface.co/facebook/sam2-hiera-large) |
| SAM 2 Base+ | 320 MB | [HuggingFace](https://huggingface.co/facebook/sam2-hiera-base-plus) |
| Grounding DINO | 1.3 GB | [GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| Cutie | 200 MB | [GitHub](https://github.com/hkchengrex/Cutie) |
| XMem | 180 MB | [GitHub](https://github.com/hkchengrex/XMem) |
| ViTMatte | 400 MB | [GitHub](https://github.com/hustvl/ViTMatte) |
