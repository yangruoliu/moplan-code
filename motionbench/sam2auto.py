"""SAM2 automatic instance masks for a first frame + box-prompted fallback (GPU; lazy imports).
Cleanup mirrors scripts/cache_sam2_masks.py: area window, higher predicted IoU wins contested pixels, largest first."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class Sam2Masks:
    def __init__(self, checkpoint: str, config: str = SAM2_CONFIG, device: str = "cuda", points_per_side: int = 32,
                 points_per_batch: int = 64, crop_n_layers: int = 0):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.model = build_sam2(config, checkpoint, device=torch.device(device))
        self.generator = SAM2AutomaticMaskGenerator(self.model, points_per_side=points_per_side,
                                                    points_per_batch=points_per_batch, crop_n_layers=crop_n_layers)
        self.predictor = SAM2ImagePredictor(self.model)

    def automatic(self, image: np.ndarray, min_area_fraction: float = 0.001, max_area_fraction: float = 0.6,
                  max_instances: int = 32) -> np.ndarray:
        """image [H,W,3] RGB uint8 → masks [K,H,W] bool, non-overlapping, largest first."""
        import torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            proposals = self.generator.generate(image)
        area = image.shape[0] * image.shape[1]
        proposals = [p for p in proposals if min_area_fraction * area <= p["area"] <= max_area_fraction * area]
        proposals.sort(key=lambda p: -float(p.get("predicted_iou", 0.0)))
        claimed = np.zeros(image.shape[:2], bool)
        cleaned = []
        for p in proposals:
            m = np.asarray(p["segmentation"], bool) & ~claimed
            if m.sum() < min_area_fraction * area:
                continue
            claimed |= m
            cleaned.append(m)
        cleaned.sort(key=lambda m: -int(m.sum()))
        return np.array(cleaned[:max_instances]) if cleaned else np.zeros((0, *image.shape[:2]), bool)

    def from_box(self, image: np.ndarray, box) -> Optional[np.ndarray]:
        """Best mask for a box prompt [x1,y1,x2,y2] in image pixels."""
        import torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(box=np.asarray(box, np.float32)[None], multimask_output=True)
        if masks is None or len(masks) == 0:
            return None
        masks = np.asarray(masks).reshape(-1, *image.shape[:2]) > 0
        scores = np.asarray(scores).reshape(-1)
        return masks[int(np.argmax(scores))]
