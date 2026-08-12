"""
inference.py

Phase 5, Step 2: a single, reusable prediction function wrapping the
trained PatchCore model. Meant to be imported ONCE (by main.py) and
reused across every request -- re-loading the checkpoint or
rebuilding the frozen ResNet18 extractor per request would be slow
and wasteful in a live API.

predict(image) -> dict with:
    label: "good" or "defect"
    score: float, raw top-K nearest-neighbor distance
    threshold: float, the calibrated decision threshold (95th
        percentile, per train_patchcore.py)
    overlay_base64: PNG image, the original image with a
        Gaussian-smoothed anomaly heatmap alpha-blended on top,
        base64-encoded so a JSON API response can carry it directly
        without a separate image-hosting step.

This reuses the exact scoring path already proven correct in
patchcore.py / compare_heatmaps.py -- no reimplementation of the
top-K scoring or heatmap smoothing logic.
"""

import base64
import io

import joblib
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

from dataset import default_transform
from patchcore import PatchCoreExtractor, score_single_image_map

CATEGORY = "transistor"
PATCHCORE_CHECKPOINT = "models/phase4_patchcore_bank.joblib"
IMAGE_SIZE = 256
OVERLAY_ALPHA = 0.45  # heatmap opacity when blended onto the original image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _unnormalize(img: torch.Tensor) -> torch.Tensor:
    """Reverses dataset.py's ImageNet normalization for display --
    same helper used in anomaly_heatmap.py / compare_heatmaps.py."""
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


class PatchCoreInferenceService:
    """Loads the model once at construction; call .predict() per request."""

    def __init__(self, checkpoint_path: str = PATCHCORE_CHECKPOINT):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        checkpoint = joblib.load(checkpoint_path)
        self.nn_index = checkpoint["nn_index"]
        self.threshold = checkpoint["threshold"]
        self.top_k_percent = checkpoint["top_k_percent"]

        self.extractor = PatchCoreExtractor(self.device)
        self.transform = default_transform(IMAGE_SIZE)

        print(
            f"[inference] PatchCore service ready -- device={self.device}, "
            f"category={CATEGORY}, threshold={self.threshold:.5f}, "
            f"top_k_percent={self.top_k_percent}"
        )

    def _make_overlay(self, image_tensor: torch.Tensor, distance_map: np.ndarray) -> str:
        """Upsamples + smooths the raw distance map (same as
        compare_heatmaps.py), colorizes it, and alpha-blends it onto
        the unnormalized original image. Returns a base64 PNG."""
        upsampled = F.interpolate(
            torch.tensor(distance_map).unsqueeze(0).unsqueeze(0).float(),
            size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False,
        ).squeeze().numpy()
        smoothed = gaussian_filter(upsampled, sigma=4)

        norm = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
        heatmap_rgb = (cm.inferno(norm)[:, :, :3] * 255).astype(np.uint8)

        original_rgb = (_unnormalize(image_tensor).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        blended = (
            original_rgb.astype(np.float32) * (1 - OVERLAY_ALPHA)
            + heatmap_rgb.astype(np.float32) * OVERLAY_ALPHA
        ).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def predict(self, pil_image: Image.Image) -> dict:
        image_tensor = self.transform(pil_image.convert("RGB"))

        score, distance_map = score_single_image_map(
            self.extractor, image_tensor, self.nn_index, self.top_k_percent
        )
        label = "defect" if score > self.threshold else "good"

        return {
            "label": label,
            "score": round(float(score), 5),
            "threshold": round(float(self.threshold), 5),
            "category": CATEGORY,
            "overlay_base64": self._make_overlay(image_tensor, distance_map),
        }


# Module-level singleton -- import get_service() from main.py.
# Do NOT instantiate PatchCoreInferenceService() per-request; loading
# the memory bank and building the extractor is meant to happen once.
_service: "PatchCoreInferenceService | None" = None


def get_service() -> PatchCoreInferenceService:
    global _service
    if _service is None:
        _service = PatchCoreInferenceService()
    return _service


if __name__ == "__main__":
    # Quick sanity check -- run directly against a known defect image
    # from the official test set, so the printed decision can be
    # compared against what train_patchcore.py / compare_heatmaps.py
    # already reported for the same kind of sample.
    from dataset import MVTecDataset

    service = get_service()

    test_ds = MVTecDataset(f"data/mvtec_anomaly_detection/{CATEGORY}", split="test", mode="anomaly")
    defect_path = next(path for path, label, _mask in test_ds.samples if label == 1)

    pil_image = Image.open(defect_path).convert("RGB")
    result = service.predict(pil_image)

    print(f"\nimage: {defect_path}")
    print(f"predicted: {result['label']} (score={result['score']}, threshold={result['threshold']})")
    print("actual: defect")
    print(f"overlay_base64 length: {len(result['overlay_base64'])} chars")