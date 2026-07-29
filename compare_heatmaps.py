"""
compare_heatmaps.py

Phase 4, Step 6: direct side-by-side comparison of localization
quality between Phase 3's autoencoder and Phase 4's PatchCore, on
the exact same defect image and ground-truth mask used in
anomaly_heatmap.py.

Four panels:
    1. Original defect image
    2. Autoencoder reconstruction-error heatmap (Phase 3)
    3. PatchCore nearest-neighbor-distance heatmap (Phase 4),
       upsampled from its native 32x32 feature-map resolution to
       256x256 for a fair visual comparison against panel 2
    4. Ground-truth defect mask

Also prints both models' anomaly score, threshold, and predicted
label for the same image, so the numbers and the pictures agree.

Run AFTER both train_autoencoder.py and train_patchcore.py have
saved their checkpoints:
    python3 compare_heatmaps.py
"""

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from autoencoder import ConvAutoencoder
from patchcore import PatchCoreExtractor, score_single_image_map
from dataset import MVTecDataset
from train_autoencoder import get_device

CATEGORY_PATH = "data/mvtec_anomaly_detection/transistor"
AUTOENCODER_CHECKPOINT = "models/phase3_autoencoder.pth"
PATCHCORE_CHECKPOINT = "models/phase4_patchcore_bank.joblib"
IMAGE_SIZE = 256

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def find_defect_sample_with_mask(test_ds: MVTecDataset):
    """Same logic as anomaly_heatmap.py -- picks the first defect
    sample with a ground-truth mask, so both scripts land on the
    identical image without hardcoding a filename."""
    for image, label, mask_path in test_ds:
        if label == 1 and mask_path is not None:
            return image, label, mask_path
    raise RuntimeError("No defect sample with a mask found in the test set.")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # --- Load Phase 3 autoencoder ---
    ae_checkpoint = torch.load(AUTOENCODER_CHECKPOINT, map_location=device)
    autoencoder = ConvAutoencoder().to(device)
    autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
    autoencoder.eval()
    ae_threshold = ae_checkpoint["threshold"]
    ae_top_k_percent = ae_checkpoint["top_k_percent"]

    # --- Load Phase 4 PatchCore ---
    pc_checkpoint = joblib.load(PATCHCORE_CHECKPOINT)
    nn_index = pc_checkpoint["nn_index"]
    pc_threshold = pc_checkpoint["threshold"]
    pc_top_k_percent = pc_checkpoint["top_k_percent"]
    extractor = PatchCoreExtractor(device)  # fresh, deterministic -- nothing to load here

    # --- Same defect sample used in anomaly_heatmap.py ---
    test_ds = MVTecDataset(CATEGORY_PATH, split="test", mode="anomaly")
    image, label, mask_path = find_defect_sample_with_mask(test_ds)
    print(f"comparing on defect sample: {mask_path}")

    # --- Autoencoder heatmap + score ---
    with torch.no_grad():
        input_batch = image.unsqueeze(0).to(device)
        recon = autoencoder(input_batch)
    ae_error_map = ((recon - input_batch) ** 2).mean(dim=1).squeeze(0).cpu().numpy()

    flat_error = ae_error_map.flatten()
    k = max(1, int(flat_error.size * ae_top_k_percent))
    top_k_indices = np.argpartition(flat_error, -k)[-k:]
    ae_score = flat_error[top_k_indices].mean()
    ae_predicted = "defect" if ae_score > ae_threshold else "good"

    # --- PatchCore heatmap + score ---
    pc_score, pc_distance_map = score_single_image_map(
        extractor, image, nn_index, pc_top_k_percent
    )
    pc_predicted = "defect" if pc_score > pc_threshold else "good"

    # Smooth at the NATIVE 32x32 resolution first, then upsample.
    # Smoothing after the 8x bilinear upsample (256/32) barely
    # touches the ~8px-wide blocky artifacts upsampling itself
    # creates -- each native cell already occupies its own 8x8 block
    # by the time it's blown up. A light sigma here, before
    # upsampling, actually merges neighboring native cells instead of
    # blurring within an already-blocky image. pc_score above is
    # unaffected either way -- it was computed from the raw distances.
    pc_distance_map_smoothed = gaussian_filter(pc_distance_map, sigma=1.2)

    pc_map_tensor = torch.tensor(pc_distance_map_smoothed).unsqueeze(0).unsqueeze(0).float()
    pc_map_display = F.interpolate(
        pc_map_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    print(
        f"\nAutoencoder -> score: {ae_score:.5f} (threshold: {ae_threshold:.5f}) "
        f"-> predicted: {ae_predicted}"
    )
    print(
        f"PatchCore   -> score: {pc_score:.5f} (threshold: {pc_threshold:.5f}) "
        f"-> predicted: {pc_predicted}"
    )
    print("actual: defect (both)")

    # --- Ground-truth mask ---
    mask_img = Image.open(mask_path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST
    )
    mask_np = np.array(mask_img)

    display_original = unnormalize(image).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(display_original)
    axes[0].set_title("Original (defect)")
    axes[0].axis("off")

    axes[1].imshow(ae_error_map, cmap="inferno")
    axes[1].set_title(f"Autoencoder heatmap\nscore={ae_score:.4f} ({ae_predicted})")
    axes[1].axis("off")

    axes[2].imshow(pc_map_display, cmap="inferno")
    axes[2].set_title(f"PatchCore heatmap\nscore={pc_score:.4f} ({pc_predicted})")
    axes[2].axis("off")

    axes[3].imshow(mask_np, cmap="gray")
    axes[3].set_title("Ground-truth defect mask")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()