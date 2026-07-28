"""
anomaly_heatmap.py

Phase 3, Step 5: visualizes where the autoencoder thinks a defect is,
for one real defect image from the official test/ split.

Loads the checkpoint saved by train_autoencoder.py (model weights +
calibrated threshold -- no retraining, no recalibration here). Shows
four panels side by side:
    1. Original defect image
    2. Autoencoder's reconstruction of it
    3. Per-pixel error heatmap (this IS the localization signal --
       the same top-K logic from train_autoencoder.py, just kept at
       full pixel resolution here instead of collapsed to one score)
    4. Ground-truth defect mask, for a direct sanity check of whether
       the heatmap's hot region actually lines up with the real
       defect location

Also prints the image's anomaly score against the calibrated
threshold, so you can see the actual pass/fail decision alongside
the visual.

Run AFTER train_autoencoder.py has saved a checkpoint:
    python3 anomaly_heatmap.py
"""

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from autoencoder import ConvAutoencoder
from dataset import MVTecDataset

CATEGORY_PATH = "data/mvtec_anomaly_detection/transistor"
CHECKPOINT_PATH = "models/phase3_autoencoder.pth"
IMAGE_SIZE = 256

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    """Reverses dataset.py's ImageNet normalization so the image is
    back in [0, 1] and displays correctly with matplotlib."""
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def find_defect_sample_with_mask(test_ds: MVTecDataset):
    """Returns the first (image, label, mask_path) in the test set
    that is both a real defect AND has a ground-truth mask on disk --
    lets us show the mask panel for a direct visual comparison."""
    for image, label, mask_path in test_ds:
        if label == 1 and mask_path is not None:
            return image, label, mask_path
    raise RuntimeError("No defect sample with a mask found in the test set.")


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold = checkpoint["threshold"]
    top_k_percent = checkpoint["top_k_percent"]
    print(f"loaded checkpoint -> threshold: {threshold:.5f}, top_k_percent: {top_k_percent}")

    test_ds = MVTecDataset(CATEGORY_PATH, split="test", mode="anomaly")
    image, label, mask_path = find_defect_sample_with_mask(test_ds)
    print(f"visualizing defect sample with mask: {mask_path}")

    # Run the image through the autoencoder.
    with torch.no_grad():
        input_batch = image.unsqueeze(0).to(device)
        recon = model(input_batch)

    # Per-pixel squared error, averaged over channels -- same as
    # top_k_error()'s first step in train_autoencoder.py, just kept
    # at full (H, W) resolution here instead of collapsed to one
    # number.
    error_map = ((recon - input_batch) ** 2).mean(dim=1).squeeze(0).cpu().numpy()

    # Anomaly score using the exact same top-K logic as training, so
    # the printed pass/fail decision matches what train_autoencoder.py
    # would have produced for this image.
    flat_error = error_map.flatten()
    k = max(1, int(flat_error.size * top_k_percent))
    top_k_indices = np.argpartition(flat_error, -k)[-k:]
    anomaly_score = flat_error[top_k_indices].mean()
    predicted_label = "defect" if anomaly_score > threshold else "good"

    print(f"anomaly score: {anomaly_score:.5f} (threshold: {threshold:.5f})")
    print(f"predicted: {predicted_label}  |  actual: defect")

    # Load and resize the ground-truth mask to match the model's
    # input resolution, for a fair visual side-by-side. NEAREST keeps
    # the binary mask's edges crisp instead of interpolation-blurred.
    mask_img = Image.open(mask_path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST
    )
    mask_np = np.array(mask_img)

    # Unnormalize both the original and reconstructed image for display.
    display_original = unnormalize(image).permute(1, 2, 0).numpy()
    display_recon = unnormalize(recon.squeeze(0).cpu()).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(display_original)
    axes[0].set_title("Original (defect)")
    axes[0].axis("off")

    axes[1].imshow(display_recon)
    axes[1].set_title("Autoencoder reconstruction")
    axes[1].axis("off")

    axes[2].imshow(error_map, cmap="inferno")
    axes[2].set_title(f"Error heatmap\nscore={anomaly_score:.4f}")
    axes[2].axis("off")

    axes[3].imshow(mask_np, cmap="gray")
    axes[3].set_title("Ground-truth defect mask")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()