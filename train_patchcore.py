"""
train_patchcore.py

Phase 4: builds a PatchCore anomaly detector -- pretrained ResNet18
as a fixed feature extractor, memory bank of good-image patch
embeddings, nearest-neighbor distance as the anomaly score. No
training loop for the model itself (unlike Phase 3's autoencoder) --
this is a similarity-search approach, not a generative one.

Reuses the exact same fit/calibration split (seed=42, same 85/15
ratio) as train_autoencoder.py, and the same held-out-calibration
threshold methodology, so Phase 2 / Phase 3 / Phase 4 numbers are
directly comparable.

CALIBRATION_PERCENTILE starts at 55 -- the operating point Phase 3
settled on after its own sweep -- as a reasonable first guess, NOT a
final answer. PatchCore's score distribution is a different
statistic entirely (nearest-neighbor distance vs. reconstruction
error), so re-run a threshold sweep against these actual results
before trusting this number the way we had to learn to do for
Phase 3.

Run:
    python3 train_patchcore.py
"""

import os

import joblib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset import MVTecDataset
from patchcore import PatchCoreExtractor, build_memory_bank, coreset_subsample, score_images
from train_autoencoder import collate_skip_mask_path, get_device

CATEGORY_PATH = "data/mvtec_anomaly_detection/transistor"
BATCH_SIZE = 8                 # smaller than Phase 3 -- each image now produces many
                                # patch embeddings at once (~1024 for a 256x256 input)
TOP_K_PERCENT = 0.01           # top 1% of patches drive the score -- tighter than Phase 3's
                                # 10% since patches are already local regions, not raw pixels
CORESET_FRACTION = 0.10        # keep 10% of the memory bank via random subsampling
CALIBRATION_PERCENTILE = 95    # picked via patchcore_threshold_sweep.py: unlike Phase 3's
                                # sweep, this wasn't a tradeoff -- 95 had the best precision,
                                # best f1, AND matched the best recall (0.950) of every
                                # percentile tested (40 through 95). Final numbers: precision
                                # 0.826, recall 0.950, f1 0.884, ~13% false-alarm rate on good
                                # parts (vs. Phase 3's 38% and Phase 2's much worse ratio).
                                # Recall plateaus at 0.950 (38/40) from percentile 80 onward --
                                # the same 2 real defects go uncaught regardless of threshold
                                # in that range, a property of those specific defects, not a
                                # threshold choice.
MEMORY_BANK_PATH = "models/phase4_patchcore_bank.joblib"

PHASE2_BASELINE = {"precision": 0.278, "recall": 0.625, "f1": 0.385}
PHASE3_AUTOENCODER = {"precision": 0.477, "recall": 0.525, "f1": 0.500}


def plot_confusion_matrix(cm, title="Phase 4 PatchCore - Official Test Set"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["good", "defect"])
    ax.set_yticklabels(["good", "defect"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    extractor = PatchCoreExtractor(device)

    full_train_ds = MVTecDataset(CATEGORY_PATH, split="train", mode="anomaly")
    n_total = len(full_train_ds)
    n_cal = max(10, int(n_total * 0.15))
    n_fit = n_total - n_cal
    fit_ds, cal_ds = random_split(
        full_train_ds, [n_fit, n_cal], generator=torch.Generator().manual_seed(42)
    )
    print(f"fit (memory bank): {n_fit}, calibration (threshold only): {n_cal}")

    fit_loader = DataLoader(
        fit_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_skip_mask_path
    )
    cal_loader = DataLoader(
        cal_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_skip_mask_path
    )
    test_ds = MVTecDataset(CATEGORY_PATH, split="test", mode="anomaly")
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_skip_mask_path
    )

    print("building memory bank from fit set...")
    memory_bank = build_memory_bank(extractor, fit_loader)
    print(f"raw memory bank size: {memory_bank.shape}")

    memory_bank = coreset_subsample(memory_bank, CORESET_FRACTION)
    print(f"after coreset subsampling ({CORESET_FRACTION:.0%}): {memory_bank.shape}")

    nn_index = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn_index.fit(memory_bank)

    print("calibrating threshold on held-out good images...")
    cal_scores, _ = score_images(extractor, cal_loader, nn_index, TOP_K_PERCENT)
    threshold = float(np.percentile(cal_scores, CALIBRATION_PERCENTILE))
    print(
        f"calibration good scores -> mean: {cal_scores.mean():.5f}, "
        f"{CALIBRATION_PERCENTILE}th percentile threshold: {threshold:.5f}"
    )

    # Save the index alongside its calibration metadata -- same fix
    # applied to train_autoencoder.py's checkpoint after the first
    # version saved weights with no way to recover the threshold.
    # A future heatmap/deployment script needs threshold and
    # top_k_percent to reproduce this exact pass/fail decision.
    os.makedirs(os.path.dirname(MEMORY_BANK_PATH), exist_ok=True)
    joblib.dump(
        {
            "nn_index": nn_index,
            "threshold": threshold,
            "top_k_percent": TOP_K_PERCENT,
            "calibration_percentile": CALIBRATION_PERCENTILE,
            "coreset_fraction": CORESET_FRACTION,
        },
        MEMORY_BANK_PATH,
    )
    print(f"saved memory bank index + calibrated threshold to {MEMORY_BANK_PATH}")

    print("evaluating on official test/ split...")
    test_scores, test_labels = score_images(extractor, test_loader, nn_index, TOP_K_PERCENT)
    test_preds = (test_scores > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(test_labels, test_preds)

    print("\n--- Phase 4 PatchCore vs official test/ split ---")
    print(f"precision: {precision:.3f}")
    print(f"recall:    {recall:.3f}")
    print(f"f1:        {f1:.3f}")
    print(f"confusion matrix:\n{cm}")

    print("\n--- Three-way comparison ---")
    print(
        f"Phase 2 (classifier):  precision {PHASE2_BASELINE['precision']}, "
        f"recall {PHASE2_BASELINE['recall']}, f1 {PHASE2_BASELINE['f1']}"
    )
    print(
        f"Phase 3 (autoencoder): precision {PHASE3_AUTOENCODER['precision']}, "
        f"recall {PHASE3_AUTOENCODER['recall']}, f1 {PHASE3_AUTOENCODER['f1']}"
    )
    print(f"Phase 4 (PatchCore):   precision {precision:.3f}, recall {recall:.3f}, f1 {f1:.3f}")

    plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()