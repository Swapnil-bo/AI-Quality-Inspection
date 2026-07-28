"""
threshold_sweep.py

Diagnostic only -- no retraining. Reuses the checkpoint saved by
train_autoencoder.py and sweeps the anomaly-score threshold across a
range of percentiles, printing precision/recall/F1 at each one.

First sweep (full train/good, all 213 images) found the 50th
percentile beat Phase 2's baseline on every metric (precision 0.408,
recall 0.725, f1 0.523 vs. 0.278 / 0.625 / 0.385). This version fixes
two things before trusting that number:
  1. Uses the EXACT same held-out calibration slice (31 images, same
     seed=42 split) that train_autoencoder.py uses -- not the full
     213, since 182 of those were seen during training and reconstruct
     artificially well, skewing the percentile calculation.
  2. Extends the sweep below 50 to check whether an even better
     operating point exists before precision collapses.

Run AFTER train_autoencoder.py has saved a checkpoint:
    python3 threshold_sweep.py
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split

from autoencoder import ConvAutoencoder
from dataset import MVTecDataset
from train_autoencoder import collate_skip_mask_path, get_device, score_loader

CATEGORY_PATH = "data/mvtec_anomaly_detection/transistor"
CHECKPOINT_PATH = "models/phase3_autoencoder.pth"
BATCH_SIZE = 16
PERCENTILES_TO_TRY = [10, 20, 30, 40, 45, 50, 55, 60, 70, 75, 80, 85, 90, 95]

PHASE2_BASELINE = {"precision": 0.278, "recall": 0.625, "f1": 0.385}


def main():
    device = get_device()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Rebuild the EXACT same fit/calibration split train_autoencoder.py
    # used (same seed=42, same 85/15 ratio) so this threshold search is
    # only ever evaluated against images the model never trained on.
    full_train_ds = MVTecDataset(CATEGORY_PATH, split="train", mode="anomaly")
    n_total = len(full_train_ds)
    n_cal = max(10, int(n_total * 0.15))
    n_fit = n_total - n_cal
    _fit_ds, cal_ds = random_split(
        full_train_ds, [n_fit, n_cal], generator=torch.Generator().manual_seed(42)
    )
    cal_loader = DataLoader(
        cal_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_skip_mask_path
    )
    cal_scores = score_loader(model, cal_loader, device)

    test_ds = MVTecDataset(CATEGORY_PATH, split="test", mode="anomaly")
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_skip_mask_path
    )
    test_scores = score_loader(model, test_loader, device)
    test_labels = [label for _, label, _ in test_ds]

    print(f"{'percentile':>10} {'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for pct in PERCENTILES_TO_TRY:
        threshold = np.percentile(cal_scores, pct)
        preds = (test_scores > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, average="binary", zero_division=0
        )
        print(f"{pct:>10} {threshold:>10.5f} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")

    print(
        f"\nPhase 2 baseline for reference -> "
        f"precision: {PHASE2_BASELINE['precision']}, "
        f"recall: {PHASE2_BASELINE['recall']}, "
        f"f1: {PHASE2_BASELINE['f1']}"
    )


if __name__ == "__main__":
    main()