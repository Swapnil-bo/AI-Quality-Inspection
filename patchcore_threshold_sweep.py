"""
patchcore_threshold_sweep.py

Diagnostic only -- no retraining, no rebuilding the memory bank.
Reuses the checkpoint saved by train_patchcore.py (nn_index +
metadata) and sweeps the calibration percentile across a range,
printing precision/recall/F1 at each one.

Why this matters: train_patchcore.py's CALIBRATION_PERCENTILE = 55
was explicitly a first guess carried over from Phase 3's autoencoder
-- PatchCore's score distribution (nearest-neighbor distance) has no
reason to share Phase 3's reconstruction-error distribution shape.
The first real run at 55 gave recall 0.975 but a 63% false-alarm
rate (38/60 good parts flagged) -- worth checking whether a higher
percentile trades a little recall for a meaningfully lower FPR,
the same tradeoff Phase 3 had to navigate.

Uses the EXACT same held-out calibration slice (same seed=42 split)
that train_patchcore.py used, so this sweep isn't contaminated by
images the memory bank was built from.

Run AFTER train_patchcore.py has saved a checkpoint:
    python3 patchcore_threshold_sweep.py
"""

import joblib
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split

from dataset import MVTecDataset
from patchcore import PatchCoreExtractor, score_images
from train_autoencoder import collate_skip_mask_path, get_device

CATEGORY_PATH = "data/mvtec_anomaly_detection/transistor"
MEMORY_BANK_PATH = "models/phase4_patchcore_bank.joblib"
BATCH_SIZE = 8
PERCENTILES_TO_TRY = [40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

PHASE2_BASELINE = {"precision": 0.278, "recall": 0.625, "f1": 0.385}
PHASE3_AUTOENCODER = {"precision": 0.477, "recall": 0.525, "f1": 0.500}


def main():
    device = get_device()
    checkpoint = joblib.load(MEMORY_BANK_PATH)
    nn_index = checkpoint["nn_index"]
    top_k_percent = checkpoint["top_k_percent"]
    print(f"loaded memory bank -> top_k_percent: {top_k_percent}")

    extractor = PatchCoreExtractor(device)

    # Rebuild the EXACT same fit/calibration split train_patchcore.py
    # used (same seed=42, same 85/15 ratio) so this sweep is only
    # ever evaluated against images the memory bank never saw.
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
    cal_scores, _ = score_images(extractor, cal_loader, nn_index, top_k_percent)

    test_ds = MVTecDataset(CATEGORY_PATH, split="test", mode="anomaly")
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_skip_mask_path
    )
    test_scores, test_labels = score_images(extractor, test_loader, nn_index, top_k_percent)

    print(f"\n{'percentile':>10} {'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for pct in PERCENTILES_TO_TRY:
        threshold = np.percentile(cal_scores, pct)
        preds = (test_scores > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, average="binary", zero_division=0
        )
        print(f"{pct:>10} {threshold:>10.5f} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")

    print(
        f"\nPhase 2 baseline  -> precision: {PHASE2_BASELINE['precision']}, "
        f"recall: {PHASE2_BASELINE['recall']}, f1: {PHASE2_BASELINE['f1']}"
    )
    print(
        f"Phase 3 autoencoder -> precision: {PHASE3_AUTOENCODER['precision']}, "
        f"recall: {PHASE3_AUTOENCODER['recall']}, f1: {PHASE3_AUTOENCODER['f1']}"
    )


if __name__ == "__main__":
    main()