"""
phase2_data.py

Builds the Phase 2 baseline classifier's data splits.

IMPORTANT: this deliberately does NOT follow MVTec's official
train(good-only)/test(good+defect) split. That strict protocol is
for Phase 3's real anomaly detection benchmark. Phase 2 only needs a
sanity-check supervised classifier to build and test the eval
pipeline (loaders, training loop, metrics, checkpointing) -- so here
we pool every labeled image (train/good + test/good + all
test/<defect_type> folders) and do a proper stratified split across
the pool. This gives a real classifier trained on real defects,
without leaking Phase 3's evaluation set (Phase 3 uses the untouched
official split, never this pooled one).

Run directly for a sanity check of split sizes and class balance:
    python3 phase2_data.py
"""

import os
from typing import List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from dataset import default_transform

Sample = Tuple[str, int]  # (image_path, label)


def collect_all_samples(root_dir: str) -> List[Sample]:
    samples = []
    for split in ("train", "test"):
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for subfolder in sorted(os.listdir(split_dir)):
            subfolder_path = os.path.join(split_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            label = 0 if subfolder == "good" else 1
            for fname in sorted(os.listdir(subfolder_path)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    samples.append((os.path.join(subfolder_path, fname), label))
    return samples


class FlatImageDataset(Dataset):
    """Simple (path, label) -> (tensor, label) dataset, used for the
    pooled Phase 2 splits produced by build_splits()."""

    def __init__(self, samples: List[Sample], transform=None):
        self.samples = samples
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def build_splits(
    root_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Returns (train_samples, val_samples, test_samples), stratified
    by label so good/defect ratio is preserved in each split."""
    samples = collect_all_samples(root_dir)
    labels = [s[1] for s in samples]

    train_samples, test_samples = train_test_split(
        samples, test_size=test_size, stratify=labels, random_state=seed
    )

    train_labels = [s[1] for s in train_samples]
    train_samples, val_samples = train_test_split(
        train_samples, test_size=val_size, stratify=train_labels, random_state=seed
    )

    return train_samples, val_samples, test_samples


def _print_split_summary(name: str, split: List[Sample]) -> None:
    n_good = sum(1 for _, label in split if label == 0)
    n_defect = sum(1 for _, label in split if label == 1)
    print(f"{name}: {len(split)} total (good={n_good}, defect={n_defect})")


if __name__ == "__main__":
    # Quick sanity check -- run directly to eyeball split sizes and
    # class balance before wiring these into train.py. With ~313
    # total images split three ways, it's worth confirming no split
    # ends up skewed before trusting any training run on it.
    category_path = "data/mvtec_anomaly_detection/transistor"

    train_split, val_split, test_split = build_splits(category_path)

    _print_split_summary("train", train_split)
    _print_split_summary("val", val_split)
    _print_split_summary("test", test_split)