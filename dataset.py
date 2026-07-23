"""
dataset.py

PyTorch Dataset class for loading the MVTec AD dataset.

Supports two modes:
  - "classification": Phase 2 baseline. Loads train/good (label 0) and
    test/good (label 0) + test/<defect_type> (label 1) images. Used to
    train/eval a simple good-vs-defect CNN classifier.
  - "anomaly": Phase 3 anomaly detection. Train split loads ONLY good
    images (unsupervised — the model never sees a defect during
    training). Test split loads good + defect images with labels, for
    evaluation only.

Directory structure expected (per category), matching MVTec AD:

    data/mvtec_anomaly_detection/<category>/
        train/
            good/
                *.png
        test/
            good/
                *.png
            <defect_type_1>/
                *.png
            <defect_type_2>/
                *.png
            ...
        ground_truth/
            <defect_type_1>/
                *_mask.png
            <defect_type_2>/
                *_mask.png
            ...

__getitem__ always returns a 3-tuple: (image, label, mask_path)
  - image: transformed tensor (C, H, W)
  - label: 0 = good, 1 = defect
  - mask_path: str path to the ground-truth mask if the image is a
    defect sample and a mask exists, else None. Downstream code that
    doesn't need masks (e.g. Phase 2 classifier) just ignores this.
"""

import os
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALID_MODES = ("classification", "anomaly")
VALID_SPLITS = ("train", "test")


def default_transform(image_size: int = 256) -> transforms.Compose:
    """Standard resize + tensor + ImageNet-normalize transform.

    256x256 matches the default input size used in the PatchCore and
    PaDiM reference implementations, and keeps memory/compute
    reasonable on an 8GB M1. Revisit only if Phase 4 localization
    precision demands more resolution (e.g. bump to 384x384).
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class MVTecDataset(Dataset):
    """Unified dataset class for MVTec AD, covering both the Phase 2
    classification setup and the Phase 3 anomaly-detection setup.

    Args:
        root_dir: path to the category folder, e.g.
            "data/mvtec_anomaly_detection/transistor"
        split: "train" or "test"
        mode: "classification" or "anomaly"
            - In "anomaly" mode, the train split is forced to good-only
              regardless of what's on disk (train/ only ever contains
              good images in MVTec anyway, but this makes the
              unsupervised intent explicit).
            - In "test" split, both modes load good + all defect types
              with labels, since evaluation always needs ground truth.
        transform: optional torchvision transform. Defaults to
            default_transform() if not provided.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        mode: str = "anomaly",
        transform: Optional[transforms.Compose] = None,
    ):
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.transform = transform or default_transform()

        self.samples = self._build_sample_list()

    def _build_sample_list(self):
        """Returns a list of (image_path, label, mask_path_or_None)."""
        samples = []
        split_dir = os.path.join(self.root_dir, self.split)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        subfolders = sorted(os.listdir(split_dir))

        for subfolder in subfolders:
            subfolder_path = os.path.join(split_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            is_good = subfolder == "good"
            label = 0 if is_good else 1

            # Train split in anomaly mode: only "good" is expected/used.
            # (MVTec train/ never actually contains defect folders, but
            # this keeps the unsupervised contract explicit and safe.)
            if self.split == "train" and self.mode == "anomaly" and not is_good:
                continue

            for fname in sorted(os.listdir(subfolder_path)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                image_path = os.path.join(subfolder_path, fname)
                mask_path = None

                if not is_good:
                    mask_path = self._find_mask_path(subfolder, fname)

                samples.append((image_path, label, mask_path))

        return samples

    def _find_mask_path(self, defect_type: str, fname: str) -> Optional[str]:
        """Looks up the ground-truth mask for a defect image.

        MVTec naming convention: test/<defect_type>/000.png ->
        ground_truth/<defect_type>/000_mask.png
        """
        name, _ext = os.path.splitext(fname)
        mask_name = f"{name}_mask.png"
        mask_path = os.path.join(self.root_dir, "ground_truth", defect_type, mask_name)
        return mask_path if os.path.isfile(mask_path) else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, int, Optional[str]]:
        image_path, label, mask_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label, mask_path


if __name__ == "__main__":
    # Quick sanity check — run directly to confirm the dataset loads
    # as expected before wiring it into training/visualize scripts.
    category_path = "data/mvtec_anomaly_detection/transistor"

    print("--- anomaly mode, train split (should be good-only) ---")
    train_ds = MVTecDataset(category_path, split="train", mode="anomaly")
    print(f"train samples: {len(train_ds)}")
    img, label, mask = train_ds[0]
    print(f"sample 0 -> image shape: {img.shape}, label: {label}, mask: {mask}")

    print("\n--- anomaly mode, test split (good + defects, for eval) ---")
    test_ds = MVTecDataset(category_path, split="test", mode="anomaly")
    print(f"test samples: {len(test_ds)}")
    n_good = sum(1 for _, l, _ in test_ds.samples if l == 0)
    n_defect = sum(1 for _, l, _ in test_ds.samples if l == 1)
    print(f"good: {n_good}, defect: {n_defect}")

    # find a defect sample with a mask to confirm mask lookup works
    for img, label, mask in test_ds:
        if label == 1:
            print(f"first defect sample -> image shape: {img.shape}, mask: {mask}")
            break