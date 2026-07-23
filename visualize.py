"""
visualize.py

Sanity-check script: displays a few "good" training images alongside
a few "defect" test images (transistor category) so you can actually
see what the model will be learning to distinguish, before writing
any model code.

Run:
    python3 visualize.py
"""

import torch
import matplotlib.pyplot as plt

from dataset import MVTecDataset

category_path = "data/mvtec_anomaly_detection/transistor"

train_ds = MVTecDataset(category_path, split="train", mode="anomaly")
test_ds = MVTecDataset(category_path, split="test", mode="anomaly")

# Unnormalize for display (dataset.py applies ImageNet normalization,
# which pushes pixel values out of the [0, 1] range matplotlib expects)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    return (img * std + mean).clamp(0, 1)


fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Row 1: good training images
for i in range(4):
    img, label, mask = train_ds[i]
    axes[0, i].imshow(unnormalize(img).permute(1, 2, 0))
    axes[0, i].set_title("good (train)")
    axes[0, i].axis("off")

# Row 2: defect images from the test set
defect_samples = []
for sample in test_ds:
    if sample[1] == 1:
        defect_samples.append(sample)
    if len(defect_samples) == 4:
        break

for i, (img, label, mask) in enumerate(defect_samples):
    axes[1, i].imshow(unnormalize(img).permute(1, 2, 0))
    axes[1, i].set_title("defect (test)")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()