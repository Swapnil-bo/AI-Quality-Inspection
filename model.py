"""
model.py

Builds a transfer-learning classifier for the Phase 2 baseline.

Loads a pretrained backbone (ResNet18 by default) and replaces the
final layer with a single-output binary head. Freezes the backbone
by default since ~313 total images is tiny by deep learning
standards -- only the new final layer gets trained.

If the frozen-backbone baseline underperforms (ImageNet features
weren't trained to notice things like bent metal leads), set
unfreeze_last_block=True to also fine-tune the backbone's last block
-- a middle ground between "fully frozen" and "fine-tune everything",
which usually helps most when the dataset is this small.
"""

import torch.nn as nn
from torchvision import models


def build_model(
    architecture: str = "resnet18",
    freeze_backbone: bool = True,
    unfreeze_last_block: bool = False,
) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            if unfreeze_last_block:
                for param in model.layer4.parameters():
                    param.requires_grad = True

        model.fc = nn.Linear(model.fc.in_features, 1)

    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            if unfreeze_last_block:
                for param in model.features[-1].parameters():
                    param.requires_grad = True

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)

    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    return model