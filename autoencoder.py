"""
autoencoder.py

Simple convolutional autoencoder for Phase 3 anomaly detection.

Trained only on train/good images (unsupervised). The bottleneck is
kept fairly compressed (256 x 8 x 8 from a 256x256x3 input) so the
model has to learn a general "what good looks like" representation
rather than trivially memorizing individual training images.
"""

import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),      # 256 -> 128
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),     # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),   # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 32 -> 16
            nn.Conv2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 16 -> 8
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 8 -> 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),    # 32 -> 64
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),     # 64 -> 128
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # 128 -> 256, no final activation (target isn't in [0,1] --
        )                                        # dataset.py applies ImageNet normalization)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon