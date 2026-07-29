"""
patchcore.py

PatchCore feature extraction, memory bank construction, coreset
subsampling, and nearest-neighbor anomaly scoring.

Uses a pretrained ResNet18 purely as a fixed feature extractor (no
training, no backprop) -- unlike the Phase 3 autoencoder, PatchCore
never learns anything from this project's data directly. It just
asks: "does this patch of features look like anything we've seen in
a good image?" Patches far from every stored good patch are
anomalous.

Layers used: layer2 and layer3 of ResNet18 (mid-level features --
layer1 is too low-level/generic, layer4 is too spatially coarse and
loses the local detail defects need). Layer3's coarser feature map is
upsampled to match layer2's spatial resolution before concatenating,
matching the original PatchCore paper's approach.

Coreset note: the real PatchCore paper uses greedy k-center coreset
selection to pick a small, maximally-diverse subset of the memory
bank. That's meaningfully more implementation complexity for a
modest accuracy gain at this dataset's scale (~213 images). This
version uses uniform random subsampling instead -- simpler, and
adequate here since the goal is controlling memory-bank size for
fast nearest-neighbor search, not squeezing out the last bit of
accuracy. Worth knowing as a documented simplification, same spirit
as Phase 2's pooled split or Phase 3's threshold choice.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models


class PatchCoreExtractor:
    """Wraps a pretrained, frozen ResNet18 and extracts concatenated
    layer2+layer3 patch features via forward hooks."""

    def __init__(self, device: torch.device):
        self.device = device
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone.to(device)

        self._features = {}
        self.backbone.layer2.register_forward_hook(self._save_hook("layer2"))
        self.backbone.layer3.register_forward_hook(self._save_hook("layer3"))

    def _save_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def extract_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W).
        Returns (B, P, D) patch embeddings, where P = spatial
        locations in layer2's feature map, D = layer2_channels +
        layer3_channels (128 + 256 = 384 for ResNet18)."""
        images = images.to(self.device)
        self._features.clear()
        self.backbone(images)  # triggers the hooks; final output unused

        f2 = self._features["layer2"]  # (B, 128, H2, W2)
        f3 = self._features["layer3"]  # (B, 256, H3, W3) -- smaller spatial size

        # Local neighborhood aggregation (3x3 average, stride 1, same
        # padding) -- "locally aware patch features" from the
        # PatchCore paper. Each patch embedding now represents a
        # small neighborhood, not a single pixel-sized point.
        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)

        # Upsample layer3 to layer2's spatial resolution so they can
        # be concatenated channel-wise at every spatial location.
        f3_upsampled = F.interpolate(
            f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
        )

        combined = torch.cat([f2, f3_upsampled], dim=1)  # (B, 384, H2, W2)

        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, P, D)
        return patches.cpu()

    def feature_map_size(self, image_size: int = 256) -> int:
        """Returns H2 (== W2), layer2's spatial size for a given
        input resolution -- needed to reshape a flat distance array
        back into a 2D map for heatmap visualization later."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=self.device)
            self._features.clear()
            self.backbone(dummy)
            return self._features["layer2"].shape[-1]


def build_memory_bank(extractor: PatchCoreExtractor, loader) -> np.ndarray:
    """Runs every image in loader through the extractor and stacks
    all patch embeddings from all images into one (N, D) array."""
    all_patches = []
    for images, _labels, _mask_paths in loader:
        patches = extractor.extract_patch_embeddings(images)  # (B, P, D)
        B, P, D = patches.shape
        all_patches.append(patches.reshape(B * P, D).numpy())
    return np.concatenate(all_patches, axis=0)


def coreset_subsample(memory_bank: np.ndarray, fraction: float, seed: int = 42) -> np.ndarray:
    """Uniform random subsampling of the memory bank down to
    `fraction` of its original size. See module docstring for why
    this is used instead of greedy coreset selection."""
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(len(memory_bank) * fraction))
    indices = rng.choice(len(memory_bank), size=n_keep, replace=False)
    return memory_bank[indices]


def score_images(extractor: PatchCoreExtractor, loader, nn_index, top_k_percent: float):
    """Returns (scores, labels) for every image in loader.
    Per-image score = mean of the top-k% highest per-patch nearest-
    neighbor distances -- same top-K spirit as Phase 3's autoencoder
    scoring: a small number of very anomalous patches should drive
    the score, not get averaged away by every normal patch in the
    same image."""
    scores, labels = [], []
    for images, batch_labels, _mask_paths in loader:
        patches = extractor.extract_patch_embeddings(images)  # (B, P, D)
        B, P, D = patches.shape
        flat = patches.reshape(B * P, D).numpy()

        distances, _ = nn_index.kneighbors(flat, n_neighbors=1)
        distances = distances.reshape(B, P)

        k = max(1, int(P * top_k_percent))
        top_k = np.partition(distances, -k, axis=1)[:, -k:]
        image_scores = top_k.mean(axis=1)

        scores.extend(image_scores.tolist())
        labels.extend(batch_labels.tolist())

    return np.array(scores), labels


def score_single_image_map(
    extractor: PatchCoreExtractor,
    image: torch.Tensor,
    nn_index,
    top_k_percent: float,
):
    """Single-image variant for heatmap visualization. Returns a
    tuple of:
        - image_score: the top-K-mean nearest-neighbor distance,
          computed identically to score_images() so the printed
          pass/fail decision matches what the pipeline actually used.
        - distance_map: the full (H2, W2) spatial distance map, for
          the heatmap panel itself.
    top_k_percent must be passed explicitly (rather than assumed)
    so this stays consistent even if TOP_K_PERCENT changes upstream."""
    patches = extractor.extract_patch_embeddings(image.unsqueeze(0))  # (1, P, D)
    P, D = patches.shape[1], patches.shape[2]
    flat = patches.reshape(P, D).numpy()

    distances, _ = nn_index.kneighbors(flat, n_neighbors=1)
    distances = distances.reshape(-1)  # (P,)

    k = max(1, int(P * top_k_percent))
    top_k = np.partition(distances, -k)[-k:]
    image_score = float(top_k.mean())

    side = int(round(P ** 0.5))
    distance_map = distances.reshape(side, side)

    return image_score, distance_map