"""
train.py

Phase 2: trains and evaluates the baseline good-vs-defect classifier.

This is a sanity-check pipeline, not the final anomaly detection
benchmark (see phase2_data.py for why the split is pooled rather than
following MVTec's official protocol). Its job is to prove the
train/eval/checkpoint pipeline works end to end and give a number to
beat once Phase 3's real anomaly detection model is built.

Handles two things that would otherwise silently break results on
this dataset:
  - Class imbalance (train split is ~87% good / 13% defect) via a
    pos_weight in BCEWithLogitsLoss, so the model isn't rewarded for
    just always predicting "good".
  - Best-checkpoint tracking on validation F1, since the val set is
    tiny (25 samples, 3 defect) and the final epoch isn't necessarily
    the best one.

Run:
    python3 train.py
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

from model import build_model
from phase2_data import FlatImageDataset, build_splits

CATEGORY_PATH = "data/mvtec_anomaly_detection/transistor"
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = "models/phase2_baseline_resnet18.pth"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_pos_weight(samples, device) -> torch.Tensor:
    """Weight for the positive (defect) class so BCEWithLogitsLoss
    doesn't let the model coast by always predicting 'good'.

    pos_weight = n_negative / n_positive -- standard formula, makes
    each defect example count proportionally more in the loss.
    """
    labels = [label for _, label in samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    weight = n_neg / max(n_pos, 1)
    print(f"class balance -> good: {n_neg}, defect: {n_pos}, pos_weight: {weight:.2f}")
    return torch.tensor([weight], device=device)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze(1).cpu()
        preds = (probs >= 0.5).long()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return precision, recall, f1, cm


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
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
    device = get_device()
    print(f"Using device: {device}")

    train_samples, val_samples, test_samples = build_splits(CATEGORY_PATH)
    print(f"train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    train_loader = DataLoader(FlatImageDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FlatImageDataset(val_samples), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(FlatImageDataset(test_samples), batch_size=BATCH_SIZE, shuffle=False)

    model = build_model("resnet18", freeze_backbone=True).to(device)

    pos_weight = compute_pos_weight(train_samples, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    best_val_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_precision, val_recall, val_f1, _ = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f} "
            f"- val_precision: {val_precision:.3f} - val_recall: {val_recall:.3f} - val_f1: {val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> new best val_f1 ({best_val_f1:.3f}), checkpoint saved")

    print(f"\nloading best checkpoint (val_f1={best_val_f1:.3f}) for final test evaluation")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    print("\n--- final evaluation on held-out test split ---")
    test_precision, test_recall, test_f1, cm = evaluate(model, test_loader, device)
    print(f"test_precision: {test_precision:.3f}")
    print(f"test_recall: {test_recall:.3f}  (catching real defects -- the one that matters most)")
    print(f"test_f1: {test_f1:.3f}")
    print(f"confusion matrix:\n{cm}")
    print(f"\nbest model weights saved to {CHECKPOINT_PATH}")

    plot_confusion_matrix(cm, title="Phase 2 Baseline - Test Set")


if __name__ == "__main__":
    main()