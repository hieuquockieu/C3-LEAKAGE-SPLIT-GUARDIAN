"""
model_trainer.py
Comparative Model Evaluation Engine: Leaky Split vs Clean Split.

Trains identical classification models under:
1. Leaky Split (Naive Random Split)
2. Clean Split (SplitGuardian Group-Preserving Stratified Split)

Evaluates:
- Validation Metric (Accuracy / F1)
- True Independent Holdout Test Metric (Unseen scenes)
- Generalization Gap (Val Metric - Holdout Metric)
Proves mathematically and empirically whether validation scores are "artificially inflated" (đẹp giả).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image

class SimpleObjectClassifier(nn.Module):
    """Simple 2-layer MLP on top of visual feature embeddings for rapid reproducible evaluation."""
    def __init__(self, in_features: int = 256, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_and_evaluate_split(
    split_name: str,
    train_ids: List[str],
    val_ids: List[str],
    holdout_test_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    metadata_map: Dict[str, Dict],
    num_classes: int = 4,
    epochs: int = 40,
    lr: float = 0.01,
    seed: int = 42
) -> Dict:
    """
    Trains a model on train_ids, evaluates on val_ids and holdout_test_ids.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare datasets
    X_train = torch.tensor(np.array([embeddings[sid] for sid in train_ids]), dtype=torch.float32)
    y_train = torch.tensor(np.array([metadata_map[sid]["class_id"] for sid in train_ids]), dtype=torch.long)

    X_val = torch.tensor(np.array([embeddings[sid] for sid in val_ids]), dtype=torch.float32)
    y_val = torch.tensor(np.array([metadata_map[sid]["class_id"] for sid in val_ids]), dtype=torch.long)

    X_test = torch.tensor(np.array([embeddings[sid] for sid in holdout_test_ids]), dtype=torch.float32)
    y_test = torch.tensor(np.array([metadata_map[sid]["class_id"] for sid in holdout_test_ids]), dtype=torch.long)

    model = SimpleObjectClassifier(in_features=256, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        # Train acc
        pred_train = torch.argmax(model(X_train), dim=1)
        train_acc = float((pred_train == y_train).float().mean().item())

        # Val acc
        pred_val = torch.argmax(model(X_val), dim=1)
        val_acc = float((pred_val == y_val).float().mean().item())

        # Holdout test acc
        pred_test = torch.argmax(model(X_test), dim=1)
        holdout_acc = float((pred_test == y_test).float().mean().item())

    # The Generalization Gap (Over-optimism gap)
    val_inflation_gap = round(val_acc - holdout_acc, 4)

    return {
        "split_name": split_name,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "holdout_test_samples": len(holdout_test_ids),
        "train_accuracy": round(train_acc * 100, 2),
        "validation_accuracy": round(val_acc * 100, 2),
        "holdout_test_accuracy": round(holdout_acc * 100, 2),
        "generalization_gap_percent": round(val_inflation_gap * 100, 2),
        "is_artificially_inflated": val_inflation_gap > 0.10
    }
