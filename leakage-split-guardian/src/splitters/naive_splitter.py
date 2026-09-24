"""
naive_splitter.py
Baseline Naive Random Splitter.

Demonstrates the default practice: independent and identically distributed (i.i.d.)
sample-level random splitting, which ignores video burst sequences and multi-camera captures,
causing massive cross-split data leakage.
"""

import random
from typing import Dict, List, Tuple

class NaiveRandomSplitter:
    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    def split(self, samples: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
        """
        Randomly assigns samples into train, val, and test splits regardless of clusters.
        Returns:
            train_ids, val_ids, test_ids
        """
        rng = random.Random(self.seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train:n_train + n_val]
        test_samples = shuffled[n_train + n_val:]

        train_ids = [s["sample_id"] for s in train_samples]
        val_ids = [s["sample_id"] for s in val_samples]
        test_ids = [s["sample_id"] for s in test_samples]

        return train_ids, val_ids, test_ids
