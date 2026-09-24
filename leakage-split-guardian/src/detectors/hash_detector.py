"""
hash_detector.py
Baseline Perceptual Hashing Leakage Detector (aHash, dHash, pHash).

Fast, metadata-blind baseline for duplicate and near-duplicate detection.
Computes Hamming distance between image hashes.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

class HashDetector:
    def __init__(self, hash_size: int = 16, hamming_threshold: int = 10):
        """
        Args:
            hash_size: Size of the hash grid (e.g. 16 -> 256 bits).
            hamming_threshold: Max Hamming distance to classify as duplicate/leak.
        """
        self.hash_size = hash_size
        self.hamming_threshold = hamming_threshold

    def compute_dhash(self, image: Image.Image) -> np.ndarray:
        """
        Computes Difference Hash (dHash) using gradient comparison.
        """
        # Resize to (hash_size + 1, hash_size) in grayscale
        resized = image.convert("L").resize((self.hash_size + 1, self.hash_size), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)
        # Compare adjacent horizontal pixels
        diff = arr[:, 1:] > arr[:, :-1]
        return diff.flatten()

    def compute_phash(self, image: Image.Image) -> np.ndarray:
        """
        Computes Perceptual Hash (pHash) using discrete cosine transform approximation.
        """
        resized = image.convert("L").resize((self.hash_size, self.hash_size), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)
        mean_val = np.mean(arr)
        return (arr > mean_val).flatten()

    def hamming_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> int:
        """Computes bitwise Hamming distance."""
        return int(np.sum(hash1 != hash2))

    def detect_pairwise_leakage(
        self,
        samples: List[Dict],
        images_lookup: Dict[str, Image.Image]
    ) -> Tuple[List[Tuple[str, str, float]], Dict[str, np.ndarray]]:
        """
        Computes all pairwise hashes and flags pairs with distance <= hamming_threshold.
        Returns:
            predicted_leak_pairs: List of (id_a, id_b, normalized_similarity)
            hashes: Dict of sample_id -> hash_vector
        """
        hashes = {}
        for sample in samples:
            sid = sample["sample_id"]
            img = images_lookup[sid]
            hashes[sid] = self.compute_dhash(img)

        sample_ids = [s["sample_id"] for s in samples]
        n = len(sample_ids)
        total_bits = self.hash_size * self.hash_size
        predicted_leak_pairs = []

        for i in range(n):
            id_a = sample_ids[i]
            h_a = hashes[id_a]
            for j in range(i + 1, n):
                id_b = sample_ids[j]
                h_b = hashes[id_b]
                dist = self.hamming_distance(h_a, h_b)
                if dist <= self.hamming_threshold:
                    sim = 1.0 - (dist / total_bits)
                    predicted_leak_pairs.append((id_a, id_b, float(sim)))

        return predicted_leak_pairs, hashes
