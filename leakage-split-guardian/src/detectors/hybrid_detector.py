"""
hybrid_detector.py
SplitGuardian Hybrid Leakage Detector:
Integrates Perceptual Hashing, Deep Visual Embeddings, and Spatio-Temporal Metadata Constraints
into a unified Graph Affinity Network for robust duplicate and multi-camera leakage detection.
"""

import networkx as nx
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Set

from src.detectors.hash_detector import HashDetector
from src.detectors.embedding_detector import EmbeddingDetector

class HybridLeakageDetector:
    def __init__(
        self,
        hash_threshold: int = 10,
        embedding_threshold: float = 0.82,
        spatiotemporal_window_sec: float = 15.0,
        w_emb: float = 0.50,
        w_hash: float = 0.25,
        w_meta: float = 0.25,
        decision_threshold: float = 0.65
    ):
        self.hash_detector = HashDetector(hash_size=16, hamming_threshold=hash_threshold)
        self.embedding_detector = EmbeddingDetector(sim_threshold=embedding_threshold)
        self.spatiotemporal_window_sec = spatiotemporal_window_sec
        self.w_emb = w_emb
        self.w_hash = w_hash
        self.w_meta = w_meta
        self.decision_threshold = decision_threshold

    def compute_spatiotemporal_affinity(self, sample_a: Dict, sample_b: Dict) -> float:
        """
        Calculates domain constraint affinity between two captures based on
        scene, sequence, camera, and timestamp proximity.
        """
        # If in completely different scenes/routes, affinity is 0
        if sample_a.get("scene_id") != sample_b.get("scene_id"):
            return 0.0

        score = 0.0
        time_diff = abs(sample_a.get("timestamp_sec", 0.0) - sample_b.get("timestamp_sec", 0.0))

        # Check burst temporal proximity
        if sample_a.get("burst_id") == sample_b.get("burst_id"):
            score += 0.6
        elif time_diff <= self.spatiotemporal_window_sec:
            score += max(0.0, 0.5 * (1.0 - time_diff / self.spatiotemporal_window_sec))

        # Check multi-camera concurrent observation
        if sample_a.get("camera_id") != sample_b.get("camera_id"):
            # Multi-camera view of the same event
            if time_diff <= 2.0:
                score += 0.4
        else:
            # Same camera sequential burst
            if time_diff <= 1.0:
                score += 0.4

        return min(1.0, score)

    def analyze_dataset(
        self,
        samples: List[Dict],
        images_lookup: Dict[str, Image.Image]
    ) -> Dict:
        """
        Performs full hybrid leakage detection:
        1. Computes perceptual hashes and embeddings
        2. Fuses visual similarity with spatiotemporal constraints
        3. Constructs graph and identifies connected leakage clusters
        """
        sample_ids = [s["sample_id"] for s in samples]
        n = len(sample_ids)
        metadata_map = {s["sample_id"]: s for s in samples}

        # 1. Feature extraction
        _, hashes = self.hash_detector.detect_pairwise_leakage(samples, images_lookup)
        embeddings = self.embedding_detector.compute_all_embeddings(samples, images_lookup)

        emb_matrix = np.array([embeddings[sid] for sid in sample_ids])
        cos_sim_matrix = np.dot(emb_matrix, emb_matrix.T)

        total_bits = 16 * 16

        # 2. Build affinity graph
        graph = nx.Graph()
        for sid in sample_ids:
            graph.add_node(sid, label=metadata_map[sid]["class_id"])

        predicted_leak_pairs = []
        pair_details = {}

        for i in range(n):
            id_a = sample_ids[i]
            scene_a = metadata_map[id_a].get("scene_id")
            for j in range(i + 1, n):
                id_b = sample_ids[j]
                scene_b = metadata_map[id_b].get("scene_id")

                # If captured in completely different scenes/routes, samples are independent (no physical leakage)
                if scene_a != scene_b:
                    pair_details[(id_a, id_b)] = {
                        "emb_sim": float(cos_sim_matrix[i, j]),
                        "hash_dist": total_bits,
                        "meta_sim": 0.0,
                        "fused_score": 0.0,
                        "is_leak": False
                    }
                    continue

                # Visual embedding similarity
                emb_sim = float(cos_sim_matrix[i, j])

                # Perceptual hash similarity
                h_dist = self.hash_detector.hamming_distance(hashes[id_a], hashes[id_b])
                hash_sim = 1.0 - (h_dist / total_bits)

                # Spatiotemporal metadata affinity
                meta_sim = self.compute_spatiotemporal_affinity(metadata_map[id_a], metadata_map[id_b])

                # Hybrid fusion score
                fused_score = (
                    self.w_emb * emb_sim +
                    self.w_hash * hash_sim +
                    self.w_meta * meta_sim
                )

                # Within same scene, burst frames or multi-camera views exceeding threshold are leaks
                is_leak = (fused_score >= self.decision_threshold) or (emb_sim >= 0.70) or (meta_sim >= 0.60)

                if is_leak:
                    graph.add_edge(id_a, id_b, weight=fused_score)
                    predicted_leak_pairs.append((id_a, id_b, float(fused_score)))

                pair_details[(id_a, id_b)] = {
                    "emb_sim": emb_sim,
                    "hash_dist": h_dist,
                    "meta_sim": meta_sim,
                    "fused_score": fused_score,
                    "is_leak": is_leak
                }

        # 3. Identify Connected Components (Leakage Clusters)
        clusters = list(nx.connected_components(graph))
        cluster_assignment = {}
        for cluster_idx, comp in enumerate(clusters):
            for sid in comp:
                cluster_assignment[sid] = f"cluster_{cluster_idx:03d}"

        return {
            "predicted_leak_pairs": predicted_leak_pairs,
            "clusters": [list(c) for c in clusters],
            "cluster_assignment": cluster_assignment,
            "graph": graph,
            "pair_details": pair_details,
            "embeddings": embeddings,
            "hashes": hashes
        }
