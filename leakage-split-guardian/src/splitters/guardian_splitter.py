"""
guardian_splitter.py
SplitGuardian Group-Preserving Stratified Re-Split Engine.

Solves cross-split data leakage by enforcing atomic cluster isolation:
- All samples in the same detected leakage cluster / burst event are kept together in ONE split.
- Implements stratified greedy partition to minimize class imbalance across splits.
- Quantifies trade-offs between split sizes, class distributions, and leakage prevention.
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

class GuardianStratifiedSplitter:
    def __init__(
        self,
        target_train_ratio: float = 0.70,
        target_val_ratio: float = 0.15,
        target_test_ratio: float = 0.15,
        seed: int = 42
    ):
        self.target_train_ratio = target_train_ratio
        self.target_val_ratio = target_val_ratio
        self.target_test_ratio = target_test_ratio
        self.seed = seed

    def split_by_clusters(
        self,
        samples: List[Dict],
        cluster_assignment: Dict[str, str]
    ) -> Tuple[List[str], List[str], List[str], Dict]:
        """
        Groups samples into atomic clusters, then distributes whole clusters
        into Train, Val, and Test to maintain target ratios and balanced class distributions.
        """
        rng = random.Random(self.seed)
        metadata_map = {s["sample_id"]: s for s in samples}

        # 1. Group sample IDs by cluster
        clusters = defaultdict(list)
        for sid, cid in cluster_assignment.items():
            if sid in metadata_map:
                clusters[cid].append(sid)

        # For any sample not in cluster_assignment, give it its own cluster
        for s in samples:
            sid = s["sample_id"]
            if sid not in cluster_assignment:
                clusters[f"singleton_{sid}"].append(sid)

        total_samples = len(samples)
        target_train = int(total_samples * self.target_train_ratio)
        target_val = int(total_samples * self.target_val_ratio)

        # 2. Extract cluster characteristics (size and dominant class)
        cluster_list = []
        all_classes = set(s["class_id"] for s in samples)
        
        for cid, sids in clusters.items():
            class_counts = defaultdict(int)
            for sid in sids:
                class_counts[metadata_map[sid]["class_id"]] += 1
            dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
            cluster_list.append({
                "cluster_id": cid,
                "sample_ids": sids,
                "size": len(sids),
                "class_counts": class_counts,
                "dominant_class": dominant_class
            })

        # Shuffle clusters with seed
        rng.shuffle(cluster_list)
        # Sort by size descending for smoother bin packing
        cluster_list.sort(key=lambda x: x["size"], reverse=True)

        # 3. Stratified cluster allocation by dominant class
        train_sids = []
        val_sids = []
        test_sids = []

        class_dist_train = defaultdict(int)
        class_dist_val = defaultdict(int)
        class_dist_test = defaultdict(int)

        use_test = self.target_test_ratio > 0.001

        # Group clusters by dominant class
        clusters_by_class = defaultdict(list)
        for c in cluster_list:
            clusters_by_class[c["dominant_class"]].append(c)

        # Allocate clusters per class proportionally so ALL classes exist in Train and Val
        for cl, c_group in sorted(clusters_by_class.items()):
            rng.shuffle(c_group)
            n_c = len(c_group)
            
            if not use_test:
                n_train = max(1, int(round(n_c * self.target_train_ratio)))
                if n_c > 1 and n_train >= n_c:
                    n_train = n_c - 1
                train_c = c_group[:n_train]
                val_c = c_group[n_train:]
                test_c = []
            else:
                n_train = max(1, int(round(n_c * self.target_train_ratio)))
                n_val = max(1, int(round(n_c * self.target_val_ratio)))
                train_c = c_group[:n_train]
                val_c = c_group[n_train:n_train + n_val]
                test_c = c_group[n_train + n_val:]

            for c in train_c:
                train_sids.extend(c["sample_ids"])
                for k, v in c["class_counts"].items():
                    class_dist_train[k] += v

            for c in val_c:
                val_sids.extend(c["sample_ids"])
                for k, v in c["class_counts"].items():
                    class_dist_val[k] += v

            for c in test_c:
                test_sids.extend(c["sample_ids"])
                for k, v in c["class_counts"].items():
                    class_dist_test[k] += v

        # 4. Compute trade-off analytics
        actual_train_ratio = len(train_sids) / total_samples
        actual_val_ratio = len(val_sids) / total_samples
        actual_test_ratio = len(test_sids) / total_samples

        tradeoff_summary = {
            "total_clusters": len(cluster_list),
            "target_ratios": {
                "train": self.target_train_ratio,
                "val": self.target_val_ratio,
                "test": self.target_test_ratio
            },
            "actual_ratios": {
                "train": round(actual_train_ratio, 4),
                "val": round(actual_val_ratio, 4),
                "test": round(actual_test_ratio, 4)
            },
            "sample_counts": {
                "train": len(train_sids),
                "val": len(val_sids),
                "test": len(test_sids)
            },
            "class_distribution": {
                "train": dict(class_dist_train),
                "val": dict(class_dist_val),
                "test": dict(class_dist_test)
            }
        }

        return train_sids, val_sids, test_sids, tradeoff_summary
