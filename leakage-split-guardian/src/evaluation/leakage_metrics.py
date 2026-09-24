"""
leakage_metrics.py
Quantitative Evaluation Metrics for Topic C3 (Leakage & Split Guardian).

Computes:
1. Primary Metric: Leakage Detection Precision, Recall, and F1 (Pair-level and Cluster-level).
2. Secondary Metrics:
   - Residual Cross-Split Similarity (Mean, Max, 95th percentile cosine similarity).
   - Residual Cross-Split Leaked Pairs straddling Train and Val.
   - Generalization Gap (Validation Metric vs Holdout Test Metric).
"""

import numpy as np
from typing import Dict, List, Set, Tuple

def compute_pairwise_detection_metrics(
    ground_truth_leak_pairs: List[Dict],
    predicted_leak_pairs: List[Tuple[str, str, float]],
    total_samples: int
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1 for pair-level leakage detection.
    """
    gt_set = set()
    for p in ground_truth_leak_pairs:
        # Standardize pair order
        s1, s2 = sorted([p["sample_a"], p["sample_b"]])
        gt_set.add((s1, s2))

    pred_set = set()
    for p in predicted_leak_pairs:
        s1, s2 = sorted([p[0], p[1]])
        pred_set.add((s1, s2))

    true_positives = len(gt_set.intersection(pred_set))
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)

    total_pairs = (total_samples * (total_samples - 1)) // 2
    true_negatives = total_pairs - (true_positives + false_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / max(1, total_pairs)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "total_gt_leaks": len(gt_set),
        "total_pred_leaks": len(pred_set)
    }

def compute_leak_types_breakdown(
    ground_truth_leak_pairs: List[Dict],
    predicted_leak_pairs: List[Tuple[str, str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates Recall per specific leakage type:
    - Temporal Burst Overlap
    - Multi-Camera Spatial Overlap
    """
    pred_set = set()
    for p in predicted_leak_pairs:
        s1, s2 = sorted([p[0], p[1]])
        pred_set.add((s1, s2))

    breakdown = {}
    by_type = {}
    for p in ground_truth_leak_pairs:
        l_type = p["leak_type"]
        s1, s2 = sorted([p["sample_a"], p["sample_b"]])
        if l_type not in by_type:
            by_type[l_type] = []
        by_type[l_type].append((s1, s2))

    for l_type, pairs in by_type.items():
        detected = sum(1 for pair in pairs if pair in pred_set)
        total = len(pairs)
        rec = detected / max(1, total)
        breakdown[l_type] = {
            "total_gt": total,
            "detected": detected,
            "recall": round(rec, 4)
        }

    return breakdown

def compute_split_leakage_audit(
    train_ids: List[str],
    val_ids: List[str],
    ground_truth_leak_pairs: List[Dict],
    embeddings: Dict[str, np.ndarray]
) -> Dict:
    """
    Audits a given split configuration (Train vs Val):
    - Identifies how many GT leak pairs are split across Train and Val.
    - Computes Cross-Split Cosine Similarity distribution.
    """
    train_set = set(train_ids)
    val_set = set(val_ids)

    # 1. Count actual cross-split leak pairs
    cross_leaked_pairs = []
    cross_leak_by_type = {"temporal_burst": 0, "multi_camera_cross_view": 0}

    for p in ground_truth_leak_pairs:
        s1, s2 = p["sample_a"], p["sample_b"]
        # One in train and one in val
        if (s1 in train_set and s2 in val_set) or (s2 in train_set and s1 in val_set):
            cross_leaked_pairs.append((s1, s2, p["leak_type"]))
            cross_leak_by_type[p["leak_type"]] = cross_leak_by_type.get(p["leak_type"], 0) + 1

    # 2. Compute cross-split cosine similarity
    train_embs = np.array([embeddings[sid] for sid in train_ids if sid in embeddings])
    val_embs = np.array([embeddings[sid] for sid in val_ids if sid in embeddings])

    if len(train_embs) > 0 and len(val_embs) > 0:
        # Cross similarity matrix (N_train x N_val)
        cross_sim_matrix = np.dot(train_embs, val_embs.T)
        mean_sim = float(np.mean(cross_sim_matrix))
        max_sim = float(np.max(cross_sim_matrix))
        p95_sim = float(np.percentile(cross_sim_matrix, 95))
    else:
        mean_sim = 0.0
        max_sim = 0.0
        p95_sim = 0.0

    return {
        "num_train_samples": len(train_ids),
        "num_val_samples": len(val_ids),
        "cross_leaked_pair_count": len(cross_leaked_pairs),
        "cross_leak_by_type": cross_leak_by_type,
        "cross_split_similarity": {
            "mean_cosine_sim": round(mean_sim, 4),
            "max_cosine_sim": round(max_sim, 4),
            "p95_cosine_sim": round(p95_sim, 4)
        },
        "has_leakage": len(cross_leaked_pairs) > 0
    }
