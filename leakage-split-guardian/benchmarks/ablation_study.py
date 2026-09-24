"""
ablation_study.py
Ablation Studies, Stress Testing, and Cost/Runtime Analysis for SplitGuardian.

Conducts:
1. Multi-modal component ablation (Hash only vs Embedding only vs Metadata vs Hybrid)
2. Decision Threshold sensitivity sweep (Precision vs Recall vs F1)
3. Computational runtime and complexity scaling analysis
"""

import os
import sys
import json
import time
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.detectors.hash_detector import HashDetector
from src.detectors.embedding_detector import EmbeddingDetector
from src.detectors.hybrid_detector import HybridLeakageDetector
from src.evaluation.leakage_metrics import compute_pairwise_detection_metrics

def run_ablation_studies():
    print("=" * 80)
    print("🔬 RUNNING SPLITGUARDIAN ABLATION & STRESS TEST SUITE")
    print("=" * 80)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, "dataset_store")
    manifest_path = os.path.join(dataset_dir, "metadata.json")
    gt_pairs_path = os.path.join(dataset_dir, "ground_truth_leakage_pairs.json")

    with open(manifest_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(gt_pairs_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        all_gt_pairs = gt_data["pairs"]

    trainval = [m for m in metadata if not m["is_holdout_scene"]]
    trainval_ids = set(m["sample_id"] for m in trainval)
    gt_pairs = [p for p in all_gt_pairs if p["sample_a"] in trainval_ids and p["sample_b"] in trainval_ids]

    images = {m["sample_id"]: Image.open(os.path.join(dataset_dir, m["file_path"])) for m in trainval}

    # -------------------------------------------------------------
    # 1. Modality Ablation (Single cue vs Fused)
    # -------------------------------------------------------------
    print("\n[Ablation 1/3] Modality Ablation Study...")
    modalities = {
        "Perceptual Hash Only": {"w_emb": 0.0, "w_hash": 1.0, "w_meta": 0.0, "thresh": 0.40},
        "Deep Embeddings Only": {"w_emb": 1.0, "w_hash": 0.0, "w_meta": 0.0, "thresh": 0.40},
        "Metadata Spatiotemporal Only": {"w_emb": 0.0, "w_hash": 0.0, "w_meta": 1.0, "thresh": 0.40},
        "SplitGuardian Hybrid (Proposed)": {"w_emb": 0.50, "w_hash": 0.25, "w_meta": 0.25, "thresh": 0.55}
    }

    modality_results = []
    for name, cfg in modalities.items():
        t0 = time.time()
        det = HybridLeakageDetector(
            w_emb=cfg["w_emb"],
            w_hash=cfg["w_hash"],
            w_meta=cfg["w_meta"],
            decision_threshold=cfg["thresh"]
        )
        res = det.analyze_dataset(trainval, images)
        dur = time.time() - t0
        metrics = compute_pairwise_detection_metrics(gt_pairs, res["predicted_leak_pairs"], len(trainval))
        modality_results.append({
            "configuration": name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "clusters_formed": len(res["clusters"]),
            "runtime_sec": round(dur, 3)
        })

    print("-" * 80)
    print(f"{'Modality Configuration':<35} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Clusters':<8}")
    print("-" * 80)
    for m in modality_results:
        print(f"{m['configuration']:<35} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['clusters_formed']:<8}")
    print("-" * 80)

    # -------------------------------------------------------------
    # 2. Decision Threshold Sensitivity Sweep
    # -------------------------------------------------------------
    print("\n[Ablation 2/3] Decision Threshold Sensitivity Sweep...")
    thresholds = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    threshold_results = []

    for th in thresholds:
        det = HybridLeakageDetector(decision_threshold=th)
        res = det.analyze_dataset(trainval, images)
        metrics = compute_pairwise_detection_metrics(gt_pairs, res["predicted_leak_pairs"], len(trainval))
        threshold_results.append({
            "threshold": th,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "clusters": len(res["clusters"])
        })

    print("-" * 70)
    print(f"{'Threshold (theta)':<18} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Clusters':<8}")
    print("-" * 70)
    for t in threshold_results:
        print(f"{t['threshold']:<18.2f} | {t['precision']:<12.4f} | {t['recall']:<12.4f} | {t['f1_score']:<12.4f} | {t['clusters']:<8}")
    print("-" * 70)

    # -------------------------------------------------------------
    # 3. Runtime & Scalability Analysis
    # -------------------------------------------------------------
    print("\n[Ablation 3/3] Computational Runtime & Complexity Analysis...")
    sample_sizes = [50, 100, 200, len(trainval)]
    scalability_results = []

    for n_sub in sample_sizes:
        sub_samples = trainval[:n_sub]
        sub_imgs = {s["sample_id"]: images[s["sample_id"]] for s in sub_samples}

        t_start = time.time()
        det = HybridLeakageDetector()
        det.analyze_dataset(sub_samples, sub_imgs)
        elapsed = time.time() - t_start

        throughput = n_sub / max(1e-6, elapsed)
        scalability_results.append({
            "sample_count": n_sub,
            "pairwise_comparisons": (n_sub * (n_sub - 1)) // 2,
            "total_time_sec": round(elapsed, 3),
            "throughput_fps": round(throughput, 1)
        })

    print("-" * 75)
    print(f"{'Samples (N)':<12} | {'Pairwise Checks':<18} | {'Time (sec)':<15} | {'Throughput (img/s)':<18}")
    print("-" * 75)
    for s in scalability_results:
        print(f"{s['sample_count']:<12} | {s['pairwise_comparisons']:<18} | {s['total_time_sec']:<15.3f} | {s['throughput_fps']:<18.1f}")
    print("-" * 75)

    # Export report
    output_path = os.path.join(base_dir, "reports", "ablation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "modality_ablation": modality_results,
            "threshold_sweep": threshold_results,
            "scalability_analysis": scalability_results
        }, f, indent=2)

    print(f"\n✅ Ablation report written to: {output_path}")
    return output_path

if __name__ == "__main__":
    run_ablation_studies()
