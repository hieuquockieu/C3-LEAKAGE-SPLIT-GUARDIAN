"""
run_full_benchmark.py
End-to-End Benchmark Execution for Topic C3: Leakage & Split Guardian.

Executes all comparisons, measures primary & secondary metrics,
audits leaky vs clean splits, trains comparative models, and outputs
all results for the README, Web Dashboard, and Presentation Slides.
"""

import os
import sys
import json
import time
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from data.dataset_generator import generate_benchmark_dataset
from src.detectors.hash_detector import HashDetector
from src.detectors.embedding_detector import EmbeddingDetector
from src.detectors.hybrid_detector import HybridLeakageDetector
from src.splitters.naive_splitter import NaiveRandomSplitter
from src.splitters.guardian_splitter import GuardianStratifiedSplitter
from src.evaluation.leakage_metrics import (
    compute_pairwise_detection_metrics,
    compute_leak_types_breakdown,
    compute_split_leakage_audit
)
from src.evaluation.model_trainer import train_and_evaluate_split

def run_benchmark():
    print("=" * 80)
    print("🚀 STARTING SPLITGUARDIAN (TOPIC C3) BENCHMARK SUITE")
    print("=" * 80)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, "dataset_store")

    # Step 1: Generate Benchmark Dataset
    print("\n[Step 1/6] Generating Multi-Camera / Video Burst Dataset...")
    manifest_path, gt_pairs_path = generate_benchmark_dataset(
        output_dir=dataset_dir,
        num_trainval_scenes=14,
        num_test_scenes=4,
        burst_per_scene=3,
        frames_per_burst=4,
        num_cameras=2,
        seed=42
    )

    with open(manifest_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(gt_pairs_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        all_gt_pairs = gt_data["pairs"]

    # Filter out holdout scenes from train/val candidate pool
    trainval_metadata = [m for m in metadata if not m["is_holdout_scene"]]
    holdout_test_metadata = [m for m in metadata if m["is_holdout_scene"]]
    holdout_test_ids = [m["sample_id"] for m in holdout_test_metadata]

    trainval_ids_set = set(m["sample_id"] for m in trainval_metadata)
    gt_pairs = [p for p in all_gt_pairs if p["sample_a"] in trainval_ids_set and p["sample_b"] in trainval_ids_set]

    # Preload images
    print(f"Loading {len(metadata)} images into cache...")
    images_lookup = {}
    for item in metadata:
        path = os.path.join(dataset_dir, item["file_path"])
        images_lookup[item["sample_id"]] = Image.open(path)

    # Step 2: Evaluate Detectors on Primary Metric
    print("\n[Step 2/6] Evaluating Leakage Detection Methods against Ground Truth...")
    
    # 2.1 Baseline: Perceptual Hash Detector
    t0 = time.time()
    hash_detector = HashDetector(hash_size=16, hamming_threshold=10)
    hash_pairs, hashes = hash_detector.detect_pairwise_leakage(trainval_metadata, images_lookup)
    hash_time = time.time() - t0
    hash_metrics = compute_pairwise_detection_metrics(gt_pairs, hash_pairs, len(trainval_metadata))
    hash_breakdown = compute_leak_types_breakdown(gt_pairs, hash_pairs)

    # 2.2 Deep Feature Embeddings
    t0 = time.time()
    emb_detector = EmbeddingDetector(sim_threshold=0.82)
    embeddings = emb_detector.compute_all_embeddings(metadata, images_lookup)
    emb_pairs = emb_detector.detect_pairwise_leakage(trainval_metadata, embeddings)
    emb_time = time.time() - t0
    emb_metrics = compute_pairwise_detection_metrics(gt_pairs, emb_pairs, len(trainval_metadata))
    emb_breakdown = compute_leak_types_breakdown(gt_pairs, emb_pairs)

    # 2.3 Proposed: Hybrid SplitGuardian Detector
    t0 = time.time()
    hybrid_detector = HybridLeakageDetector(
        hash_threshold=10,
        embedding_threshold=0.82,
        spatiotemporal_window_sec=15.0,
        decision_threshold=0.62
    )
    hybrid_analysis = hybrid_detector.analyze_dataset(trainval_metadata, images_lookup)
    hybrid_time = time.time() - t0
    hybrid_pairs = hybrid_analysis["predicted_leak_pairs"]
    hybrid_metrics = compute_pairwise_detection_metrics(gt_pairs, hybrid_pairs, len(trainval_metadata))
    hybrid_breakdown = compute_leak_types_breakdown(gt_pairs, hybrid_pairs)

    print("\n" + "-" * 75)
    print(f"{'Method':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Time (s)':<8}")
    print("-" * 75)
    print(f"{'Baseline: Perceptual Hash':<28} | {hash_metrics['precision']:<10.4f} | {hash_metrics['recall']:<10.4f} | {hash_metrics['f1']:<10.4f} | {hash_time:<8.2f}")
    print(f"{'Deep Visual Embeddings':<28} | {emb_metrics['precision']:<10.4f} | {emb_metrics['recall']:<10.4f} | {emb_metrics['f1']:<10.4f} | {emb_time:<8.2f}")
    print(f"{'SplitGuardian (Hybrid Graph)':<28} | {hybrid_metrics['precision']:<10.4f} | {hybrid_metrics['recall']:<10.4f} | {hybrid_metrics['f1']:<10.4f} | {hybrid_time:<8.2f}")
    print("-" * 75)

    # Step 3: Compare Splitting Strategies
    print("\n[Step 3/6] Executing Splitting Strategies (Naive Leaky vs Clean Re-Split)...")
    
    # 3.1 Naive Random Split (Causes Leakage)
    naive_splitter = NaiveRandomSplitter(train_ratio=0.75, val_ratio=0.25, seed=42)
    leaky_train_ids, leaky_val_ids, _ = naive_splitter.split(trainval_metadata)

    # 3.2 SplitGuardian Group-Preserving Stratified Split (Isolates Clusters)
    guardian_splitter = GuardianStratifiedSplitter(target_train_ratio=0.75, target_val_ratio=0.25, target_test_ratio=0.0, seed=42)
    clean_train_ids, clean_val_ids, _, tradeoff_info = guardian_splitter.split_by_clusters(
        trainval_metadata,
        hybrid_analysis["cluster_assignment"]
    )

    # Step 4: Audit Splits for Residual Leakage & Cross-Split Similarity
    print("\n[Step 4/6] Auditing Splits for Cross-Split Leakage & Secondary Metrics...")
    leaky_audit = compute_split_leakage_audit(leaky_train_ids, leaky_val_ids, gt_pairs, embeddings)
    clean_audit = compute_split_leakage_audit(clean_train_ids, clean_val_ids, gt_pairs, embeddings)

    print("\n" + "-" * 85)
    print(f"{'Split Strategy':<25} | {'Cross Leaked Pairs':<18} | {'Max Cosine Sim':<15} | {'Mean Cosine Sim':<15}")
    print("-" * 85)
    print(f"{'Naive Random Split':<25} | {leaky_audit['cross_leaked_pair_count']:<18} | {leaky_audit['cross_split_similarity']['max_cosine_sim']:<15.4f} | {leaky_audit['cross_split_similarity']['mean_cosine_sim']:<15.4f}")
    print(f"{'SplitGuardian Re-Split':<25} | {clean_audit['cross_leaked_pair_count']:<18} | {clean_audit['cross_split_similarity']['max_cosine_sim']:<15.4f} | {clean_audit['cross_split_similarity']['mean_cosine_sim']:<15.4f}")
    print("-" * 85)

    # Step 5: Model Training & Generalization Gap Demonstration ("Đẹp giả" vs "Đáng tin")
    print("\n[Step 5/6] Training Comparative Models on Leaky vs Clean Splits...")
    metadata_map = {m["sample_id"]: m for m in metadata}

    eval_leaky = train_and_evaluate_split(
        split_name="Leaky Random Split",
        train_ids=leaky_train_ids,
        val_ids=leaky_val_ids,
        holdout_test_ids=holdout_test_ids,
        embeddings=embeddings,
        metadata_map=metadata_map,
        epochs=45,
        seed=42
    )

    eval_clean = train_and_evaluate_split(
        split_name="SplitGuardian Clean Split",
        train_ids=clean_train_ids,
        val_ids=clean_val_ids,
        holdout_test_ids=holdout_test_ids,
        embeddings=embeddings,
        metadata_map=metadata_map,
        epochs=45,
        seed=42
    )

    print("\n" + "=" * 90)
    print(f"{'Split Policy':<25} | {'Val Acc (%)':<12} | {'Holdout Test Acc (%)':<20} | {'Gen Gap (%)':<12} | {'Reliability Status':<18}")
    print("=" * 90)
    print(f"{eval_leaky['split_name']:<25} | {eval_leaky['validation_accuracy']:<12.1f} | {eval_leaky['holdout_test_accuracy']:<20.1f} | {eval_leaky['generalization_gap_percent']:<12.1f} | {'🔴 ĐẸP GIẢ (Overfit)':<18}")
    print(f"{eval_clean['split_name']:<25} | {eval_clean['validation_accuracy']:<12.1f} | {eval_clean['holdout_test_accuracy']:<20.1f} | {eval_clean['generalization_gap_percent']:<12.1f} | {'🟢 ĐÁNG TIN (Honest)':<18}")
    print("=" * 90)

    # Step 6: Export Results Report for Web Dashboard and Docs
    print("\n[Step 6/6] Exporting Comprehensive Results for Web Dashboard...")
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_summary": {
            "total_samples": len(metadata),
            "trainval_samples": len(trainval_metadata),
            "holdout_test_samples": len(holdout_test_metadata),
            "num_scenes": 18,
            "num_classes": 4,
            "total_gt_leak_pairs": len(gt_pairs)
        },
        "primary_metrics": {
            "perceptual_hash": {**hash_metrics, "runtime_sec": round(hash_time, 3), "breakdown": hash_breakdown},
            "deep_embeddings": {**emb_metrics, "runtime_sec": round(emb_time, 3), "breakdown": emb_breakdown},
            "splitguardian_hybrid": {**hybrid_metrics, "runtime_sec": round(hybrid_time, 3), "breakdown": hybrid_breakdown}
        },
        "split_audits": {
            "leaky_random_split": leaky_audit,
            "splitguardian_clean_split": clean_audit
        },
        "tradeoff_analysis": tradeoff_info,
        "model_evaluation": {
            "leaky_split": eval_leaky,
            "clean_split": eval_clean,
            "gap_difference": round(eval_leaky['generalization_gap_percent'] - eval_clean['generalization_gap_percent'], 2)
        }
    }

    # Write report to web_dashboard and reports directory
    os.makedirs(os.path.join(base_dir, "web_dashboard"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "reports"), exist_ok=True)

    with open(os.path.join(base_dir, "reports", "clean_split.json"), "w", encoding="utf-8") as f:
        json.dump({"train": clean_train_ids, "val": clean_val_ids}, f, indent=2)

    with open(os.path.join(base_dir, "reports", "leaky_split.json"), "w", encoding="utf-8") as f:
        json.dump({"train": leaky_train_ids, "val": leaky_val_ids}, f, indent=2)

    with open(os.path.join(base_dir, "web_dashboard", "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    with open(os.path.join(base_dir, "reports", "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n✅ All benchmark artifacts successfully generated and exported.")
    print("=" * 80)
    return report_data

if __name__ == "__main__":
    run_benchmark()
