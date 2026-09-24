"""
split_guardian_gate.py
Automated Production CI/CD Gate for Cross-Split Leakage Prevention.

Can be run as a pre-commit or pre-training step in ML pipelines (e.g. GitHub Actions, Kubeflow, Airflow).
Exits with code 0 if split passes production safety checks, or code 1 if cross-split leakage is detected.
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.detectors.hybrid_detector import HybridLeakageDetector
from src.evaluation.leakage_metrics import compute_split_leakage_audit

def run_ci_gate(manifest_path: str, policy_path: str, split_assignment_path: str = None) -> int:
    print("=" * 70)
    print(" [SplitGuardian] Starting Production CI/CD Leakage Security Gate")
    print("=" * 70)

    # 1. Load policy configuration
    if not os.path.exists(policy_path):
        print(f"[Error] Policy config not found at {policy_path}")
        return 1

    with open(policy_path, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    # 2. Load dataset manifest
    if not os.path.exists(manifest_path):
        print(f"[Error] Manifest not found at {manifest_path}")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    base_dir = os.path.dirname(manifest_path)
    images_lookup = {}
    for item in metadata:
        img_path = os.path.join(base_dir, item["file_path"])
        if os.path.exists(img_path):
            images_lookup[item["sample_id"]] = Image.open(img_path)

    print(f" Loaded {len(metadata)} samples and {len(images_lookup)} image assets.")

    # 3. Load or simulate split assignment
    if split_assignment_path and os.path.exists(split_assignment_path):
        with open(split_assignment_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
            train_ids = splits.get("train", [])
            val_ids = splits.get("val", [])
    else:
        print("[Warning] No explicit split provided. Evaluating standard naive random split as demo...")
        from src.splitters.naive_splitter import NaiveRandomSplitter
        splitter = NaiveRandomSplitter(seed=42)
        train_ids, val_ids, _ = splitter.split(metadata)

    # 4. Run Hybrid Leakage Detection
    detector = HybridLeakageDetector(
        hash_threshold=policy["detection"]["phash_max_hamming_dist"],
        embedding_threshold=policy["detection"]["embedding_sim_threshold"],
        spatiotemporal_window_sec=policy["detection"]["spatiotemporal_window_sec"]
    )
    analysis = detector.analyze_dataset(metadata, images_lookup)

    # 5. Load Ground Truth pairs if available for audit
    gt_pairs_path = os.path.join(base_dir, "ground_truth_leakage_pairs.json")
    gt_pairs = []
    if os.path.exists(gt_pairs_path):
        with open(gt_pairs_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            gt_pairs = gt_data.get("pairs", [])

    # 6. Audit Split
    audit_res = compute_split_leakage_audit(train_ids, val_ids, gt_pairs, analysis["embeddings"])

    max_sim = audit_res["cross_split_similarity"]["max_cosine_sim"]
    sim_threshold = policy["ci_gate"]["max_allowed_cross_split_sim"]
    leaked_pairs = audit_res["cross_leaked_pair_count"]
    max_allowed_leaks = policy["ci_gate"]["max_allowed_leaked_pairs"]

    print("\n[Gate Results Summary]")
    print(f" - Train Samples: {audit_res['num_train_samples']}")
    print(f" - Val Samples:   {audit_res['num_val_samples']}")
    print(f" - Cross-Split Leaked Pairs: {leaked_pairs} (Max Allowed: {max_allowed_leaks})")
    print(f" - Max Cross-Split Cosine Sim: {max_sim:.4f} (Threshold: {sim_threshold:.4f})")
    print(f" - Mean Cross-Split Cosine Sim: {audit_res['cross_split_similarity']['mean_cosine_sim']:.4f}")

    # Decision logic
    passed = (leaked_pairs <= max_allowed_leaks) and (max_sim <= sim_threshold)

    report = {
        "status": "PASSED" if passed else "REJECTED",
        "decision": "DEPLOY" if passed else "REWORK",
        "audit": audit_res,
        "policy_thresholds": {
            "max_allowed_cross_split_sim": sim_threshold,
            "max_allowed_leaked_pairs": max_allowed_leaks
        }
    }

    os.makedirs("reports", exist_ok=True)
    report_file = policy["ci_gate"].get("output_report_path", "reports/ci_gate_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n Detailed security report written to: {report_file}")

    if passed:
        print("\n [GATE STATUS: PASSED] Dataset splits satisfy production isolation standards.")
        print(" Action: Safe to proceed with production training pipeline.")
        return 0
    else:
        print("\n [GATE STATUS: FAILED / REJECTED] Cross-split leakage detected!")
        print(" Danger: Training on this split will cause artificially inflated validation metrics.")
        print(" Action: Block build. Run SplitGuardian re-splitter before training.")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SplitGuardian CI/CD Security Gate")
    parser.add_argument("--manifest", default="dataset_store/metadata.json", help="Path to manifest json")
    parser.add_argument("--policy", default="configs/guardian_policy.yaml", help="Path to policy yaml")
    parser.add_argument("--split", default=None, help="Path to split assignment json")
    args = parser.parse_args()

    exit_code = run_ci_gate(args.manifest, args.policy, args.split)
    sys.exit(exit_code)
