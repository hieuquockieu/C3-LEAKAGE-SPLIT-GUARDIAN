"""
generate_notebook.py
Generates a complete, clean, runnable Jupyter Notebook (.ipynb) for Topic C3.
"""

import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Topic C3: Leakage & Split Guardian\n",
            "### Cross-Split Leakage Detection, Severity Quantification & Graph-Stratified Re-Splitting\n",
            "\n",
            "This notebook provides a complete, runnable reproduction of the **SplitGuardian** engine:\n",
            "1. **Dataset Generation:** Generates multi-camera, video burst visual dataset with known Ground Truth.\n",
            "2. **Dual-Leakage Demonstration:** Temporal burst near-duplicates and multi-camera viewpoint shifts.\n",
            "3. **Primary Metric:** Leakage Detection F1 against baselines.\n",
            "4. **Re-Splitting:** Naive random split vs Group-Preserving Stratified Re-Split.\n",
            "5. **Comparative Model Evaluation:** Proving whether validation metrics are 'artificially inflated' (*đẹp giả*)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Setup and Imports\n",
            "import os\n",
            "import sys\n",
            "import json\n",
            "import torch\n",
            "from PIL import Image\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Add parent directory to path\n",
            "sys.path.insert(0, os.path.abspath('..'))\n",
            "\n",
            "from data.dataset_generator import generate_benchmark_dataset\n",
            "from src.detectors.hash_detector import HashDetector\n",
            "from src.detectors.embedding_detector import EmbeddingDetector\n",
            "from src.detectors.hybrid_detector import HybridLeakageDetector\n",
            "from src.splitters.naive_splitter import NaiveRandomSplitter\n",
            "from src.splitters.guardian_splitter import GuardianStratifiedSplitter\n",
            "from src.evaluation.leakage_metrics import compute_pairwise_detection_metrics, compute_split_leakage_audit\n",
            "from src.evaluation.model_trainer import train_and_evaluate_split\n",
            "\n",
            "print('PyTorch Version:', torch.__version__)\n",
            "print('All modules successfully imported!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Benchmark Dataset Generation with Dual-Leakage Ground Truth\n",
            "We generate a synthetic dataset with:\n",
            "- 14 Train/Val scenes and 4 Independent Holdout Test scenes\n",
            "- 4 classes (sedan, truck, motorcycle, emergency vehicle)\n",
            "- 2 distinct forms of cross-split leakage: **Temporal Burst** and **Multi-Camera Viewpoint Overlap**\n",
            "- Exhaustive ground truth manifest with pairwise labels."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 2: Generate dataset\n",
            "dataset_dir = os.path.abspath('../dataset_store')\n",
            "manifest_path, gt_pairs_path = generate_benchmark_dataset(\n",
            "    output_dir=dataset_dir,\n",
            "    num_trainval_scenes=14,\n",
            "    num_test_scenes=4,\n",
            "    burst_per_scene=3,\n",
            "    frames_per_burst=4,\n",
            "    num_cameras=2,\n",
            "    seed=42\n",
            ")\n",
            "\n",
            "with open(manifest_path, 'r') as f:\n",
            "    metadata = json.load(f)\n",
            "with open(gt_pairs_path, 'r') as f:\n",
            "    gt_pairs = json.load(f)['pairs']\n",
            "\n",
            "trainval = [m for m in metadata if not m['is_holdout_scene']]\n",
            "holdout = [m for m in metadata if m['is_holdout_scene']]\n",
            "trainval_ids = set(m['sample_id'] for m in trainval)\n",
            "trainval_gt = [p for p in gt_pairs if p['sample_a'] in trainval_ids and p['sample_b'] in trainval_ids]\n",
            "\n",
            "print(f'Total samples: {len(metadata)}')\n",
            "print(f'Train/Val pool: {len(trainval)} samples')\n",
            "print(f'Holdout Test set: {len(holdout)} samples')\n",
            "print(f'Ground Truth leak pairs in Train/Val pool: {len(trainval_gt)}')\n",
            "\n",
            "# Load images\n",
            "images_lookup = {m['sample_id']: Image.open(os.path.join(dataset_dir, m['file_path'])) for m in metadata}"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Leakage Detection Methods & Primary Metric Evaluation\n",
            "We evaluate three detection strategies on the Primary Metric (**Leakage Detection F1**):\n",
            "1. **Baseline 1:** Perceptual Hash (dHash/pHash with Hamming distance threshold)\n",
            "2. **Baseline 2:** Deep CNN Visual Embeddings (Cosine similarity threshold)\n",
            "3. **Proposed:** SplitGuardian Hybrid Multimodal Graph Detector"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 3: Run Detectors\n",
            "# 3.1 Perceptual Hash Baseline\n",
            "hash_det = HashDetector(hash_size=16, hamming_threshold=10)\n",
            "hash_pairs, _ = hash_det.detect_pairwise_leakage(trainval, images_lookup)\n",
            "hash_metrics = compute_pairwise_detection_metrics(trainval_gt, hash_pairs, len(trainval))\n",
            "\n",
            "# 3.2 Deep Visual Embeddings\n",
            "emb_det = EmbeddingDetector(sim_threshold=0.82)\n",
            "embeddings = emb_det.compute_all_embeddings(metadata, images_lookup)\n",
            "emb_pairs = emb_det.detect_pairwise_leakage(trainval, embeddings)\n",
            "emb_metrics = compute_pairwise_detection_metrics(trainval_gt, emb_pairs, len(trainval))\n",
            "\n",
            "# 3.3 SplitGuardian Hybrid Graph Detector\n",
            "hybrid_det = HybridLeakageDetector()\n",
            "hybrid_analysis = hybrid_det.analyze_dataset(trainval, images_lookup)\n",
            "hybrid_metrics = compute_pairwise_detection_metrics(trainval_gt, hybrid_analysis['predicted_leak_pairs'], len(trainval))\n",
            "\n",
            "print(f'Baseline pHash      -> Precision: {hash_metrics[\"precision\"]:.4f} | Recall: {hash_metrics[\"recall\"]:.4f} | F1: {hash_metrics[\"f1\"]:.4f}')\n",
            "print(f'Deep Embeddings     -> Precision: {emb_metrics[\"precision\"]:.4f} | Recall: {emb_metrics[\"recall\"]:.4f} | F1: {emb_metrics[\"f1\"]:.4f}')\n",
            "print(f'SplitGuardian Hybrid-> Precision: {hybrid_metrics[\"precision\"]:.4f} | Recall: {hybrid_metrics[\"recall\"]:.4f} | F1: {hybrid_metrics[\"f1\"]:.4f}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Naive Random Split vs SplitGuardian Re-Split Audit\n",
            "We compare:\n",
            "- **Naive Random Split:** Shuffles samples as i.i.d., causing burst & camera leakage.\n",
            "- **SplitGuardian Re-Split:** Group-Preserving Stratified Split allocating whole atomic clusters."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 4: Execute Splitting Strategies\n",
            "# 4.1 Naive Random Split\n",
            "naive_splitter = NaiveRandomSplitter(train_ratio=0.75, val_ratio=0.25, seed=42)\n",
            "leaky_train_ids, leaky_val_ids, _ = naive_splitter.split(trainval)\n",
            "\n",
            "# 4.2 SplitGuardian Re-Split\n",
            "guardian_splitter = GuardianStratifiedSplitter(target_train_ratio=0.75, target_val_ratio=0.25, target_test_ratio=0.0, seed=42)\n",
            "clean_train_ids, clean_val_ids, _, tradeoff = guardian_splitter.split_by_clusters(trainval, hybrid_analysis['cluster_assignment'])\n",
            "\n",
            "# Audit residual leakage\n",
            "leaky_audit = compute_split_leakage_audit(leaky_train_ids, leaky_val_ids, trainval_gt, embeddings)\n",
            "clean_audit = compute_split_leakage_audit(clean_train_ids, clean_val_ids, trainval_gt, embeddings)\n",
            "\n",
            "print(f'Naive Random Split     -> Cross Leaked Pairs: {leaky_audit[\"cross_leaked_pair_count\"]} | Max Cosine Sim: {leaky_audit[\"cross_split_similarity\"][\"max_cosine_sim\"]:.4f}')\n",
            "print(f'SplitGuardian Re-Split -> Cross Leaked Pairs: {clean_audit[\"cross_leaked_pair_count\"]} | Max Cosine Sim: {clean_audit[\"cross_split_similarity\"][\"max_cosine_sim\"]:.4f}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Model Training & Proof of \"Đẹp Giả\" (Generalization Gap)\n",
            "We train identical PyTorch classifiers on both splits and evaluate them on the **True Independent Holdout Test Set** (unseen scenes 14-17).\n",
            "\n",
            "> **LƯU Ý / BẪY:** Metric thấp hơn sau clean split không tự động nghĩa là model tệ hơn; nhóm phải chứng minh evaluation nào đáng tin hơn."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 5: Comparative Model Evaluation\n",
            "meta_map = {m['sample_id']: m for m in metadata}\n",
            "holdout_ids = [m['sample_id'] for m in holdout]\n",
            "\n",
            "eval_leaky = train_and_evaluate_split('Leaky Random Split', leaky_train_ids, leaky_val_ids, holdout_ids, embeddings, meta_map, epochs=45, seed=42)\n",
            "eval_clean = train_and_evaluate_split('SplitGuardian Clean Split', clean_train_ids, clean_val_ids, holdout_ids, embeddings, meta_map, epochs=45, seed=42)\n",
            "\n",
            "print('--- COMPARATIVE MODEL RESULTS ---')\n",
            "print(f'Leaky Split:  Validation Acc = {eval_leaky[\"validation_accuracy\"]}% | Holdout Test Acc = {eval_leaky[\"holdout_test_accuracy\"]}% | Generalization Gap = {eval_leaky[\"generalization_gap_percent\"]}% (ĐẸP GIẢ)')\n",
            "print(f'Clean Split:  Validation Acc = {eval_clean[\"validation_accuracy\"]}% | Holdout Test Acc = {eval_clean[\"holdout_test_accuracy\"]}% | Generalization Gap = {eval_clean[\"generalization_gap_percent\"]}% (ĐÁNG TIN)')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Production CI/CD Gate Verification\n",
            "We can run the automated security gate directly from Python or CLI to verify splits before training."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 6: Test CI/CD Gate on Clean Split\n",
            "from src.cli.split_guardian_gate import run_ci_gate\n",
            "\n",
            "exit_code = run_ci_gate(\n",
            "    manifest_path='../dataset_store/metadata.json',\n",
            "    policy_path='../configs/guardian_policy.yaml',\n",
            "    split_assignment_path='../reports/clean_split.json'\n",
            ")\n",
            "print('Gate Exit Code:', exit_code, '(0 = PASSED)')"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebook"))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "c3_leakage_split_guardian.ipynb")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully at:", out_path)
