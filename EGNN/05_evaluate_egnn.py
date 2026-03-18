#!/usr/bin/env python3
"""
==============================================================================
ToxGuard Phase 2 — EGNN Pipeline
Step 05: Comprehensive Evaluation on Internal Test + External Validation
==============================================================================

Purpose:
    Evaluate the trained ToxEGNN model on:
      1. Internal test set (from scaffold split)
      2. T3DB external validation (nearly all toxic — tests recall)
      3. ClinTox external validation (mostly safe — tests specificity)

    Also performs:
      - Threshold sweep (0.10 to 0.90) to find optimal operating point
      - Per-dataset breakdown for training datasets
      - Generates ROC and PR curves
      - Confidence calibration analysis
      - Outputs publication-ready tables

Input:
    - outputs_egnn/<run_id>/best_model.pt
    - final_egnn_datasets/*_dataset.pt

Output:
    - outputs_egnn/<run_id>/evaluation_results.json
    - outputs_egnn/<run_id>/roc_curves.png
    - outputs_egnn/<run_id>/pr_curves.png
    - outputs_egnn/<run_id>/threshold_sweep.png
    - outputs_egnn/<run_id>/calibration_plot.png

Usage:
    python EGNN/05_evaluate_egnn.py --run_dir outputs_egnn/<run_id>

Author: ToxGuard Team
==============================================================================
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve,
    calibration_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_full_metrics(y_true, y_prob, threshold=0.5):
    """Compute comprehensive metrics at a given threshold."""
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {}
    
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = 0.5
    
    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auprc"] = 0.0
    
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    metrics["sensitivity"] = float(tp / max(tp + fn, 1))
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)
    metrics["threshold"] = threshold
    metrics["n_samples"] = len(y_true)
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())
    
    return metrics


def threshold_sweep(y_true, y_prob, thresholds=None):
    """Sweep across thresholds to find optimal operating points."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
    
    results = []
    for t in thresholds:
        m = compute_full_metrics(y_true, y_prob, threshold=t)
        results.append(m)
    
    # Find optimal by different criteria
    best_f1 = max(results, key=lambda x: x["f1"])
    best_mcc = max(results, key=lambda x: x["mcc"])
    best_balanced = max(results, key=lambda x: (x["sensitivity"] + x["specificity"]) / 2)
    
    return {
        "per_threshold": results,
        "best_f1_threshold": best_f1["threshold"],
        "best_mcc_threshold": best_mcc["threshold"],
        "best_balanced_threshold": best_balanced["threshold"],
    }


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_predictions(model, loader, device):
    """Get predictions for a dataset."""
    model.eval()
    all_labels = []
    all_probs = []
    all_smiles = []
    
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        labels = batch.y.cpu().numpy().flatten()
        
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        
        # Collect SMILES if available
        if hasattr(batch, "smiles"):
            if isinstance(batch.smiles, list):
                all_smiles.extend(batch.smiles)
            elif isinstance(batch.smiles, str):
                all_smiles.append(batch.smiles)
    
    return np.array(all_labels), np.array(all_probs), all_smiles


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_roc_curves(results_dict, save_path):
    """Plot ROC curves for multiple datasets."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        
        colors = {
            "Test (internal)": "#2196F3",
            "T3DB": "#F44336",
            "ClinTox": "#4CAF50",
        }
        
        for name, data in results_dict.items():
            y_true, y_prob = data["y_true"], data["y_prob"]
            if len(set(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auroc = roc_auc_score(y_true, y_prob)
            color = colors.get(name, "#9E9E9E")
            ax.plot(fpr, tpr, linewidth=2, color=color,
                    label=f"{name} (AUC = {auroc:.3f})")
        
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — ToxEGNN", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  ROC curves saved to {save_path}")
    except ImportError:
        logger.warning("  matplotlib not available, skipping plots.")


def plot_pr_curves(results_dict, save_path):
    """Plot Precision-Recall curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        
        colors = {
            "Test (internal)": "#2196F3",
            "T3DB": "#F44336",
            "ClinTox": "#4CAF50",
        }
        
        for name, data in results_dict.items():
            y_true, y_prob = data["y_true"], data["y_prob"]
            if len(set(y_true)) < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            color = colors.get(name, "#9E9E9E")
            ax.plot(recall, precision, linewidth=2, color=color,
                    label=f"{name} (AP = {auprc:.3f})")
        
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curves — ToxEGNN", fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  PR curves saved to {save_path}")
    except ImportError:
        pass


def plot_threshold_sweep(sweep_results, save_path):
    """Plot metrics vs threshold."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        thresholds = [r["threshold"] for r in sweep_results["per_threshold"]]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: F1, Precision, Recall
        ax = axes[0]
        ax.plot(thresholds, [r["f1"] for r in sweep_results["per_threshold"]],
                label="F1", linewidth=2)
        ax.plot(thresholds, [r["precision"] for r in sweep_results["per_threshold"]],
                label="Precision", linewidth=2)
        ax.plot(thresholds, [r["recall"] for r in sweep_results["per_threshold"]],
                label="Recall", linewidth=2)
        ax.axvline(x=sweep_results["best_f1_threshold"], color="red",
                   linestyle="--", alpha=0.5, label=f"Best F1 (t={sweep_results['best_f1_threshold']:.2f})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("F1 / Precision / Recall vs Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Sensitivity, Specificity, MCC
        ax = axes[1]
        ax.plot(thresholds, [r["sensitivity"] for r in sweep_results["per_threshold"]],
                label="Sensitivity", linewidth=2)
        ax.plot(thresholds, [r["specificity"] for r in sweep_results["per_threshold"]],
                label="Specificity", linewidth=2)
        ax.plot(thresholds, [r["mcc"] for r in sweep_results["per_threshold"]],
                label="MCC", linewidth=2)
        ax.axvline(x=sweep_results["best_balanced_threshold"], color="red",
                   linestyle="--", alpha=0.5,
                   label=f"Best Balanced (t={sweep_results['best_balanced_threshold']:.2f})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Sensitivity / Specificity / MCC vs Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle("Threshold Analysis — ToxEGNN (Internal Test Set)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  Threshold sweep plot saved to {save_path}")
    except ImportError:
        pass


def plot_calibration(y_true, y_prob, save_path):
    """Plot calibration curve (reliability diagram)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calibration curve
        ax = axes[0]
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred, fraction_pos, "o-", linewidth=2, label="ToxEGNN")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Histogram of predicted probabilities
        ax = axes[1]
        ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="Toxic",
                color="#F44336", density=True)
        ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="Non-toxic",
                color="#4CAF50", density=True)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle("Calibration Analysis — ToxEGNN",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  Calibration plot saved to {save_path}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Step 05: Evaluate ToxEGNN on internal test + external validation"
    )
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to training run directory (outputs_egnn/<run_id>)")
    parser.add_argument("--data_dir", type=str, default="final_egnn_datasets")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    data_dir = project_root / args.data_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 60)
    logger.info("ToxGuard Phase 2 — ToxEGNN Evaluation")
    logger.info("=" * 60)
    logger.info(f"Run dir:  {run_dir}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Device:   {device}")
    
    # --- Load model ---
    checkpoint = torch.load(run_dir / "best_model.pt", weights_only=False,
                            map_location=device)
    config = checkpoint["config"]
    
    egnn_module = import_module("03_egnn_model")
    model = egnn_module.create_model(
        node_feat_dim=config["data"]["node_feat_dim"],
        edge_feat_dim=config["data"]["edge_feat_dim"],
        config=config["model"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    params = model.count_parameters()
    logger.info(f"\nModel loaded (epoch {checkpoint['epoch']}, "
                f"val_auroc={checkpoint['val_auroc']:.4f})")
    logger.info(f"  Parameters: {params['total']:,}")
    
    # --- Load datasets ---
    datasets = {}
    
    # Internal splits
    for split_name in ["test"]:
        path = data_dir / f"{split_name}_dataset.pt"
        if path.exists():
            datasets[f"Test (internal)"] = torch.load(path, weights_only=False)
    
    # External validation
    for ds_name, display_name in [("t3db", "T3DB"), ("clintox", "ClinTox")]:
        path = data_dir / f"{ds_name}_dataset.pt"
        if path.exists():
            datasets[display_name] = torch.load(path, weights_only=False)
    
    # --- Evaluate ---
    all_results = {}
    plot_data = {}
    
    for ds_name, ds_data in datasets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on: {ds_name} ({len(ds_data)} molecules)")
        logger.info("=" * 60)
        
        loader = DataLoader(ds_data, batch_size=args.batch_size, shuffle=False)
        y_true, y_prob, smiles = get_predictions(model, loader, device)
        
        # Standard metrics at t=0.5
        metrics = compute_full_metrics(y_true, y_prob, threshold=0.5)
        
        logger.info(f"\n  Results at threshold = 0.50:")
        logger.info(f"    AUC-ROC:     {metrics['auroc']:.4f}")
        logger.info(f"    AUC-PRC:     {metrics['auprc']:.4f}")
        logger.info(f"    Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"    Precision:   {metrics['precision']:.4f}")
        logger.info(f"    Recall:      {metrics['recall']:.4f}")
        logger.info(f"    F1:          {metrics['f1']:.4f}")
        logger.info(f"    MCC:         {metrics['mcc']:.4f}")
        logger.info(f"    Specificity: {metrics['specificity']:.4f}")
        logger.info(f"    Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"    TP={metrics['tp']} FP={metrics['fp']} "
                    f"TN={metrics['tn']} FN={metrics['fn']}")
        
        # Threshold sweep
        sweep = threshold_sweep(y_true, y_prob)
        
        logger.info(f"\n  Optimal thresholds:")
        logger.info(f"    Best F1:       t={sweep['best_f1_threshold']:.2f}")
        logger.info(f"    Best MCC:      t={sweep['best_mcc_threshold']:.2f}")
        logger.info(f"    Best Balanced: t={sweep['best_balanced_threshold']:.2f}")
        
        # Metrics at optimal F1 threshold
        opt_metrics = compute_full_metrics(y_true, y_prob,
                                           threshold=sweep["best_f1_threshold"])
        logger.info(f"\n  Results at optimal F1 threshold (t={sweep['best_f1_threshold']:.2f}):")
        logger.info(f"    Accuracy:    {opt_metrics['accuracy']:.4f}")
        logger.info(f"    Precision:   {opt_metrics['precision']:.4f}")
        logger.info(f"    Recall:      {opt_metrics['recall']:.4f}")
        logger.info(f"    F1:          {opt_metrics['f1']:.4f}")
        logger.info(f"    MCC:         {opt_metrics['mcc']:.4f}")
        logger.info(f"    Specificity: {opt_metrics['specificity']:.4f}")
        
        all_results[ds_name] = {
            "metrics_t050": metrics,
            "metrics_optimal_f1": opt_metrics,
            "threshold_sweep": sweep,
        }
        
        plot_data[ds_name] = {
            "y_true": y_true,
            "y_prob": y_prob,
        }
    
    # --- Generate plots ---
    logger.info("\n\nGenerating plots...")
    plot_roc_curves(plot_data, run_dir / "roc_curves.png")
    plot_pr_curves(plot_data, run_dir / "pr_curves.png")
    
    if "Test (internal)" in plot_data:
        test_data = plot_data["Test (internal)"]
        sweep = all_results["Test (internal)"]["threshold_sweep"]
        plot_threshold_sweep(sweep, run_dir / "threshold_sweep.png")
        plot_calibration(test_data["y_true"], test_data["y_prob"],
                        run_dir / "calibration_plot.png")
    
    # --- Publication-ready summary table ---
    logger.info("\n" + "=" * 80)
    logger.info("PUBLICATION SUMMARY TABLE")
    logger.info("=" * 80)
    logger.info(f"{'Dataset':<20} {'AUC-ROC':>8} {'AUC-PRC':>8} {'Acc':>7} "
                f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'Spec':>7} {'MCC':>7}")
    logger.info("-" * 80)
    
    for ds_name, res in all_results.items():
        m = res["metrics_t050"]
        logger.info(
            f"{ds_name:<20} {m['auroc']:>8.4f} {m['auprc']:>8.4f} "
            f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1']:>7.4f} "
            f"{m['specificity']:>7.4f} {m['mcc']:>7.4f}"
        )
    logger.info("=" * 80)
    
    # --- Save results ---
    # Convert numpy arrays for JSON serialisation
    serialisable_results = {}
    for ds_name, res in all_results.items():
        serialisable_results[ds_name] = {
            "metrics_t050": res["metrics_t050"],
            "metrics_optimal_f1": res["metrics_optimal_f1"],
            "best_f1_threshold": res["threshold_sweep"]["best_f1_threshold"],
            "best_mcc_threshold": res["threshold_sweep"]["best_mcc_threshold"],
            "best_balanced_threshold": res["threshold_sweep"]["best_balanced_threshold"],
        }
    
    with open(run_dir / "evaluation_results.json", "w") as f:
        json.dump(serialisable_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {run_dir / 'evaluation_results.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
