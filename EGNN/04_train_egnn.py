#!/usr/bin/env python3
"""
==============================================================================
ToxGuard Phase 2 — EGNN Pipeline
Step 04: Training with Scaffold Splits
==============================================================================

Purpose:
    Train the ToxEGNN model on the combined training dataset
    (toxcast + tox21 + herg + dili + common_molecules) using the
    scaffold split produced by Step 02.

Training details (mirroring Phase 1 where applicable):
    - Loss:          Focal loss (γ=2.0, α=0.45) with label smoothing
    - Optimiser:     AdamW (lr=5e-4, weight_decay=0.01)
    - Scheduler:     Linear warmup (300 steps) + cosine annealing
    - Early stopping: patience=15 on val AUC-ROC
    - Gradient clip:  max_norm=1.0
    - Mixed precision: FP16 (if CUDA available)

Input:
    - final_egnn_datasets/train_dataset.pt
    - final_egnn_datasets/val_dataset.pt
    - final_egnn_datasets/test_dataset.pt
    - final_egnn_datasets/split_info.pt

Output:
    - outputs_egnn/<run_id>/best_model.pt        — Best model checkpoint
    - outputs_egnn/<run_id>/final_model.pt       — Final model checkpoint
    - outputs_egnn/<run_id>/training_log.json     — Training curves
    - outputs_egnn/<run_id>/config.json           — Hyperparameters
    - outputs_egnn/<run_id>/training_curves.png   — Loss/metric plots

Usage:
    python EGNN/04_train_egnn.py [options]

    Key options:
        --batch_size 64         Batch size
        --max_epochs 80         Maximum epochs
        --lr 5e-4               Learning rate
        --hidden_dim 256        Hidden dimension
        --num_layers 6          EGNN layers
        --dropout 0.1           Dropout rate
        --patience 15           Early stopping patience

Author: ToxGuard Team
==============================================================================
"""

import os
import sys
import json
import argparse
import logging
import time
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix,
)

# Import our model
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
# Focal Loss with Label Smoothing
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with optional label smoothing.
    
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    
    Reduces the loss for well-classified examples, focusing training
    on hard negatives. Important for our imbalanced toxicity datasets.
    """
    
    def __init__(self, gamma=2.0, alpha=0.45, label_smoothing=0.1,
                 pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, 1) or (B,) raw logits
            targets: (B, 1) or (B,) binary labels
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Probabilities
        probs = torch.sigmoid(logits)
        
        # Focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=self.pos_weight,
        )
        
        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute comprehensive binary classification metrics."""
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)
    
    metrics = {}
    
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = 0.5
    
    try:
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auprc"] = 0.0
    
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = tn / max(tn + fp, 1)
    metrics["sensitivity"] = tp / max(tp + fn, 1)
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)
    
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(dtype=torch.float16):
                logits = model(batch)
                loss = criterion(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = batch.y.cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = avg_loss
    
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    for batch in loader:
        batch = batch.to(device)
        
        if use_amp:
            with autocast(dtype=torch.float16):
                logits = model(batch)
                loss = criterion(logits, batch.y)
        else:
            logits = model(batch)
            loss = criterion(logits, batch.y)
        
        total_loss += loss.item() * batch.num_graphs
        
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        labels = batch.y.cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
    
    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = avg_loss
    
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(history, save_path):
    """Generate training curve plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("ToxEGNN Training Curves", fontsize=14, fontweight="bold")
        
        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], label="Train", linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Focal Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # AUC-ROC
        ax = axes[0, 1]
        ax.plot(epochs, history["train_auroc"], label="Train", linewidth=2)
        ax.plot(epochs, history["val_auroc"], label="Val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("AUC-ROC")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # AUC-PRC
        ax = axes[1, 0]
        ax.plot(epochs, history["train_auprc"], label="Train", linewidth=2)
        ax.plot(epochs, history["val_auprc"], label="Val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC-PRC")
        ax.set_title("AUC-PRC (Average Precision)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # LR
        ax = axes[1, 1]
        ax.plot(epochs, history["lr"], linewidth=2, color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Training curves saved to {save_path}")
    except ImportError:
        logger.warning("  matplotlib not available, skipping plot generation.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Step 04: Train ToxEGNN model"
    )
    
    # Data
    parser.add_argument("--data_dir", type=str, default="final_egnn_datasets")
    parser.add_argument("--output_dir", type=str, default="outputs_egnn")
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for EGNN layers")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of EGNN message-passing layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--pool_method", type=str, default="attention",
                        choices=["attention", "mean", "sum"])
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--max_epochs", type=int, default=80,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay")
    parser.add_argument("--warmup_steps", type=int, default=300,
                        help="Linear warmup steps")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=1e-3,
                        help="Minimum improvement for early stopping")
    
    # Focal loss
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.45)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main process)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    
    args = parser.parse_args()
    
    # --- Setup ---
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    
    # Add file handler for logging
    fh = logging.FileHandler(run_dir / "training.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    
    logger.info("=" * 60)
    logger.info("ToxGuard Phase 2 — ToxEGNN Training")
    logger.info("=" * 60)
    logger.info(f"Run ID:  {run_id}")
    logger.info(f"Device:  {device}")
    logger.info(f"AMP:     {use_amp}")
    logger.info(f"Run dir: {run_dir}")
    
    # --- Load datasets ---
    logger.info("\nLoading datasets...")
    
    train_data = torch.load(data_dir / "train_dataset.pt", weights_only=False)
    val_data = torch.load(data_dir / "val_dataset.pt", weights_only=False)
    test_data = torch.load(data_dir / "test_dataset.pt", weights_only=False)
    split_info = torch.load(data_dir / "split_info.pt", weights_only=False)
    
    logger.info(f"  Train: {len(train_data)} molecules")
    logger.info(f"  Val:   {len(val_data)} molecules")
    logger.info(f"  Test:  {len(test_data)} molecules")
    
    node_feat_dim = split_info["node_feature_dim"]
    edge_feat_dim = split_info["edge_feature_dim"]
    logger.info(f"  Node feat dim: {node_feat_dim}")
    logger.info(f"  Edge feat dim: {edge_feat_dim}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    # --- Compute class weights ---
    n_toxic = sum(1 for d in train_data if d.y.item() >= 0.5)
    n_nontoxic = len(train_data) - n_toxic
    pos_weight = torch.tensor([n_nontoxic / max(n_toxic, 1)], device=device)
    logger.info(f"  Class balance: {n_toxic} toxic, {n_nontoxic} non-toxic")
    logger.info(f"  pos_weight: {pos_weight.item():.3f}")
    
    # --- Create model ---
    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pool_method": args.pool_method,
        "update_coords": True,
        "num_classes": 1,
    }
    
    # Import model
    egnn_module = import_module("03_egnn_model")
    model = egnn_module.create_model(node_feat_dim, edge_feat_dim, model_config)
    model = model.to(device)
    
    params_info = model.count_parameters()
    logger.info(f"\nModel created:")
    logger.info(f"  Parameters: {params_info['total']:,} total, "
                f"{params_info['trainable']:,} trainable")
    
    # --- Loss, optimiser, scheduler ---
    criterion = FocalLoss(
        gamma=args.focal_gamma,
        alpha=args.focal_alpha,
        label_smoothing=args.label_smoothing,
        pos_weight=pos_weight,
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Scheduler: linear warmup then cosine annealing
    steps_per_epoch = len(train_loader)
    total_steps = args.max_epochs * steps_per_epoch
    warmup_epochs = max(1, args.warmup_steps // steps_per_epoch)
    
    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs - warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
    )
    
    scaler = GradScaler(enabled=use_amp)
    
    # --- Save config ---
    config = {
        "model": model_config,
        "training": {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "patience": args.patience,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "label_smoothing": args.label_smoothing,
            "pos_weight": pos_weight.item(),
            "seed": args.seed,
            "use_amp": use_amp,
        },
        "data": {
            "node_feat_dim": node_feat_dim,
            "edge_feat_dim": edge_feat_dim,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "n_toxic_train": n_toxic,
            "n_nontoxic_train": n_nontoxic,
        },
        "run_id": run_id,
    }
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # --- Training loop ---
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    history = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_auroc": [], "val_auroc": [], "test_auroc": [],
        "train_auprc": [], "val_auprc": [], "test_auprc": [],
        "lr": [],
    }
    
    best_val_auroc = 0.0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        
        # Test (for monitoring, not for model selection)
        test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["test_loss"].append(test_metrics["loss"])
        history["train_auroc"].append(train_metrics["auroc"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["test_auroc"].append(test_metrics["auroc"])
        history["train_auprc"].append(train_metrics["auprc"])
        history["val_auprc"].append(val_metrics["auprc"])
        history["test_auprc"].append(test_metrics["auprc"])
        history["lr"].append(current_lr)
        
        elapsed = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch:3d}/{args.max_epochs} | "
            f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f}/{test_metrics['loss']:.4f} | "
            f"AUROC: {train_metrics['auroc']:.4f}/{val_metrics['auroc']:.4f}/{test_metrics['auroc']:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )
        
        # Early stopping check
        if val_metrics["auroc"] > best_val_auroc + args.min_delta:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Save best model
            torch.save({
                "model_state_dict": best_model_state,
                "epoch": best_epoch,
                "val_auroc": best_val_auroc,
                "config": config,
            }, run_dir / "best_model.pt")
            
            logger.info(f"  >> New best! Val AUROC = {best_val_auroc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"\n  Early stopping at epoch {epoch} "
                           f"(no improvement for {args.patience} epochs)")
                break
    
    total_time = time.time() - start_time
    
    # --- Save final model ---
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "config": config,
    }, run_dir / "final_model.pt")
    
    # --- Evaluate best model ---
    logger.info("\n" + "=" * 60)
    logger.info(f"Training complete. Best epoch: {best_epoch} "
                f"(Val AUROC = {best_val_auroc:.4f})")
    logger.info(f"Total training time: {total_time:.1f}s "
                f"({total_time/60:.1f} min)")
    logger.info("=" * 60)
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    logger.info("\nFinal evaluation with best model:")
    for split_name, loader in [("Train", train_loader), 
                                ("Val", val_loader),
                                ("Test", test_loader)]:
        metrics = evaluate(model, loader, criterion, device, use_amp)
        logger.info(f"\n  {split_name}:")
        logger.info(f"    Loss:        {metrics['loss']:.4f}")
        logger.info(f"    AUC-ROC:     {metrics['auroc']:.4f}")
        logger.info(f"    AUC-PRC:     {metrics['auprc']:.4f}")
        logger.info(f"    Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"    Precision:   {metrics['precision']:.4f}")
        logger.info(f"    Recall:      {metrics['recall']:.4f}")
        logger.info(f"    F1:          {metrics['f1']:.4f}")
        logger.info(f"    MCC:         {metrics['mcc']:.4f}")
        logger.info(f"    Specificity: {metrics['specificity']:.4f}")
        logger.info(f"    Confusion: TP={metrics['tp']} FP={metrics['fp']} "
                    f"TN={metrics['tn']} FN={metrics['fn']}")
    
    # --- Save history ---
    with open(run_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # --- Plot training curves ---
    plot_training_curves(history, run_dir / "training_curves.png")
    
    logger.info(f"\nAll outputs saved to: {run_dir}")
    logger.info("=" * 60)
    
    return str(run_dir)


if __name__ == "__main__":
    main()
