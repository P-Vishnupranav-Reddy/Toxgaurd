#!/usr/bin/env python3
"""
STEP 5 -- Train ToxGuard (LoRA Fine-tuning)
============================================
Reads  :  data/toxcast_final.csv             <- ToxCast (binary from 617 assays)
          data/tox21_final.csv              <- Tox21 (binary from 12 assays)
          data/t3db_processed.csv           <- T3DB (all toxic)
          data/clintox_final.csv            <- ClinTox (FDA approval + clinical tox)
          data/herg_final.csv               <- hERG (cardiotoxicity — hERG blocker)
          data/dili_final.csv               <- DILI (drug-induced liver injury)
          data/common_molecules_final.csv   <- curated ~1100 short-IUPAC molecules
          iupacGPT/iupac-gpt/checkpoints/iupac/  <- IUPACGPT backbone
          outputs/lora_config.json               <- LoRA config from step 4

Outputs:  outputs/<run_name>/lora_weights.pt    <- trained LoRA adapter weights
          outputs/<run_name>/checkpoints/       <- Lightning checkpoints
          outputs/<run_name>/tensorboard/       <- TensorBoard training logs
          outputs/<run_name>/config.json        <- full training config
          outputs/<run_name>/results.json       <- test metrics
          outputs/last_run.txt                  <- pointer to latest run

Architecture:
  IUPAC Name -> GPT-2 + LoRA -> Binary Head (toxic/non-toxic)
  P(toxic) = sigmoid(binary_logit), severity derived from fixed thresholds.

Run from project root:
  python steps/step5_train.py
  python steps/step5_train.py --max_epochs 30 --batch_size 16
  python steps/step5_train.py --learning_rate 1e-4 --lora_rank 16
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toxguard.model import ToxGuardModel, ToxGuardLitModel
from toxguard.tokenizer import get_tokenizer
from toxguard.lora import apply_lora_to_model, LoRAConfig, save_lora_weights
from toxguard.data_pipeline import prepare_combined_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join("iupacGPT", "iupac-gpt", "checkpoints", "iupac")
SPM_PATH       = os.path.join("iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model")
DATA_DIR       = "./data"
OUTPUT_DIR     = "./outputs"


def main(args: argparse.Namespace):
    """Main training pipeline."""

    print("\n" + "=" * 60)
    print("  STEP 5 — Train ToxGuard (LoRA Fine-tuning)")
    print("=" * 60)

    # ── Seed for reproducibility ──
    seed_everything(args.seed, workers=True)

    # Enable Tensor Core utilization on RTX GPUs
    torch.set_float32_matmul_precision("medium")

    # ── Create run directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save pointer for step6/step7
    with open(os.path.join(args.output_dir, "last_run.txt"), "w") as f:
        f.write(run_dir)

    # Save config
    config_dict = vars(args)
    config_dict["run_dir"] = run_dir
    config_dict["run_name"] = run_name
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Run directory: {run_dir}")

    # ── Tokenizer ──
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(vocab_path=args.tokenizer)
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ── Data ──
    logger.info("Preparing datasets...")
    train_loader, val_loader, test_loader = prepare_combined_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_split=args.val_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(f"Train batches: {len(train_loader)}, "
                f"Val: {len(val_loader)}, Test: {len(test_loader)}")

    # ── Model ──
    logger.info("Loading IUPACGPT backbone...")
    model = ToxGuardModel.from_pretrained_iupacgpt(
        args.checkpoint
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Set loss weight
    model.binary_loss_weight = args.binary_loss_weight

    # ── LoRA ──
    # Try to load LoRA config from step4's output
    lora_cfg_path = os.path.join(args.output_dir, "lora_config.json")
    if os.path.exists(lora_cfg_path) and not any(
        getattr(args, a, None) is not None
        for a in ["_lora_rank_set", "_lora_alpha_set"]
    ):
        with open(lora_cfg_path) as f:
            saved_cfg = json.load(f)
        logger.info(f"Using LoRA config from step 4: {lora_cfg_path}")
        lora_config = LoRAConfig(
            r=saved_cfg.get("r", args.lora_rank),
            alpha=saved_cfg.get("alpha", args.lora_alpha),
            dropout=saved_cfg.get("dropout", args.lora_dropout),
            target_modules=saved_cfg.get("target_modules",
                                          args.lora_targets.split(",")),
            fan_in_fan_out=True,
        )
    else:
        lora_config = LoRAConfig(
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_targets.split(","),
            fan_in_fan_out=True,
        )

    logger.info(f"Applying LoRA (rank={lora_config.r}, alpha={lora_config.alpha}, "
                f"dropout={lora_config.dropout})...")
    model, lora_stats = apply_lora_to_model(model, lora_config)

    logger.info(f"Parameter summary:")
    logger.info(f"  Total:      {lora_stats['total_params']:>10,}")
    logger.info(f"  Trainable:  {lora_stats['trainable_params']:>10,} "
                f"({lora_stats['trainable_pct']:.2f}%)")
    logger.info(f"  LoRA:       {lora_stats['lora_params']:>10,}")
    logger.info(f"  Frozen:     {lora_stats['frozen_params']:>10,}")

    # ── Lightning Module ──
    max_steps = len(train_loader) * args.max_epochs
    lit_model = ToxGuardLitModel(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
        scheduler_type=args.scheduler,
    )

    # ── Callbacks ──
    callbacks = [
        EarlyStopping(
            monitor="val_auroc",
            min_delta=args.es_min_delta,
            patience=args.es_patience,
            mode="max",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(run_dir, "checkpoints"),
            filename="toxguard-{epoch:02d}-{val_auroc:.3f}",
            monitor="val_auroc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Optional progress bar
    try:
        from pytorch_lightning.callbacks import RichProgressBar
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    # ── Logger ──
    tb_logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard")

    # ── Trainer ──
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.grad_accumulation,
        precision=args.precision,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        deterministic=False,
    )

    # ── Train ──
    logger.info("=" * 60)
    logger.info("  Starting training...")
    logger.info(f"  Epochs:   {args.max_epochs}")
    logger.info(f"  Batch:    {args.batch_size}")
    logger.info(f"  LR:       {args.learning_rate}")
    logger.info(f"  Task:     Binary classification (toxic / non-toxic)")
    logger.info("=" * 60)

    trainer.fit(lit_model, train_loader, val_loader)

    # ── Test ──
    logger.info("Running test evaluation...")
    test_results = trainer.test(model=lit_model, dataloaders=test_loader)

    # ── Save LoRA weights ──
    lora_save_path = os.path.join(run_dir, "lora_weights.pt")
    save_lora_weights(model, lora_save_path)
    logger.info(f"LoRA weights saved to: {lora_save_path}")

    # ── Save results ──
    results = {
        "test_results": test_results,
        "lora_stats": lora_stats,
        "config": config_dict,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Print summary ──
    test_auroc = test_results[0].get("test_auroc", "N/A") if test_results else "N/A"
    test_auprc = test_results[0].get("test_auprc", "N/A") if test_results else "N/A"
    test_acc = test_results[0].get("test_acc", "N/A") if test_results else "N/A"

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"    Run directory   : {run_dir}")
    print(f"    LoRA weights    : {lora_save_path}")
    print(f"    Test AUC-ROC    : {test_auroc}")
    print(f"    Test AUC-PRC    : {test_auprc}")
    print(f"    Test Binary Acc : {test_acc}")
    print(f"\n  Next → run:  python steps/step6_evaluate.py")
    print("=" * 60 + "\n")

    return test_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STEP 5: Train ToxGuard — LoRA fine-tune IUPACGPT for toxicity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR,
                        help="Path to IUPACGPT checkpoint directory")
    parser.add_argument("--tokenizer", default=SPM_PATH,
                        help="Path to iupac_spm.model")
    parser.add_argument("--data_dir", default=DATA_DIR,
                        help="Directory containing processed data CSVs")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Output directory for run artifacts")

    # Data
    parser.add_argument("--max_length", type=int, default=300,
                        help="Maximum token sequence length (dataset max is 287 tokens)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Fraction of data for test")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (r). Rank 16 balances capacity vs efficiency for this 7M-param model.")
    parser.add_argument("--lora_alpha", type=float, default=32.0,
                        help="LoRA scaling alpha (keep alpha = 2*rank for stable scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.2,
                        help="LoRA dropout (0.2 to reduce overfitting)")
    parser.add_argument("--lora_targets", default="c_attn,c_proj,c_fc",
                        help="LoRA target modules: c_attn+c_proj (attention) + c_fc (MLP FFN)")

    # Training
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size (use 8 if OOM, 32 if >8GB VRAM)")
    parser.add_argument("--max_epochs", type=int, default=40,
                        help="Maximum training epochs. Early stopping (patience=7) "
                             "prevents overfitting if val_auroc plateaus.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Peak learning rate (1e-4 for stable LoRA fine-tuning)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay")
    parser.add_argument("--warmup_steps", type=int, default=300,
                        help="Linear warmup steps (increased for rank-16 LoRA stability)")
    parser.add_argument("--scheduler", default="cosine",
                        choices=["cosine", "exponential", "none"],
                        help="LR scheduler type")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--grad_accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--precision", default="16-mixed",
                        help="Training precision: 32, 16-mixed (FP16), bf16-mixed (Ampere+)")
    parser.add_argument("--val_check_interval", type=float, default=1.0,
                        help="How often to run validation (1.0 = every epoch)")

    # Loss weights
    parser.add_argument("--binary_loss_weight", type=float, default=1.0,
                        help="Weight for binary classification loss")


    # Early stopping
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping patience (epochs). 5 prevents overfitting "
                             "while allowing recovery from brief plateaus.")
    parser.add_argument("--es_min_delta", type=float, default=1e-3,
                        help="Early stopping minimum improvement")

    # Hardware
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs/devices")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (0 = main thread, 4 = background prefetch)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
