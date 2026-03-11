#!/usr/bin/env python3
"""
STEP 6 -- Evaluate Trained Model
=================================
Reads  :  outputs/last_run.txt                  <- pointer to the run folder from step 5
          outputs/<run>/lora_weights.pt         <- trained LoRA adapter weights
          data/toxcast_final.csv                <- ToxCast test data
          data/tox21_final.csv                  <- Tox21 test data
          data/t3db_processed.csv               <- T3DB test data
          data/clintox_final.csv                <- ClinTox test data
          data/herg_final.csv                   <- hERG cardiotoxicity test data
          data/dili_final.csv                   <- DILI test data
          data/common_molecules_final.csv       <- Common Molecules test data

Outputs:  outputs/<run>/evaluation_report.txt  <- full evaluation report

What it does:
  1. Loads the trained LoRA model
  2. Evaluates on the held-out test set (ToxCast + Tox21 + T3DB + ClinTox + hERG + DILI + Common Molecules)
     using the SAME stratified split as step5/data_pipeline (seed=42)
  3. Reports:
       - Binary metrics: AUC-ROC, AUC-PRC, Accuracy, F1, MCC
       - Binary confusion matrix
       - Severity label distribution (derived from P(toxic))
       - Example predictions with CoT explanations

Run from project root:
  python steps/step6_evaluate.py

To evaluate a specific run (not the latest):
  python steps/step6_evaluate.py --run outputs/run_20260219_143000
"""

import os
import sys
import json
import argparse
import logging

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join("iupacGPT", "iupac-gpt", "checkpoints", "iupac")
SPM_PATH       = os.path.join("iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model")
DATA_DIR       = "./data"
OUTPUT_DIR     = "./outputs"

# Example molecules: (iupac_name, expected_toxic)
# All drawn from common_molecules_final.csv (confirmed in training data)
EVAL_MOLECULES = [
    # --- Non-toxic ---
    ("propan-1-ol",                    False),   # simple alcohol; safe at normal exposure
    ("methyl benzoate",                False),   # food-grade flavour ester
    ("cyclohexene",                    False),   # inert cyclic alkene
    ("2-methylpropanoic acid",         False),   # isobutyric acid; low-hazard
    ("(2S)-2-aminopentanedioic acid",  False),   # L-glutamic acid; dietary amino acid
    # --- Toxic ---
    ("ethanal",                        True),    # acetaldehyde; carcinogen / metabolic toxin
    ("sodium azide",                   True),    # highly toxic inorganic azide
    ("nitrogen dioxide",               True),    # toxic gas; respiratory damage
    ("methylhydrazine",                True),    # rocket propellant; hepatotoxic
    ("ethyl prop-2-enoate",            True),    # ethyl acrylate; irritant / genotoxic
]


def get_last_run_dir() -> str:
    """Get the most recent training run directory."""
    pointer = os.path.join(OUTPUT_DIR, "last_run.txt")
    if os.path.exists(pointer):
        with open(pointer) as f:
            return f.read().strip()

    # Fallback: pick most recent run_ folder
    runs = sorted([
        d for d in os.listdir(OUTPUT_DIR)
        if d.startswith("run_") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ], reverse=True)
    if runs:
        return os.path.join(OUTPUT_DIR, runs[0])
    return None


def compute_binary_metrics(all_probs: list, all_labels: list) -> dict:
    """Compute binary classification metrics (toxic vs non-toxic)."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  accuracy_score, f1_score, matthews_corrcoef,
                                  confusion_matrix, precision_score, recall_score)
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    try:
        auroc = roc_auc_score(labels, probs)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(labels, probs)
    except Exception:
        auprc = float("nan")

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "auc_roc":   auroc,
        "auc_prc":   auprc,
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "mcc":       matthews_corrcoef(labels, preds),
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_on_test_set(model, test_loader, device):
    """Run inference on test loader and collect metrics."""
    model.eval()
    model.to(device)

    all_binary_probs  = []
    all_binary_labels = []

    logger.info(f"Evaluating on {len(test_loader)} test batches...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_labels  = batch["binary_labels"]

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            # Binary predictions
            binary_prob = torch.sigmoid(output.binary_logits).squeeze(-1).cpu().numpy()
            all_binary_probs.extend(binary_prob.tolist())
            all_binary_labels.extend(binary_labels.numpy().tolist())

    binary_metrics = compute_binary_metrics(all_binary_probs, all_binary_labels)
    return binary_metrics


def run_molecule_examples(model, tokenizer, device):
    """Run predictions on example molecules and print a summary table."""
    from toxguard.inference import ToxGuardPredictor

    predictor = ToxGuardPredictor(model, tokenizer, device=str(device))

    print("\n  Example Predictions (Binary + P(toxic)):")
    print("  " + "-" * 75)
    print(f"  {'Molecule':<30} {'Prediction':<14} {'P(toxic)':<10} {'Severity':<18} {'OK'}")
    print("  " + "-" * 75)

    for iupac, expected_toxic in EVAL_MOLECULES:
        pred = predictor.predict(iupac, return_egnn_vector=False)
        pred_toxic = pred.is_toxic
        match = "Y" if pred_toxic == expected_toxic else "N"
        toxic_str = "TOXIC" if pred_toxic else "Non-toxic"
        print(f"  {iupac:<30} {toxic_str:<14} "
              f"{pred.toxicity_score:.3f}      {pred.severity_label:<18} {match}")

    print("  " + "-" * 75)


def main():
    parser = argparse.ArgumentParser(description="STEP 6: Evaluate trained ToxGuard model")
    parser.add_argument("--run", type=str, default=None,
                        help="Run folder path (default: latest run)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  STEP 6 -- Evaluate Trained Model")
    print("=" * 60)

    # -- Find run directory
    run_dir = args.run or get_last_run_dir()
    if run_dir is None or not os.path.exists(run_dir):
        print("\n[ERROR] No trained run found.")
        print("  -> Complete training first: python steps/step5_train.py")
        sys.exit(1)

    lora_path = os.path.join(run_dir, "lora_weights.pt")
    if not os.path.exists(lora_path):
        print(f"\n[ERROR] lora_weights.pt not found in {run_dir}")
        sys.exit(1)

    logger.info(f"Evaluating run: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # -- Load model
    from toxguard.tokenizer import get_tokenizer
    from toxguard.model import ToxGuardModel, SEVERITY_LABELS, score_to_severity_label
    from toxguard.lora import apply_lora_to_model, load_lora_weights, LoRAConfig
    from toxguard.data_pipeline import prepare_combined_dataset

    tokenizer = get_tokenizer(vocab_path=SPM_PATH)

    model = ToxGuardModel.from_pretrained_iupacgpt(CHECKPOINT_DIR)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA config if available
    lora_cfg_path = os.path.join(OUTPUT_DIR, "lora_config.json")
    if os.path.exists(lora_cfg_path):
        with open(lora_cfg_path) as f:
            lora_cfg = json.load(f)
        lora_config = LoRAConfig(
            r=lora_cfg["r"], alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg.get("target_modules", ["c_attn", "c_proj", "c_fc"]),
            fan_in_fan_out=True,
        )
    else:
        lora_config = LoRAConfig()

    model, _ = apply_lora_to_model(model, lora_config)
    model = load_lora_weights(model, lora_path)
    logger.info(f"Loaded LoRA weights from {lora_path}")

    # -- Build test DataLoader using the SAME stratified split as training (BUG #1 FIX)
    # This calls prepare_combined_dataset which uses StratifiedShuffleSplit with seed=42
    _, _, test_loader = prepare_combined_dataset(
        data_dir=DATA_DIR,
        tokenizer=tokenizer,
        max_length=128,
        val_split=0.1,
        test_split=0.1,
        batch_size=32,
        num_workers=0,  # 0 for evaluation stability
    )
    logger.info(f"Test set: {len(test_loader)} batches")

    # -- Evaluate
    binary_metrics = evaluate_on_test_set(model, test_loader, device)

    # -- Print report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  TOXGUARD EVALUATION REPORT")
    report_lines.append("=" * 60)

    # Binary metrics
    report_lines.append(f"\n  Binary Classification Metrics (toxic vs non-toxic):")
    report_lines.append(f"    AUC-ROC   : {binary_metrics['auc_roc']:.4f}")
    report_lines.append(f"    AUC-PRC   : {binary_metrics['auc_prc']:.4f}")
    report_lines.append(f"    Accuracy  : {binary_metrics['accuracy']:.4f}")
    report_lines.append(f"    Precision : {binary_metrics['precision']:.4f}")
    report_lines.append(f"    Recall    : {binary_metrics['recall']:.4f}")
    report_lines.append(f"    F1 Score  : {binary_metrics['f1']:.4f}")
    report_lines.append(f"    MCC       : {binary_metrics['mcc']:.4f}")
    report_lines.append(f"    Samples   : {binary_metrics['n_samples']} "
                        f"({binary_metrics['n_positive']} toxic, "
                        f"{binary_metrics['n_negative']} non-toxic)")

    cm = binary_metrics["confusion_matrix"]
    report_lines.append(f"\n  Binary Confusion Matrix:")
    report_lines.append(f"                  Pred Non-toxic   Pred Toxic")
    report_lines.append(f"    True Non-toxic  {cm[0][0]:>10}   {cm[0][1]:>10}")
    report_lines.append(f"    True Toxic      {cm[1][0]:>10}   {cm[1][1]:>10}")

    report_lines.append(f"\n  Severity labels (anchored to 0.5 binary decision boundary):")
    report_lines.append(f"    0.00-0.20 = Non-toxic       (very confident non-toxic)")
    report_lines.append(f"    0.20-0.50 = Unlikely toxic  (leans non-toxic)")
    report_lines.append(f"    0.50-0.65 = Likely toxic    (leans toxic, lower confidence)")
    report_lines.append(f"    0.65-0.80 = Moderately toxic (moderately confident toxic)")
    report_lines.append(f"    0.80-1.00 = Highly toxic    (very confident toxic)")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # -- Run example predictions
    run_molecule_examples(model, tokenizer, device)

    # -- Save report
    report_path = os.path.join(run_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved evaluation report to {report_path}")

    # -- Save metrics as JSON
    metrics_path = os.path.join(run_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"binary": binary_metrics}, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")

    print("\n" + "-" * 60)
    print("  Step 6 complete.")
    print(f"    Evaluation report  : {report_path}")
    print(f"    Metrics JSON       : {metrics_path}")
    print("  Next -> run:  python steps/step7_predict.py")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
