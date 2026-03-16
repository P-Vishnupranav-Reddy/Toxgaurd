#!/usr/bin/env python3
"""
STEP 6 -- Evaluate Trained Model
=================================
Reads  :  outputs/last_run.txt                  <- pointer to the run folder from step 5
          outputs/<run>/lora_weights.pt         <- trained LoRA adapter weights
          data/toxcast_final.csv                <- ToxCast (training split, test portion)
          data/tox21_final.csv                  <- Tox21 (training split, test portion)
          data/herg_final.csv                   <- hERG (training split, test portion)
          data/dili_final.csv                   <- DILI (training split, test portion)
          data/common_molecules_final.csv       <- Common Molecules (training split, test portion)
          data/t3db_processed.csv               <- T3DB EXTERNAL VALIDATION (never in training)
          data/clintox_final.csv                <- ClinTox EXTERNAL VALIDATION (never in training)

Outputs:  outputs/<run>/evaluation_report.txt  <- full evaluation report
          outputs/<run>/eval_metrics.json       <- metrics in JSON form

What it does:
  1. Loads the trained LoRA model
  2. Evaluates on the held-out test split from the TRAINING datasets
     (ToxCast + Tox21 + hERG + DILI + Common Molecules) using the SAME
     stratified split as step5/data_pipeline (seed=42).
  3. Reports primary metrics (AUC-ROC, AUC-PRC, F1, MCC) at both the
     default 0.5 threshold and the val-set best-MCC threshold.
  4. Evaluates on EXTERNAL validation datasets (T3DB, ClinTox) which were
     never seen during training — reported separately to avoid contamination:
       - T3DB   (99.2% toxic)  : headline metric is Recall (sensitivity)
       - ClinTox (3.4% toxic)  : headline metrics are Specificity and Precision
  5. Calibrates temperature scaling on the validation set.

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


def compute_binary_metrics(all_probs: list, all_labels: list,
                            threshold: float = 0.5) -> dict:
    """Compute binary classification metrics at a given decision threshold."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  accuracy_score, f1_score, matthews_corrcoef,
                                  confusion_matrix, precision_score, recall_score)
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)

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
        "threshold": threshold,
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


def tune_threshold(val_probs: list, val_labels: list) -> dict:
    """Sweep thresholds 0.10–0.90 on the validation set.

    Returns a dict with:
        best_f1_threshold  : threshold maximising F1
        best_mcc_threshold : threshold maximising MCC
        best_acc_threshold : threshold maximising accuracy
        sweep              : full list of (threshold, f1, mcc, acc) tuples
    """
    from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

    probs  = np.array(val_probs)
    labels = np.array(val_labels)

    thresholds = np.arange(0.10, 0.91, 0.01)
    best_f1  = (-1.0, 0.5)   # (score, threshold)
    best_mcc = (-2.0, 0.5)
    best_acc = (-1.0, 0.5)
    sweep = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1  = f1_score(labels, preds, zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
        acc = accuracy_score(labels, preds)
        sweep.append({"threshold": round(float(t), 2), "f1": round(f1, 4),
                      "mcc": round(mcc, 4), "accuracy": round(acc, 4)})
        if f1  > best_f1[0]:  best_f1  = (f1,  float(t))
        if mcc > best_mcc[0]: best_mcc = (mcc, float(t))
        if acc > best_acc[0]: best_acc = (acc, float(t))

    result = {
        "best_f1_threshold":  round(best_f1[1],  2),
        "best_f1_score":      round(best_f1[0],  4),
        "best_mcc_threshold": round(best_mcc[1], 2),
        "best_mcc_score":     round(best_mcc[0], 4),
        "best_acc_threshold": round(best_acc[1], 2),
        "best_acc_score":     round(best_acc[0], 4),
        "sweep": sweep,
    }
    logger.info(
        f"Threshold tuning → best F1={best_f1[0]:.4f} @ t={best_f1[1]:.2f} | "
        f"best MCC={best_mcc[0]:.4f} @ t={best_mcc[1]:.2f} | "
        f"best Acc={best_acc[0]:.4f} @ t={best_acc[1]:.2f}"
    )
    return result


def collect_probs(model, loader, device) -> tuple:
    """Run inference on a DataLoader; return (probs, labels) lists."""
    model.eval()
    model.to(device)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_labels  = batch["binary_labels"]
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(output.binary_logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(binary_labels.numpy().tolist())
    return all_probs, all_labels


def evaluate_on_test_set(model, test_loader, device, threshold: float = 0.5):
    """Run inference on test loader and compute metrics at the given threshold."""
    logger.info(f"Evaluating on {len(test_loader)} test batches "
                f"(threshold={threshold:.2f})...")
    all_probs, all_labels = collect_probs(model, test_loader, device)
    return compute_binary_metrics(all_probs, all_labels, threshold=threshold)


def run_molecule_examples(model, tokenizer, device, threshold: float = 0.5):
    """Run predictions on example molecules and print a summary table."""
    from toxguard.inference import ToxGuardPredictor

    predictor = ToxGuardPredictor(model, tokenizer, device=str(device),
                                   threshold=threshold)

    print(f"\n  Example Predictions (threshold={threshold:.2f}):")
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
    from toxguard.data_pipeline import (prepare_combined_dataset,
                                        load_external_validation_datasets)

    tokenizer = get_tokenizer(vocab_path=SPM_PATH)

    model = ToxGuardModel.from_pretrained_iupacgpt(CHECKPOINT_DIR)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load run config to match data splitting strategy AND LoRA config
    # Always read LoRA config from the run's own config.json — this is the
    # authoritative source of what rank/alpha were actually used in that run.
    config_path = os.path.join(run_dir, "config.json")
    split_method = "scaffold"
    val_split = 0.1
    test_split = 0.1
    lora_rank = 32
    lora_alpha = 64.0
    lora_dropout = 0.2
    lora_targets = ["c_attn", "c_proj", "c_fc"]
    if os.path.exists(config_path):
        with open(config_path) as f:
            run_config = json.load(f)
        split_method = run_config.get("split_method", split_method)
        val_split    = run_config.get("val_split",    val_split)
        test_split   = run_config.get("test_split",   test_split)
        lora_rank    = run_config.get("lora_rank",    lora_rank)
        lora_alpha   = run_config.get("lora_alpha",   lora_alpha)
        lora_dropout = run_config.get("lora_dropout", lora_dropout)
        raw_targets  = run_config.get("lora_targets", ",".join(lora_targets))
        lora_targets = raw_targets.split(",") if isinstance(raw_targets, str) else raw_targets
        logger.info(f"Loaded config from {config_path} "
                    f"(split_method={split_method}, lora_rank={lora_rank})")

    lora_config = LoRAConfig(
        r=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=lora_targets,
        fan_in_fan_out=True,
    )

    model, _ = apply_lora_to_model(model, lora_config)
    model = load_lora_weights(model, lora_path)
    model = model.to(device)
    logger.info(f"Loaded LoRA weights from {lora_path}")

    # -- Build val + test DataLoaders using the SAME split configuration as training
    # This calls prepare_combined_dataset which uses the chosen split_method (deterministic if seed=42)
    _, val_loader, test_loader, _ = prepare_combined_dataset(
        data_dir=DATA_DIR,
        tokenizer=tokenizer,
        max_length=128,
        val_split=val_split,
        test_split=test_split,
        batch_size=32,
        num_workers=0,  # 0 for evaluation stability
        split_method=split_method,
    )
    logger.info(f"Test set: {len(test_loader)} batches")

    # -- Temperature scaling calibration on validation set --
    from toxguard.calibration import TemperatureScaler
    scaler = TemperatureScaler()
    optimal_temp = scaler.calibrate(model, val_loader, device)
    logger.info(f"Temperature scaling: T={optimal_temp:.4f}")

    # Save calibration
    calib_path = os.path.join(run_dir, "temperature.pt")
    scaler.save(calib_path)

    # -- Threshold tuning on validation set --
    logger.info("Tuning decision threshold on validation set...")
    val_probs, val_labels = collect_probs(model, val_loader, device)
    threshold_info = tune_threshold(val_probs, val_labels)
    operative_threshold = threshold_info["best_mcc_threshold"]
    logger.info(f"Using threshold={operative_threshold:.2f} (best MCC on val set)")

    # Save threshold info
    threshold_path = os.path.join(run_dir, "threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)
    logger.info(f"Saved threshold info to {threshold_path}")

    # -- Evaluate at default 0.5 AND at tuned threshold --
    binary_metrics_default = evaluate_on_test_set(model, test_loader, device, threshold=0.5)
    binary_metrics = evaluate_on_test_set(model, test_loader, device, threshold=operative_threshold)

    # -- Print report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  TOXGUARD EVALUATION REPORT")
    report_lines.append("=" * 60)

    # Threshold tuning summary
    report_lines.append(f"\n  Threshold Tuning (validation set sweep 0.10–0.90):")
    report_lines.append(f"    Best F1  threshold : {threshold_info['best_f1_threshold']:.2f}  "
                        f"(val F1={threshold_info['best_f1_score']:.4f})")
    report_lines.append(f"    Best MCC threshold : {threshold_info['best_mcc_threshold']:.2f}  "
                        f"(val MCC={threshold_info['best_mcc_score']:.4f})")
    report_lines.append(f"    Best Acc threshold : {threshold_info['best_acc_threshold']:.2f}  "
                        f"(val Acc={threshold_info['best_acc_score']:.4f})")
    report_lines.append(f"    Operative threshold: {operative_threshold:.2f}  (best MCC)")

    # Binary metrics at default 0.5
    report_lines.append(f"\n  Binary Classification Metrics @ threshold=0.50 (default):")
    report_lines.append(f"    AUC-ROC   : {binary_metrics_default['auc_roc']:.4f}")
    report_lines.append(f"    AUC-PRC   : {binary_metrics_default['auc_prc']:.4f}")
    report_lines.append(f"    Accuracy  : {binary_metrics_default['accuracy']:.4f}")
    report_lines.append(f"    Precision : {binary_metrics_default['precision']:.4f}")
    report_lines.append(f"    Recall    : {binary_metrics_default['recall']:.4f}")
    report_lines.append(f"    F1 Score  : {binary_metrics_default['f1']:.4f}")
    report_lines.append(f"    MCC       : {binary_metrics_default['mcc']:.4f}")

    # Binary metrics at tuned threshold
    report_lines.append(f"\n  Binary Classification Metrics @ threshold={operative_threshold:.2f} (tuned):")
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
    report_lines.append(f"\n  Binary Confusion Matrix (tuned threshold={operative_threshold:.2f}):")
    report_lines.append(f"                  Pred Non-toxic   Pred Toxic")
    report_lines.append(f"    True Non-toxic  {cm[0][0]:>10}   {cm[0][1]:>10}")
    report_lines.append(f"    True Toxic      {cm[1][0]:>10}   {cm[1][1]:>10}")

    report_lines.append(f"\n  Calibration:")
    report_lines.append(f"    Temperature : {optimal_temp:.4f}")
    report_lines.append(f"    (T > 1 = softened predictions, T < 1 = sharpened)")

    report_lines.append(f"\n  Severity labels (anchored to 0.5 binary decision boundary):")
    report_lines.append(f"    0.00-0.20 = Non-toxic       (very confident non-toxic)")
    report_lines.append(f"    0.20-0.50 = Unlikely toxic  (leans non-toxic)")
    report_lines.append(f"    0.50-0.65 = Likely toxic    (leans toxic, lower confidence)")
    report_lines.append(f"    0.65-0.80 = Moderately toxic (moderately confident toxic)")
    report_lines.append(f"    0.80-1.00 = Highly toxic    (very confident toxic)")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # -- Run example predictions at tuned threshold
    run_molecule_examples(model, tokenizer, device, threshold=operative_threshold)

    # ------------------------------------------------------------------
    # EXTERNAL VALIDATION (T3DB + ClinTox — never in training)
    # ------------------------------------------------------------------
    ext_loaders = load_external_validation_datasets(
        data_dir=DATA_DIR,
        tokenizer=tokenizer,
        max_length=128,
        batch_size=32,
        num_workers=0,
    )

    ext_metrics: dict = {}
    ext_report_lines: list = []

    if ext_loaders:
        ext_report_lines.append("\n" + "=" * 60)
        ext_report_lines.append("  EXTERNAL VALIDATION (datasets excluded from training)")
        ext_report_lines.append("=" * 60)
        ext_report_lines.append(
            "  NOTE: These datasets were NEVER used during training.\n"
            "  Results reflect generalisation to unseen distributions."
        )

    for ds_name, ext_loader in ext_loaders.items():
        logger.info(f"Running external validation on {ds_name.upper()} "
                    f"({len(ext_loader)} batches)...")
        ext_probs, ext_labels = collect_probs(model, ext_loader, device)
        m = compute_binary_metrics(ext_probs, ext_labels, threshold=operative_threshold)

        # Specificity = TN / (TN + FP) — not returned by compute_binary_metrics
        cm_ext = m["confusion_matrix"]
        tn, fp = cm_ext[0][0], cm_ext[0][1]
        fn, tp = cm_ext[1][0], cm_ext[1][1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        m["specificity"] = specificity   # store for JSON export

        ext_metrics[ds_name] = m

        t3db_note    = "  <-- headline metric (near-all-toxic corpus)" if ds_name == "t3db"    else ""
        clintox_note = "  <-- headline metric (FDA-approved drug corpus)" if ds_name == "clintox" else ""

        ext_report_lines.append(
            f"\n  {ds_name.upper()} External Validation "
            f"(n={m['n_samples']}, "
            f"{m['n_positive']} toxic / {m['n_negative']} non-toxic):"
        )
        ext_report_lines.append(f"    Threshold used : {operative_threshold:.2f} (best val MCC)")
        ext_report_lines.append(f"    AUC-ROC        : {m['auc_roc']:.4f}")
        ext_report_lines.append(f"    AUC-PRC        : {m['auc_prc']:.4f}")
        ext_report_lines.append(f"    Accuracy       : {m['accuracy']:.4f}")
        ext_report_lines.append(f"    Precision      : {m['precision']:.4f}")
        ext_report_lines.append(f"    Recall         : {m['recall']:.4f}{t3db_note}")
        ext_report_lines.append(f"    Specificity    : {specificity:.4f}{clintox_note}")
        ext_report_lines.append(f"    F1             : {m['f1']:.4f}")
        ext_report_lines.append(f"    MCC            : {m['mcc']:.4f}")
        ext_report_lines.append(f"    Confusion matrix:")
        ext_report_lines.append(f"                      Pred Non-toxic   Pred Toxic")
        ext_report_lines.append(f"      True Non-toxic  {tn:>10}   {fp:>10}")
        ext_report_lines.append(f"      True Toxic      {fn:>10}   {tp:>10}")

    if ext_report_lines:
        ext_text = "\n".join(ext_report_lines)
        print(ext_text)
    else:
        ext_text = ""
        logger.warning("No external validation datasets found — skipping external validation section.")

    # -- Save report (main + external validation combined)
    full_report = report_text + ("\n" + ext_text if ext_text else "")
    report_path = os.path.join(run_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    logger.info(f"Saved evaluation report to {report_path}")

    # -- Save metrics as JSON (includes external validation if available)
    metrics_path = os.path.join(run_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "binary_default_threshold": binary_metrics_default,
            "binary_tuned_threshold":   binary_metrics,
            "threshold_tuning":         threshold_info,
            "calibration":              {"temperature": optimal_temp},
            "external_validation":      ext_metrics,
        }, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")

    print("\n" + "-" * 60)
    print("  Step 6 complete.")
    print(f"    Evaluation report  : {report_path}")
    print(f"    Metrics JSON       : {metrics_path}")
    print(f"    Calibration saved  : {calib_path}")
    print(f"    Threshold JSON     : {threshold_path}")
    print(f"    Operative threshold: {operative_threshold:.2f} (best val MCC)")
    if ext_metrics:
        for ds_name, m in ext_metrics.items():
            print(f"    Ext. val [{ds_name.upper():>7}]: "
                  f"AUC-ROC={m['auc_roc']:.4f}  "
                  f"Recall={m['recall']:.4f}  "
                  f"Spec={m.get('specificity', float('nan')):.4f}")
    print("  Next -> run:  python steps/step7_predict.py")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
