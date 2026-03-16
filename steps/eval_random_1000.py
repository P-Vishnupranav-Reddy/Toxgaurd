#!/usr/bin/env python3
"""
Random 1000-Molecule Evaluation
=================================
Samples 1000 molecules at random from all datasets, runs model inference
(no CoT), and reports P(toxic) vs actual label.

Datasets sampled (proportionally by size):
  data/toxcast_final.csv
  data/tox21_final.csv
  data/clintox_final.csv
  data/herg_final.csv
  data/dili_final.csv
  data/common_molecules_final.csv
  data/t3db_processed.csv

Outputs:
  Console : per-molecule table + summary metrics
  JSON    : --output_file path (optional)

Run from project root:
  python steps/eval_random_1000.py
  python steps/eval_random_1000.py --n 500
  python steps/eval_random_1000.py --n 1000 --seed 99
  python steps/eval_random_1000.py --output_file results/random_eval.json
  python steps/eval_random_1000.py --run outputs/run_20260227_234540
"""

import os
import sys
import json
import argparse
import logging

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join("iupacGPT", "iupac-gpt", "checkpoints", "iupac")
SPM_PATH       = os.path.join("iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model")
OUTPUT_DIR     = "./outputs"
DATA_DIR       = "./data"

DATASETS = {
    "toxcast": os.path.join(DATA_DIR, "toxcast_final.csv"),
    "tox21":   os.path.join(DATA_DIR, "tox21_final.csv"),
    "clintox": os.path.join(DATA_DIR, "clintox_final.csv"),
    "herg":    os.path.join(DATA_DIR, "herg_final.csv"),
    "dili":    os.path.join(DATA_DIR, "dili_final.csv"),
    "common":  os.path.join(DATA_DIR, "common_molecules_final.csv"),
    "t3db":    os.path.join(DATA_DIR, "t3db_processed.csv"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_last_run_dir() -> str:
    pointer = os.path.join(OUTPUT_DIR, "last_run.txt")
    if os.path.exists(pointer):
        with open(pointer) as f:
            return f.read().strip()
    runs = sorted(
        [d for d in os.listdir(OUTPUT_DIR)
         if d.startswith("run_") and os.path.isdir(os.path.join(OUTPUT_DIR, d))],
        reverse=True,
    )
    return os.path.join(OUTPUT_DIR, runs[0]) if runs else None


def load_all_datasets() -> pd.DataFrame:
    """Load and merge all datasets into a single DataFrame with source tag."""
    frames = []
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            logger.warning(f"Dataset not found, skipping: {path}")
            continue
        df = pd.read_csv(path, usecols=lambda c: c in ("iupac_name", "is_toxic"))
        df = df.dropna(subset=["iupac_name"]).copy()
        df["iupac_name"] = df["iupac_name"].str.strip()
        df = df[df["iupac_name"] != ""]
        df["source"] = name
        frames.append(df[["iupac_name", "is_toxic", "source"]])
        logger.info(f"  {name}: {len(df):,} molecules")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="iupac_name")
    return combined


def sample_molecules(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n molecules proportionally from each dataset, balanced 50/50 toxic/non-toxic.
    Falls back to plain random sample if not enough balanced data.
    """
    rng = np.random.default_rng(seed)
    n_each = n // 2

    toxic_pool     = df[df["is_toxic"] == 1].sample(frac=1, random_state=seed)
    nontoxic_pool  = df[df["is_toxic"] == 0].sample(frac=1, random_state=seed)

    toxic_sample    = toxic_pool.head(min(n_each, len(toxic_pool)))
    nontoxic_sample = nontoxic_pool.head(min(n_each, len(nontoxic_pool)))

    sampled = pd.concat([toxic_sample, nontoxic_sample]).sample(frac=1, random_state=seed)
    logger.info(f"Sampled {len(sampled):,} molecules "
                f"(toxic={len(toxic_sample)}, non-toxic={len(nontoxic_sample)})")
    return sampled.reset_index(drop=True)


def load_model(run_dir: str, device: str):
    """Load ToxGuard model (no CoT explainer)."""
    from toxguard.tokenizer import get_tokenizer
    from toxguard.model import ToxGuardModel
    from toxguard.lora import apply_lora_to_model, load_lora_weights, LoRAConfig

    lora_path = os.path.join(run_dir, "lora_weights.pt")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"lora_weights.pt not found in {run_dir}")

    tokenizer = get_tokenizer(vocab_path=SPM_PATH)

    model = ToxGuardModel.from_pretrained_iupacgpt(CHECKPOINT_DIR)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA config from run dir first, fall back to global
    run_cfg_path    = os.path.join(run_dir, "config.json")
    global_cfg_path = os.path.join(OUTPUT_DIR, "lora_config.json")

    if os.path.exists(run_cfg_path):
        with open(run_cfg_path) as f:
            cfg = json.load(f)
        lora_config = LoRAConfig(
            r=cfg.get("lora_rank", 32),
            alpha=cfg.get("lora_alpha", 64.0),
            dropout=cfg.get("lora_dropout", 0.2),
            target_modules=cfg.get("lora_targets", "c_attn,c_proj,c_fc").split(","),
            fan_in_fan_out=True,
        )
    elif os.path.exists(global_cfg_path):
        with open(global_cfg_path) as f:
            cfg = json.load(f)
        lora_config = LoRAConfig(
            r=cfg["r"], alpha=cfg["alpha"], dropout=cfg["dropout"],
            target_modules=cfg.get("target_modules", ["c_attn", "c_proj", "c_fc"]),
            fan_in_fan_out=True,
        )
    else:
        lora_config = LoRAConfig()

    model, _ = apply_lora_to_model(model, lora_config)
    model = load_lora_weights(model, lora_path)
    model.to(torch.device(device))
    model.eval()

    return model, tokenizer


@torch.no_grad()
def predict_batch(model, tokenizer, names: list[str], device: str,
                  max_length: int = 300, batch_size: int = 64) -> list[float]:
    """Run batched inference; returns list of P(toxic) floats."""
    dev = torch.device(device)
    all_probs = []

    for start in range(0, len(names), batch_size):
        chunk = names[start: start + batch_size]
        encoded = [tokenizer(n)["input_ids"] for n in chunk]

        # Prepend BOS token id
        bos_id = tokenizer._convert_token_to_id(tokenizer.unk_token)
        sequences = [
            torch.tensor([bos_id] + ids[:max_length - 1], dtype=torch.long)
            for ids in encoded
        ]

        # Pad to same length within batch
        max_len = max(s.size(0) for s in sequences)
        pad_id  = tokenizer.pad_token_id or 0
        input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            input_ids[i, : seq.size(0)] = seq
            attn_mask[i, : seq.size(0)] = 1

        input_ids = input_ids.to(dev)
        attn_mask = attn_mask.to(dev)

        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_attentions=False, return_hidden=False)
        probs = torch.sigmoid(out.binary_logits).cpu().tolist()
        all_probs.extend(probs)

    return all_probs


def severity_label(p: float) -> str:
    if p < 0.20: return "Non-toxic"
    if p < 0.50: return "Unlikely toxic"
    if p < 0.65: return "Likely toxic"
    if p < 0.80: return "Moderately toxic"
    return "Highly toxic"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Random 1000-molecule evaluation (no CoT)"
    )
    parser.add_argument("--n",           type=int,   default=1000,
                        help="Number of molecules to evaluate (default: 1000)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--run",         type=str,   default=None,
                        help="Run folder to load weights from (default: latest)")
    parser.add_argument("--device",      type=str,   default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size",  type=int,   default=64,
                        help="Inference batch size (default: 64)")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--output_file", type=str,   default=None,
                        help="Save full results as JSON")
    parser.add_argument("--show_wrong",  action="store_true",
                        help="Print only wrong predictions in the table")
    args = parser.parse_args()

    # ── Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("\n" + "=" * 65)
    print("  Random Molecule Evaluation  (no CoT)")
    print("=" * 65)
    logger.info(f"Device: {device} | N: {args.n} | Seed: {args.seed}")

    # ── Load run
    run_dir = args.run or get_last_run_dir()
    if run_dir is None or not os.path.exists(run_dir):
        print("\n[ERROR] No trained model found. Run step5 first.")
        sys.exit(1)
    logger.info(f"Loading weights from: {run_dir}")

    # ── Load datasets
    logger.info("Loading datasets...")
    all_df = load_all_datasets()
    logger.info(f"Total unique molecules across all datasets: {len(all_df):,}")

    # ── Sample
    sample = sample_molecules(all_df, args.n, args.seed)

    # ── Load model
    logger.info("Loading model...")
    model, tokenizer = load_model(run_dir, device)

    # ── Inference
    logger.info(f"Running inference on {len(sample)} molecules...")
    names  = sample["iupac_name"].tolist()
    labels = sample["is_toxic"].tolist()
    probs  = predict_batch(model, tokenizer, names, device,
                           batch_size=args.batch_size)

    # ── Build results
    records = []
    tp = fp = tn = fn = 0
    for iupac, actual, p in zip(names, labels, probs):
        predicted = int(p >= args.threshold)
        correct   = predicted == actual
        if   actual == 1 and predicted == 1: tp += 1
        elif actual == 0 and predicted == 1: fp += 1
        elif actual == 0 and predicted == 0: tn += 1
        else:                                fn += 1
        records.append({
            "iupac_name":   iupac,
            "actual":       actual,
            "p_toxic":      round(p, 4),
            "predicted":    predicted,
            "correct":      correct,
            "severity":     severity_label(p),
            "source":       sample.loc[sample["iupac_name"] == iupac, "source"].values[0],
        })

    # ── Print table
    print()
    header = f"  {'IUPAC Name':<45} {'Actual':<8} {'P(toxic)':<10} {'Predicted':<11} {'OK':<4} {'Source'}"
    print(header)
    print("  " + "-" * 100)

    to_print = [r for r in records if (not args.show_wrong or not r["correct"])]
    for r in to_print:
        actual_str = "TOXIC" if r["actual"] else "non-toxic"
        pred_str   = "TOXIC" if r["predicted"] else "non-toxic"
        ok         = "✓" if r["correct"] else "✗"
        name_disp  = r["iupac_name"][:43] + ".." if len(r["iupac_name"]) > 45 else r["iupac_name"]
        print(f"  {name_disp:<45} {actual_str:<8} {r['p_toxic']:<10.4f} {pred_str:<11} {ok:<4} {r['source']}")

    # ── Summary metrics
    total   = len(records)
    correct = sum(r["correct"] for r in records)
    acc     = correct / total

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUROC via sklearn if available
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
        auroc_str = f"{auroc:.4f}"
        auprc_str = f"{auprc:.4f}"
    except Exception:
        auroc_str = "n/a"
        auprc_str = "n/a"

    # Per-source breakdown
    sources = sample["source"].unique()
    source_stats = {}
    for src in sources:
        src_idx = [i for i, r in enumerate(records) if r["source"] == src]
        src_correct = sum(records[i]["correct"] for i in src_idx)
        source_stats[src] = {"n": len(src_idx), "correct": src_correct,
                             "acc": src_correct / len(src_idx) if src_idx else 0}

    print("\n  " + "=" * 65)
    print("  SUMMARY")
    print("  " + "=" * 65)
    print(f"  Molecules evaluated : {total}")
    print(f"  Correct predictions : {correct}  ({acc:.1%})")
    print(f"  Accuracy            : {acc:.4f}")
    print(f"  Precision           : {prec:.4f}  (of predicted toxic, how many truly are)")
    print(f"  Recall / Sensitivity: {rec:.4f}  (of actual toxic, how many caught)")
    print(f"  Specificity         : {spec:.4f}  (of actual non-toxic, how many correctly non-toxic)")
    print(f"  F1 Score            : {f1:.4f}")
    print(f"  AUC-ROC             : {auroc_str}")
    print(f"  AUC-PRC             : {auprc_str}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred Toxic  Pred Non-toxic")
    print(f"  Actual Toxic     {tp:>5}          {fn:>5}")
    print(f"  Actual Non-toxic {fp:>5}          {tn:>5}")
    print(f"\n  Per-source Accuracy:")
    for src, s in sorted(source_stats.items()):
        bar_len = int(s["acc"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {src:<10} n={s['n']:>4}  acc={s['acc']:.1%}  [{bar}]")
    print("  " + "=" * 65 + "\n")

    # ── Save JSON
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        output = {
            "run_dir":    run_dir,
            "n_evaluated": total,
            "seed":       args.seed,
            "threshold":  args.threshold,
            "metrics": {
                "accuracy":     round(acc,  4),
                "precision":    round(prec, 4),
                "recall":       round(rec,  4),
                "specificity":  round(spec, 4),
                "f1":           round(f1,   4),
                "auroc":        auroc_str,
                "auprc":        auprc_str,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            },
            "per_source": {k: {"n": v["n"], "accuracy": round(v["acc"], 4)}
                           for k, v in source_stats.items()},
            "predictions": records,
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved → {args.output_file}")


if __name__ == "__main__":
    main()
