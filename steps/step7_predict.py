#!/usr/bin/env python3
"""
STEP 7 — Predict Toxicity for Custom Molecules
================================================
Reads  :  outputs/last_run.txt          <- pointer to run folder from step 5
           outputs/<run>/lora_weights.pt <- trained LoRA weights

Predicts toxicity for:
  - A single IUPAC name (--molecule)
  - A batch of names from a text file (--input_file, one name per line)
  - Default demo molecules if no input given

Outputs:  prints CoT explanation to console
          optionally saves results to --output_file (JSON)

Run from project root:
  python steps/step7_predict.py
  python steps/step7_predict.py --molecule "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"
  python steps/step7_predict.py --input_file my_molecules.txt --output_file results.json
  python steps/step7_predict.py --molecule "butan-2-one" --egnn_vector
"""

import os
import sys
import json
import argparse
import logging

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join("iupacGPT", "iupac-gpt", "checkpoints", "iupac")
SPM_PATH       = os.path.join("iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model")
OUTPUT_DIR     = "./outputs"

DEFAULT_MOLECULES = [
    # Toxic examples
    "formonitrile",                                      # HCN — cytochrome c oxidase inhibitor
    "nitrobenzene",                                      # carcinogen / methaemoglobin former
    "tetrachloromethane",                                # CCl4 — hepatotoxin
    "phenylarsonic acid",                                # organoarsenic — toxic
    "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione",  # caffeine — CNS stimulant (mild)
    # Non-toxic examples
    "oxidane",                                           # water
    "ethanol",                                           # alcohol — low acute toxicity
    "2-(acetyloxy)benzoic acid",                         # aspirin — therapeutic
    "(2R,3R)-2,3-dihydroxybutanedioic acid",             # L-tartaric acid — food additive
    "(3R,4S,5S,6R)-6-(hydroxymethyl)oxane-2,3,4,5-tetrol",  # D-glucose — nutrients
]


def get_last_run_dir() -> str:
    pointer = os.path.join(OUTPUT_DIR, "last_run.txt")
    if os.path.exists(pointer):
        with open(pointer) as f:
            return f.read().strip()
    runs = sorted([d for d in os.listdir(OUTPUT_DIR)
                   if d.startswith("run_") and os.path.isdir(os.path.join(OUTPUT_DIR, d))],
                  reverse=True)
    if runs:
        return os.path.join(OUTPUT_DIR, runs[0])
    return None


def load_predictor(run_dir: str, device: str):
    """Load ToxGuard predictor from a trained run."""
    import json
    from toxguard.tokenizer import get_tokenizer
    from toxguard.model import ToxGuardModel
    from toxguard.lora import apply_lora_to_model, load_lora_weights, LoRAConfig
    from toxguard.inference import ToxGuardPredictor

    lora_path = os.path.join(run_dir, "lora_weights.pt")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"lora_weights.pt not found in {run_dir}")

    tokenizer = get_tokenizer(vocab_path=SPM_PATH)

    model = ToxGuardModel.from_pretrained_iupacgpt(CHECKPOINT_DIR)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg_path = os.path.join(OUTPUT_DIR, "lora_config.json")
    if os.path.exists(lora_cfg_path):
        with open(lora_cfg_path) as f:
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

    logger.info(f"Loaded trained model from: {run_dir}")
    return ToxGuardPredictor(model, tokenizer, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="STEP 7: Predict toxicity for custom molecules"
    )
    parser.add_argument("--molecule",    type=str, default=None,
                        help="Single IUPAC name to predict")
    parser.add_argument("--input_file",  type=str, default=None,
                        help="Text file with one IUPAC name per line")
    parser.add_argument("--output_file", type=str, default=None,
                        help="JSON file to save results")
    parser.add_argument("--run",         type=str, default=None,
                        help="Run folder to load weights from (default: latest)")
    parser.add_argument("--device",      type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use")
    parser.add_argument("--egnn_vector", action="store_true",
                        help="Also output the 256-dim EGNN vector (for Phase 2)")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  STEP 7 — Predict Toxicity")
    print("=" * 55)

    # ── Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    # ── Find run dir
    run_dir = args.run or get_last_run_dir()
    if run_dir is None or not os.path.exists(run_dir):
        print("\n[ERROR] No trained model found.")
        print("  → Complete training first: python steps/step5_train.py")
        sys.exit(1)

    # ── Load predictor
    try:
        predictor = load_predictor(run_dir, device)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("  → Complete training first: python steps/step5_train.py")
        sys.exit(1)

    # ── Build molecule list
    if args.molecule:
        molecules = [args.molecule]
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"\n[ERROR] Input file not found: {args.input_file}")
            sys.exit(1)
        with open(args.input_file) as f:
            molecules = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(molecules)} molecules from {args.input_file}")
    else:
        molecules = DEFAULT_MOLECULES
        logger.info(f"No input specified — using {len(molecules)} default example molecules")

    # ── Predict
    results = []
    print()

    for iupac_name in molecules:
        print(f"{'═' * 55}")
        print(f"  Molecule: {iupac_name}")
        print(f"{'═' * 55}")

        pred = predictor.predict(iupac_name, return_egnn_vector=args.egnn_vector)
        toxic_str  = "TOXIC" if pred.is_toxic else "Non-toxic"
        print(f"  Prediction    : {toxic_str}")
        print(f"  P(toxic)      : {pred.toxicity_score:.4f}")
        print(f"  Severity      : {pred.severity_label}")
        print(f"  Confidence    : {pred.confidence:.4f}")

        result_dict = {
            "iupac_name":     iupac_name,
            "is_toxic":       pred.is_toxic,
            "toxicity_score": pred.toxicity_score,
            "severity_label": pred.severity_label,
            "confidence":     pred.confidence,
        }

        if args.egnn_vector and pred.egnn_vector:
            result_dict["egnn_vector_preview"] = pred.egnn_vector[:8]
            result_dict["egnn_vector_dim"]      = len(pred.egnn_vector)
            print(f"\n  EGNN Vector (dim=256, for Phase 2):")
            print(f"    First 8 values: {[round(v, 4) for v in pred.egnn_vector[:8]]}")
            print(f"    Full vector ready for EGNN node feature input")

        results.append(result_dict)

    # ── Save results
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")

    print("\n" + "─" * 55)
    print(f"  Step 7 complete. Predicted {len(molecules)} molecule(s).")
    if args.output_file:
        print(f"  Results saved to: {args.output_file}")
    if args.egnn_vector:
        print("  EGNN vectors included (256-dim, ready for Phase 2 EGNN input)")
    print("─" * 55 + "\n")


if __name__ == "__main__":
    main()
