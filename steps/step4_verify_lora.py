#!/usr/bin/env python3
"""
STEP 4 — Apply & Verify LoRA Adapters
=======================================
Reads  :  iupacGPT/iupac-gpt/checkpoints/iupac/  (IUPACGPT pretrained weights)
Outputs:  outputs/lora_config.json   ← the LoRA config that will be used for training

What it does:
  1. Loads the IUPACGPT base model (7.1M params, GPT-2, 8 layers, 256 dim)
  2. Injects LoRA adapters into attention + MLP layers (c_attn, c_proj, c_fc)
  3. Freezes base model weights — only LoRA + toxicity head are trainable
  4. Prints a full parameter summary table
  5. Runs a test forward pass to confirm the model works end-to-end
  6. Saves the LoRA config to disk for use in training

Run from project root:
  python steps/step4_verify_lora.py
  python steps/step4_verify_lora.py --rank 4    # try a smaller rank
  python steps/step4_verify_lora.py --rank 32   # rank 32 (default)
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
OUTPUT_DIR     = "./outputs"


def print_parameter_table(model, stats: dict):
    """Print a formatted parameter summary."""
    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │  PARAMETER SUMMARY                              │")
    print("  ├─────────────────────────────────────────────────┤")
    print(f"  │  Total parameters   : {stats['total_params']:>12,}            │")
    print(f"  │  Trainable (LoRA+head): {stats['trainable_params']:>10,}            │")
    print(f"  │  Frozen (base model): {stats['frozen_params']:>12,}            │")
    print(f"  │  LoRA parameters    : {stats['lora_params']:>12,}            │")
    print(f"  │  LoRA layers applied: {stats['num_lora_layers']:>12}            │")
    print(f"  │  Trainable %        : {stats['trainable_pct']:>11.2f}%            │")
    print("  └─────────────────────────────────────────────────┘")

    print("\n  Trainable modules breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name:<55} {param.numel():>10,} params")


def run_forward_pass_test(model, tokenizer):
    """Run a quick forward pass to verify end-to-end wiring is correct."""
    from toxguard.model import score_to_severity_label

    logger.info("Running forward pass test with 'formonitrile'...")

    tokenized = tokenizer("formonitrile")
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    bos = torch.tensor([tokenizer._convert_token_to_id(tokenizer.unk_token)])
    input_ids = torch.cat([bos, input_ids]).unsqueeze(0)  # (1, L)
    attention_mask = torch.ones_like(input_ids)

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    binary_prob = output.toxicity_score.item()  # P(toxic) = sigmoid(binary_logit)
    severity = score_to_severity_label(binary_prob)

    print("\n  Forward pass test results:")
    print(f"    Input tokens        : {input_ids.shape[1]} tokens")
    print(f"    Binary logit        : {output.binary_logits.item():.4f}")
    print(f"    P(toxic)            : {binary_prob:.4f}  -> {severity}")
    print(f"    EGNN vector shape   : {output.hidden_state.shape}  (dim=256)")
    print("    [OK] Forward pass successful!")


def main():
    parser = argparse.ArgumentParser(
        description="STEP 4: Apply and verify LoRA adapters on IUPACGPT"
    )
    parser.add_argument("--rank",    type=int,   default=32,    help="LoRA rank (default: 32)")
    parser.add_argument("--alpha",   type=float, default=64.0,  help="LoRA alpha (default: 64.0 = 2×rank)")
    parser.add_argument("--dropout", type=float, default=0.2,   help="LoRA dropout (default: 0.2)")
    parser.add_argument("--targets", type=str,   default="c_attn,c_proj,c_fc",
                        help="Target modules (default: c_attn,c_proj,c_fc — attention + MLP FFN)")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  STEP 4 — Apply & Verify LoRA Adapters")
    print("=" * 55)

    # ── Check checkpoint exists
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"\n[ERROR] Checkpoint not found: {CHECKPOINT_DIR}")
        sys.exit(1)

    # ── Check tokenizer
    spm_path = os.path.join("iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model")
    if not os.path.exists(spm_path):
        print(f"\n[ERROR] Tokenizer not found: {spm_path}")
        sys.exit(1)

    # ── Load tokenizer
    logger.info("Loading IUPAC tokenizer...")
    from toxguard.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(vocab_path=spm_path)
    logger.info(f"  Vocabulary size: {tokenizer.vocab_size} tokens")

    # ── Load base model
    logger.info("Loading IUPACGPT backbone...")
    from toxguard.model import ToxGuardModel
    model = ToxGuardModel.from_pretrained_iupacgpt(CHECKPOINT_DIR)

    base_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Base model parameters: {base_params:,}")
    logger.info(f"  Architecture: {model.config.n_layer} layers, "
                f"{model.config.n_head} heads, {model.config.n_embd} dim")

    # ── Apply LoRA
    logger.info(f"\nApplying LoRA: rank={args.rank}, alpha={args.alpha}, "
                f"targets={args.targets}")

    from toxguard.lora import apply_lora_to_model, LoRAConfig
    lora_config = LoRAConfig(
        r=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.targets.split(","),
        fan_in_fan_out=True,  # GPT-2 uses Conv1D (transposed weights)
    )
    model, stats = apply_lora_to_model(model, lora_config)

    # ── Print parameter table
    print_parameter_table(model, stats)

    # ── Verify frozen/trainable split
    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\n  Trainable modules: {len(trainable_names)}")
    print(f"  Frozen modules   : {len(frozen_names)}")

    # ── Forward pass test
    model.config.pad_token_id = tokenizer.pad_token_id
    run_forward_pass_test(model, tokenizer)

    # ── Save LoRA config for training step
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_out = os.path.join(OUTPUT_DIR, "lora_config.json")
    config_dict = {
        "r": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": args.targets.split(","),
        "fan_in_fan_out": True,
        "total_params": stats["total_params"],
        "trainable_params": stats["trainable_params"],
        "lora_params": stats["lora_params"],
        "trainable_pct": round(stats["trainable_pct"], 4),
        "checkpoint_dir": CHECKPOINT_DIR,
    }
    with open(config_out, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"\nSaved LoRA config to: {config_out}")

    print("\n" + "─" * 55)
    print("  Step 4 complete.")
    print(f"    LoRA config saved to: {config_out}")
    print(f"    Trainable params: {stats['trainable_params']:,} "
          f"({stats['trainable_pct']:.2f}% of {stats['total_params']:,})")
    print("  Next → run:  python steps/step5_train.py")
    print("─" * 55 + "\n")


if __name__ == "__main__":
    main()
