#!/usr/bin/env python3
"""Quick verification script: checks all ToxGuard components load correctly."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check(name, fn):
    try:
        fn()
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("=" * 50)
print("  ToxGuard — Environment Check")
print("=" * 50)

results = []

# 1. PyTorch + CUDA
print("\n1. PyTorch & CUDA:")
results.append(check("PyTorch import", lambda: __import__("torch")))
import torch
results.append(check(f"CUDA available: {torch.cuda.is_available()}", lambda: None))
if torch.cuda.is_available():
    results.append(check(f"GPU: {torch.cuda.get_device_name(0)}", lambda: None))

# 2. Key dependencies
print("\n2. Dependencies:")
for pkg in ["transformers", "pytorch_lightning", "torchmetrics", "sentencepiece", 
            "pandas", "numpy", "requests"]:
    results.append(check(pkg, lambda p=pkg: __import__(p)))

# 3. ToxGuard modules
print("\n3. ToxGuard modules:")
results.append(check("toxguard.tokenizer", lambda: __import__("toxguard.tokenizer")))
results.append(check("toxguard.lora", lambda: __import__("toxguard.lora")))
results.append(check("toxguard.model", lambda: __import__("toxguard.model")))
results.append(check("toxguard.data_pipeline", lambda: __import__("toxguard.data_pipeline")))
results.append(check("toxguard.inference", lambda: __import__("toxguard.inference")))

# 4. IUPACGPT checkpoint
print("\n4. IUPACGPT Checkpoint:")
ckpt_dir = os.path.join("iupacGPT", "iupac-gpt", "checkpoints", "iupac")
results.append(check(f"config.json exists", 
    lambda: os.path.exists(os.path.join(ckpt_dir, "config.json")) or (_ for _ in ()).throw(FileNotFoundError("not found"))))
has_weights = os.path.exists(os.path.join(ckpt_dir, "model.safetensors")) or os.path.exists(os.path.join(ckpt_dir, "pytorch_model.bin"))
results.append(check(f"model weights exist (safetensors or bin)",
    lambda: has_weights or (_ for _ in ()).throw(FileNotFoundError("not found"))))

# 5. Tokenizer
print("\n5. Tokenizer:")
spm_path = os.path.join("iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model")
results.append(check(f"iupac_spm.model exists", 
    lambda: os.path.exists(spm_path) or (_ for _ in ()).throw(FileNotFoundError("not found"))))

try:
    from toxguard.tokenizer import get_tokenizer
    tok = get_tokenizer(vocab_path=spm_path)
    test_tokens = tok("formonitrile")
    results.append(check(f"Tokenizer works: 'formonitrile' → {len(test_tokens['input_ids'])} tokens", lambda: None))
except Exception as e:
    results.append(check(f"Tokenizer load", lambda: (_ for _ in ()).throw(e)))

# 6. Model load test
print("\n6. Model load:")
try:
    from toxguard.model import ToxGuardModel
    model = ToxGuardModel.from_pretrained_iupacgpt(ckpt_dir)
    total_params = sum(p.numel() for p in model.parameters())
    results.append(check(f"ToxGuardModel loaded ({total_params:,} params)", lambda: None))
    
    from toxguard.lora import apply_lora_to_model, LoRAConfig
    model, stats = apply_lora_to_model(model, LoRAConfig())
    results.append(check(
        f"LoRA applied: {stats['trainable_params']:,} trainable / {stats['total_params']:,} total ({stats['trainable_pct']:.2f}%)",
        lambda: None
    ))
except Exception as e:
    results.append(check(f"Model load", lambda: (_ for _ in ()).throw(e)))

# Summary
print("\n" + "=" * 50)
passed = sum(1 for r in results if r)
total = len(results)
print(f"  Results: {passed}/{total} checks passed")
if passed == total:
    print("  ALL CHECKS PASSED — Ready to train!")
else:
    print(f"  {total - passed} checks failed — fix issues above first.")
print("=" * 50)
