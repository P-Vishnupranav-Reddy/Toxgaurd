# ToxGuard Comprehensive Improvement Plan

## Background & Context

After thorough review of the entire ToxGuard project (all source files, training config, evaluation results, and web research), here is a comprehensive analysis and plan addressing each of the points you raised.

---

## Your Questions — Answered

### Q1: "Are there other IUPAC models better than IUPACGPT?"

**No — IUPACGPT is currently the only language model pre-trained specifically on IUPAC nomenclature.**

From my web research:
- **ChemBERTa, MolBERT, ChemGPT** — all trained exclusively on SMILES strings
- **molT5, T5Chem** — use SMILES ↔ text translation but not IUPAC-native pretraining
- **nach0** — encoder-decoder LLM covering scientific text + molecular strings, but not IUPAC-specific
- **CoTox** (Aug 2025) — uses IUPAC names as input to LLMs like GPT-4o for Chain-of-Thought toxicity reasoning, but it's a *prompting framework* not a pre-trained model

**Recommendation: Improve the existing IUPACGPT architecture** rather than switching models. The original IUPACGPT paper shows it outperforms smilesGPT on property prediction tasks, and its IUPAC-native tokenizer gives semantic understanding of chemical nomenclature that SMILES-based models lack. The improvements below can push AUC-ROC from 0.84 → 0.90+ without changing the backbone.

---

### Q2: "How to handle failed IUPAC resolutions (128KB of failures)?"

Your decision to drop unresolvable molecules was **correct** — attempting to force SMILES fallbacks would fundamentally break the architecture since IUPACGPT's tokenizer was trained on IUPAC syntax, not SMILES.

**Better approaches to reduce failed resolutions:**

| Strategy | Description | Effort |
|----------|-------------|--------|
| **Multi-API cascade (already done)** | PubChem → ChemSpider → NCI CIR | ✅ Already implemented |
| **RDKit local conversion** | Use `rdkit.Chem.inchi.InchiToInchi` + name lookup for simple molecules | Medium |
| **Programmatic IUPAC generation** | Use OPSIN (Open Parser for Systematic IUPAC Nomenclature) Java library for direct SMILES→IUPAC | Medium |
| **Accept partial coverage** | The 128KB of failures is a few hundred compounds out of ~24,000 — ~1-2% loss is acceptable | None |

**My recommendation:** The current approach is pragmatically sound. The data loss from failed resolutions (~1-2%) is minor. Adding OPSIN as a fallback would recover most failures but requires a Java dependency. For now, **focus on the higher-impact improvements below instead**.

---

### Q3: "Is CoT and RAG enough for explainability? What about attention visualization?"

**CoT + RAG and attention visualization serve different purposes — they are complementary, not competing:**

| Approach | What it shows | Audience | Integration |
|----------|--------------|----------|-------------|
| **CoT + RAG** | High-level reasoning ("this molecule is toxic because it contains a nitro group, which is known to cause liver damage based on literature") | End users, clinicians, regulators | At inference time, post-prediction |
| **Attention visualization** | Which specific tokens/substrings in the IUPAC name the model "looked at" when making its decision | ML researchers, model developers | Diagnostic/debugging tool |

**For your project, CoT + RAG is the right choice.** Attention visualization is useful for model debugging but:
- It doesn't replace mechanistic reasoning
- Attention weights can be misleading (attention ≠ importance — well documented in NLP literature)
- It would add implementation complexity without directly improving predictions

**Recommendation: Proceed with CoT + RAG. Skip attention visualization unless you need it for a research paper.**

---

### Q4: "Are confidence calibration and ensemble necessary?"

**Confidence calibration (temperature/Platt scaling) — YES, recommended:**
- Currently [P(toxic) = sigmoid(binary_logit)](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/toxguard/inference.py#35-198) which is often poorly calibrated
- Temperature scaling is a single-parameter post-hoc calibration — trivial to implement
- Makes the output scores more trustworthy: when the model says 80% toxic, it should truly be 80%
- **Effort: Very low (< 30 lines of code)**

**Ensemble (multi-seed LoRA) — OPTIONAL, lower priority:**
- Trains multiple LoRA adapters with different random seeds and averages predictions
- Typical improvement: 1-2% AUC-ROC boost
- Requires N× training time
- **Worth doing only after other improvements are exhausted**

---

### Q5: "Should LoRA settings be changed?" (Screenshot 1)

Your current LoRA config: `r=16, alpha=32, dropout=0.1, targets=[c_attn, c_proj, c_fc]`

The screenshot suggests: `Apply LoRA only to q_proj, v_proj` with `r=8, alpha=16, dropout=0.05`

**My analysis — your current config is better:**

| Setting | Screenshot suggestion | Current | My recommendation |
|---------|----------------------|---------|-------------------|
| Target modules | `q_proj, v_proj` only | `c_attn, c_proj, c_fc` | **Keep current** — IUPACGPT uses GPT-2 Conv1D (`c_attn` = combined QKV), separating q_proj/v_proj isn't applicable here |
| Rank | 8 | 16 | **Keep 16** — your model is only 7.65M params, rank-16 gives 8.15% trainable which is appropriate |
| Alpha | 16 | 32 | **Keep 32** — alpha = 2× rank is the standard scaling |
| Dropout | 0.05 | 0.1 (config) / 0.2 (training) | **Use 0.15** — compromise; 0.05 is too low given overfitting risk, 0.2 may be too aggressive |

> [!IMPORTANT]
> The screenshot's suggestion of `q_proj, v_proj` assumes a standard QKV-split architecture. IUPACGPT's GPT-2 uses Conv1D where `c_attn` is the *combined* QKV projection. Your current target of `c_attn` already covers Q, K, and V. The suggestion doesn't directly apply.

**One change I DO recommend:** Increase dropout from 0.1 to **0.15** as a compromise.

---

### Q6: "Should model size be increased?" (Screenshot 2)

The screenshot suggests: `n_embd=512, layers=12` (up from `n_embd=256, layers=8`)

**This requires re-pretraining IUPACGPT from scratch** — you cannot simply change the dimensions of a pre-trained checkpoint. The checkpoint you're using (`iupacGPT/iupac-gpt/checkpoints/iupac/`) has weights shaped for 256-dim, 8-layer architecture.

> [!CAUTION]
> Changing `n_embd` or `n_layer` would require re-training the entire IUPACGPT foundation model on the IUPAC corpus, which is a massive undertaking (the original paper used significant compute). **This is NOT a viable quick change.**

**Better alternatives to increase effective model capacity:**
1. **Increase LoRA rank** from 16 → 32 (doubles adapter capacity, easy change)
2. **Add class-weighted loss** (fixes the biggest performance issue — class imbalance)
3. **Fix data issues** (scaffold splitting, severity threshold bug)

These three changes combined should push AUC-ROC from 0.84 → 0.90+ *within the current 256-dim architecture*.

---

## Proposed Changes

### Component 1: Bug Fix — Severity Threshold Inconsistency

#### [MODIFY] [data_pipeline.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/toxguard/data_pipeline.py)

Unify `SEVERITY_THRESHOLDS` to match `model.py`'s values `[0.20, 0.50, 0.65, 0.80]` (which are anchored to the 0.5 binary decision boundary).

```diff
-SEVERITY_THRESHOLDS = [0.20, 0.40, 0.60, 0.80]
+SEVERITY_THRESHOLDS = [0.20, 0.50, 0.65, 0.80]
```

**Why `model.py`'s thresholds are correct:** The 0.50 threshold aligns with the binary classification boundary. Bands 0–1 (P < 0.50) = non-toxic, bands 2–4 (P ≥ 0.50) = toxic. The `data_pipeline.py` thresholds at 0.40 and 0.60 break this alignment.

---

### Component 2: Class Imbalance — Weighted BCE + Focal Loss

#### [MODIFY] [model.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/toxguard/model.py)

Add class-weighted BCE loss to counteract the 61.5%/38.5% toxic/non-toxic imbalance. Also add a **focal loss** option (γ=2) to down-weight easy examples and focus on hard cases.

Changes:
1. Add `pos_weight` parameter to `ToxGuardModel.__init__()` for class-weighted BCE
2. Add `use_focal_loss` flag with focal loss implementation (γ=2, α=0.25)
3. Modify `_compute_loss()` to apply class weighting
4. Auto-compute `pos_weight` from training data distribution

#### [MODIFY] [data_pipeline.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/toxguard/data_pipeline.py)

Add a utility function `compute_class_weights()` that scans the combined dataset and returns `pos_weight = n_negative / n_positive` for use in the loss function.

#### [MODIFY] [step5_train.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/steps/step5_train.py)

Add CLI args `--use_focal_loss`, `--focal_gamma`, `--focal_alpha`, and `--auto_class_weight`. Pass class weights to the model before training.

---

### Component 3: Proper Batch Inference

#### [MODIFY] [inference.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/toxguard/inference.py)

Replace sequential `predict_batch()` loop with proper GPU-batched inference:
1. Tokenize all inputs
2. Pad variable-length sequences in a single batch
3. Run a single forward pass
4. Split results back to individual predictions

This provides 10–100× speedup for large-scale screening.

---

### Component 4: Scaffold Splitting

#### [MODIFY] [data_pipeline.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/toxguard/data_pipeline.py)

Add `split_method` parameter to `prepare_combined_dataset()`:
- `"random"` — current behavior (stratified random split)
- `"scaffold"` — Bemis-Murcko scaffold-based splitting using RDKit

This requires loading SMILES from the CSVs and computing Murcko scaffolds for grouping. Molecules with the same scaffold backbone go to the same split.

#### [MODIFY] [step5_train.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/steps/step5_train.py)

Add `--split_method` CLI arg (default: `"random"`, options: `"random"`, `"scaffold"`).

---

### Component 5: LoRA Dropout Adjustment  

#### [MODIFY] [lora_config.json](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/outputs/lora_config.json)

Update LoRA dropout from 0.1 → 0.15.

#### [MODIFY] [step5_train.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/steps/step5_train.py)

Change default `--lora_dropout` from 0.2 → 0.15.

---

### Component 6: Temperature Scaling (Confidence Calibration)

#### [NEW] [calibration.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/toxguard/calibration.py)

Implement temperature scaling as a post-training calibration step:
1. `TemperatureScaler` class with a single learnable temperature parameter
2. `calibrate()` method that optimizes temperature on the validation set
3. `calibrated_predict()` that applies temperature scaling before sigmoid

#### [MODIFY] [step6_evaluate.py](file:///c:/Users/vishn/OneDrive/Desktop/ToxGaurd/steps/step6_evaluate.py)

Add calibration step after loading the model — calibrate temperature on validation set, then report calibrated metrics alongside uncalibrated ones.

---

## Summary of Impact

| Change | Expected Impact | Effort |
|--------|----------------|--------|
| Fix severity thresholds | Bug fix — correct severity labels | Trivial |
| Class-weighted BCE/focal loss | **AUC-ROC ↑ 2-5%**, FPR ↓ significantly | Low |
| Batched inference | **10-100× speedup** for screening | Low |
| Scaffold splitting | More realistic performance estimates | Medium |
| LoRA dropout 0.15 | Marginally better generalization | Trivial |
| Temperature scaling | Better-calibrated probability scores | Low |

---

## Verification Plan

### Automated Verification

1. **Severity threshold fix** — Verify by searching both files for `SEVERITY_THRESHOLDS` and confirming identical values:
   ```
   grep -n "SEVERITY_THRESHOLDS" toxguard/model.py toxguard/data_pipeline.py
   ```

2. **Class-weighted BCE** — Run a quick training test to verify class weights are applied:
   ```
   python steps/step5_train.py --max_epochs 1 --auto_class_weight --batch_size 8
   ```
   Verify the log output shows "Using class weight pos_weight=..." and the training completes without errors.

3. **Focal loss** — Same as above but with focal loss:
   ```
   python steps/step5_train.py --max_epochs 1 --use_focal_loss --focal_gamma 2.0 --batch_size 8
   ```

4. **Batched inference** — Run step7_predict.py and verify predictions match the sequential version:
   ```
   python steps/step7_predict.py
   ```

5. **Scaffold splitting** — Run training with scaffold split and verify the split produces valid results:
   ```
   python steps/step5_train.py --max_epochs 1 --split_method scaffold --batch_size 8
   ```

### Manual Verification

6. **Full re-training comparison** — After all improvements are applied, run a full training (40 epochs) and compare against the baseline metrics (AUC-ROC=0.8375, Accuracy=77.08%, MCC=0.511). This requires user intervention since it takes significant GPU time:
   ```
   python steps/step5_train.py --auto_class_weight --split_method scaffold
   python steps/step6_evaluate.py
   ```
   The user should verify that AUC-ROC improves and FPR decreases.
