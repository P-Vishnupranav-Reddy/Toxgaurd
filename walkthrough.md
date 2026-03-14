# ToxGuard / IUPAC-GPT — Comprehensive Project Analysis

## Overview

ToxGuard is a **molecular toxicity prediction system** that fine-tunes the [IUPACGPT](https://github.com/iupacgpt/iupac-gpt) language model (a GPT-2 variant trained on IUPAC chemical nomenclature) using **LoRA (Low-Rank Adaptation)** to predict whether a molecule is toxic or non-toxic from its IUPAC name alone.

```
IUPAC Name → SentencePiece Tokenizer → GPT-2 (frozen) + LoRA → Binary Head → P(toxic) → Severity Label
```

### Key Numbers

| Metric | Value |
|--------|-------|
| Backbone | GPT-2 (8 layers, 8 heads, 256 dim) |
| Total parameters | 7.65M |
| Trainable (LoRA + head) | ~624K (8.15%) |
| Tokenizer | SentencePiece, vocab 1,491 |
| Training datasets | 7 (ToxCast, Tox21, T3DB, ClinTox, hERG, DILI, Common Molecules) |
| Total compounds | ~23,800 |
| Test AUC-ROC | **0.8375** |
| Test AUC-PRC | **0.8801** |
| Test Accuracy | **77.08%** |
| Test F1 | **0.817** |
| Test MCC | **0.511** |

---

## Architecture Deep Dive

### Core Library (`toxguard/`)

| File | Purpose | Lines |
|------|---------|-------|
| [model.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/model.py) | [ToxGuardModel](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/model.py#186-336) wrapping GPT-2 + binary head + EGNN projection; [ToxGuardLitModel](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/model.py#342-503) for PyTorch Lightning training | 503 |
| [lora.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/lora.py) | Custom LoRA implementation (LoRALayer, apply/save/load), targets `c_attn`, `c_proj`, `c_fc` | 271 |
| [tokenizer.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/tokenizer.py) | [ToxGuardTokenizer](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/tokenizer.py#9-37) (T5-based SentencePiece wrapper for IUPAC names) | 90 |
| [data_pipeline.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/data_pipeline.py) | [MoleculeDataset](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/data_pipeline.py#288-353), [T3DBDataset](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/data_pipeline.py#364-377), [ToxicityDataset](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/data_pipeline.py#379-401) (combined), collator, stratified splitting | 561 |
| [inference.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/inference.py) | [ToxGuardPredictor](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/inference.py#35-198) with single/batch prediction + EGNN vector extraction | 198 |

### Pipeline Scripts (`steps/`)

| Script | Purpose |
|--------|---------|
| [step1_download_data.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/steps/step1_download_data.py) | Downloads ToxCast, Tox21, ClinTox from MoleculeNet; hERG and DILI via PyTDC; processes local T3DB |
| [step2_preprocess.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/steps/step2_preprocess.py) | Canonicalizes SMILES (RDKit), computes binary labels, cross-dataset deduplication |
| [step3_smiles_to_iupac.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/steps/step3_smiles_to_iupac.py) | Resolves SMILES → IUPAC via PubChem → ChemSpider → NCI CIR cascade (cached) |
| [step4_verify_lora.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/steps/step4_verify_lora.py) | Verifies LoRA injection, test forward pass, saves LoRA config |
| [step5_train.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/steps/step5_train.py) | Full training with PyTorch Lightning (early stopping, cosine LR, TensorBoard) |
| [step6_evaluate.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/steps/step6_evaluate.py) | Evaluates on held-out test set; computes AUC-ROC, AUC-PRC, F1, MCC, confusion matrix |
| [step7_predict.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/steps/step7_predict.py) | CLI inference on custom molecules with severity labels |

### Evaluation Results (Latest Run)

```
Binary Confusion Matrix:
                  Pred Non-toxic   Pred Toxic
  True Non-toxic         617          299
  True Toxic             247         1219
```

- **299 false positives** (non-toxic compounds predicted as toxic — 32.6% FPR)
- **247 false negatives** (toxic compounds predicted as non-toxic — 16.8% FNR)

---

## ✅ Pros (Strengths)

### 1. Novel and Well-Motivated Approach
- Uses **IUPAC names as text input** instead of SMILES/fingerprints — leverages the chemical nomenclature's structural encoding naturally via language models
- Builds on IUPACGPT, a pre-trained model that already "understands" chemical name syntax

### 2. Clean, Well-Organized Codebase
- Clear **7-step pipeline** with each step self-contained and re-runnable
- Comprehensive docstrings and architecture diagrams in code comments
- Good README with step-by-step setup, table of datasets, and model architecture specs
- Proper [__init__.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/__init__.py) with clean public API

### 3. Parameter-Efficient Fine-Tuning
- **LoRA** (rank-16) trains only 8.15% of parameters (~624K vs 7.65M total)
- Custom LoRA implementation handles GPT-2's Conv1D layers correctly (`fan_in_fan_out=True`)
- Supports merge/unmerge for efficient inference
- LoRA weights checkpoint is only ~2.5MB vs ~30MB for full model

### 4. Robust Data Pipeline
- **7 diverse toxicity datasets** covering different toxicity endpoints (cardiotoxicity, liver injury, clinical trial failure, etc.)
- Smart **cross-dataset deduplication** with clear priority ordering
- **SMILES canonicalization** via RDKit prevents duplicate compounds across datasets
- LD50-to-toxicity score conversion using WHO/GHS classification
- **Cascade API resolution** (PubChem → ChemSpider → NCI CIR) with persistent caching

### 5. Good Training Infrastructure
- PyTorch Lightning for clean training loops
- Early stopping on `val_auroc` to prevent overfitting
- Cosine annealing with linear warmup
- TensorBoard logging, gradient clipping, mixed precision (FP16)
- Configurable via CLI arguments
- Reproducibility via `seed_everything(42)`

### 6. Forward-Thinking Design
- **EGNN embedding vector** (256-dim) output ready for Phase 2 graph neural network integration
- Severity labels derived from prediction probability — flexible and interpretable
- Label smoothing (ε=0.1) on BCE loss for better calibration

---

## ❌ Cons (Weaknesses)

### 1. Moderate Performance — Room for Improvement
- **AUC-ROC = 0.8375** is decent but not state-of-the-art for toxicity prediction
- **Accuracy 77%** means roughly 1 in 4 predictions is wrong
- **MCC = 0.511** indicates only moderate agreement (MCC of 1.0 is perfect)
- **32.6% false positive rate** is concerning for practical applications — many safe compounds flagged as toxic

### 2. Class Imbalance Not Addressed
- Dataset is **61.5% toxic / 38.5% non-toxic** (1,466 vs 916 test samples)
- T3DB contributes 3,505 samples that are **all toxic** — biases model toward predicting toxic
- No class weighting, focal loss, or oversampling to compensate
- This likely explains the high false positive rate

### 3. Small Model for the Task
- GPT-2 with only **8 layers, 256 dim** is very small by modern standards
- Vocabulary of **1,491 tokens** may be too small for complex IUPAC nomenclature
- Larger pre-trained chemical language models (ChemBERTa, MolBERT, etc.) might perform better

### 4. IUPAC Name Dependency is a Bottleneck
- SMILES → IUPAC resolution via external APIs (PubChem, ChemSpider) introduces fragility
- Molecules without IUPAC names are **dropped entirely** — data loss
- [failed_resolve.csv](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/data/failed_resolve.csv) (128KB) suggests a substantial number of compounds couldn't be resolved
- At inference time, user must provide exact IUPAC names — less practical than SMILES input

### 5. No Automated Testing
- Zero unit tests anywhere in the codebase
- [verify_setup.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/verify_setup.py) only checks imports and model loading, not correctness
- No CI/CD pipeline or test framework configured
- Makes refactoring risky and bug detection manual

### 6. Batch Inference is Sequential
- [predict_batch()](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/inference.py#163-179) in [inference.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/inference.py#L163-L178) simply loops `self.predict()` — no GPU batching
- For large-scale screening, this is very slow
- [get_egnn_vectors()](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/inference.py#180-198) has the same issue

### 7. No Explainability
- Binary toxic/non-toxic prediction with no mechanistic reasoning
- No attention visualization or feature attribution
- No indication of **which toxicity endpoint** was triggered (cardio? liver? etc.)
- Severity labels are just probability thresholds — not truly informative

### 8. Inconsistent Severity Thresholds
- [model.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/model.py) defines `SEVERITY_THRESHOLDS = [0.20, 0.50, 0.65, 0.80]`
- [data_pipeline.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/data_pipeline.py) defines `SEVERITY_THRESHOLDS = [0.20, 0.40, 0.60, 0.80]`
- These are different! Could cause confusion if used inconsistently

### 9. No Cross-Validation
- Single random 80/10/10 split — evaluation may not generalize well
- No k-fold cross-validation or scaffold splitting (scaffold split is standard in cheminformatics)
- Structurally similar molecules may leak between train/test sets

### 10. Hardcoded Paths
- `CHECKPOINT_DIR`, `SPM_PATH`, `DATA_DIR` are hardcoded relative paths in multiple scripts
- Not configurable via environment variables
- Breaks if project root changes or scripts are called from different directories

---

## 💡 Improvement Suggestions

### High Priority

| # | Suggestion | Impact | Effort |
|---|-----------|--------|--------|
| 1 | **Address class imbalance** — Add class-weighted BCE loss or focal loss; consider down-sampling T3DB (all-toxic dataset) | Reduce false positive rate significantly | Low |
| 2 | **Add scaffold splitting** — Use Murcko scaffold-based train/test split instead of random to prevent data leakage from structurally similar molecules | More realistic performance estimates | Medium |
| 3 | **Implement proper batch inference** — Pad and batch tokenized inputs in [predict_batch()](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/inference.py#163-179) for GPU-accelerated screening | 10-100× inference speedup | Low |
| 4 | **Fix inconsistent severity thresholds** — Unify thresholds between [model.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/model.py) and [data_pipeline.py](file:///c:/Users/PAVAN%20K%20AITHAL/OneDrive/Desktop/PROJECT/Toxgaurd/toxguard/data_pipeline.py) | Bug fix | Trivial |
| 5 | **Add unit tests** — At minimum for tokenizer, LoRA layer math, loss computation, and data pipeline correctness | Reliability and refactorability | Medium |

### Medium Priority

| # | Suggestion | Impact | Effort |
|---|-----------|--------|--------|
| 6 | **Add multi-task endpoint prediction** — Instead of a single binary output, predict per-endpoint toxicity (cardio, liver, mutagenicity, etc.) | Much more informative predictions | High |
| 7 | **Attention-based explainability** — Extract and visualize attention weights over IUPAC name substrings to show which chemical groups drive the prediction | Interpretability and trust | Medium |
| 8 | **Try larger backbones** — Experiment with ChemBERTa-2 or a larger GPT-2 variant for potentially better performance | Performance boost | Medium |
| 9 | **Add confidence calibration** — Apply temperature scaling or Platt scaling post-training so P(toxic) is a true probability | Better-calibrated scores | Low |
| 10 | **Web UI / API** — Build a Flask/FastAPI REST endpoint or Streamlit/Gradio demo for interactive prediction | Usability and demonstration | Medium |

### Lower Priority

| # | Suggestion | Impact | Effort |
|---|-----------|--------|--------|
| 11 | **Ensemble models** — Train multiple LoRA adapters with different seeds and average predictions | Marginal performance boost | Low |
| 12 | **Data augmentation** — Use SMILES randomization or IUPAC name variants (synonyms) to increase effective training set | Better generalization | Medium |
| 13 | **ONNX export** — Export the merged model to ONNX for deployment without PyTorch dependency | Production deployment | Medium |
| 14 | **Docker containerization** — Package the entire pipeline in Docker for reproducible environments | Reproducibility | Low |
| 15 | **Add external validation** — Evaluate on completely independent benchmarks (e.g., MoleculeNet leaderboard, TDC benchmark) | Publishable results | Medium |

---

## Summary

ToxGuard is a **well-structured, clearly documented** project with a creative approach (using IUPAC names as chemical descriptors via language models). The 7-step pipeline is clean and the LoRA fine-tuning is implemented correctly. However, performance (77% accuracy, AUC-ROC 0.84) is limited by **class imbalance**, **small model size**, and **lack of structural data leakage prevention**. The highest-impact improvements would be addressing class imbalance (suggestion #1), adding scaffold splitting (suggestion #2), and implementing proper batch inference (suggestion #3) — all achievable with moderate effort.
