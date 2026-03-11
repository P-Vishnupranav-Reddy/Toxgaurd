# ToxGuard — Architecture & Project Reference

Complete reference for understanding what is happening in this project, how the model works, and what every component does.

---

## Table of Contents

1. [What is IUPAC-GPT?](#1-what-is-iupac-gpt)
2. [Pretrained Model Architecture](#2-pretrained-model-architecture)
3. [The Tokenizer](#3-the-tokenizer)
4. [Full Pipeline: Preprocessing → Training → Inference](#4-full-pipeline)
5. [LoRA: Frozen Backbone + Trainable Adapters](#5-lora-frozen-backbone--trainable-adapters)
6. [Parameter Count Breakdown](#6-parameter-count-breakdown)
7. [What Happens Inside Each Transformer Layer](#7-what-happens-inside-each-transformer-layer)
8. [Why the IUPAC Name Encodes Toxicity](#8-why-the-iupac-name-encodes-toxicity)
9. [The Three Output Heads](#9-the-three-output-heads)
10. [Severity Label System](#10-severity-label-system)
11. [Training Loss](#11-training-loss)
12. [At Inference Time](#12-at-inference-time)
13. [File Map](#13-file-map)

---

## 1. What is IUPAC-GPT?

IUPAC-GPT is a **GPT-2 language model pretrained on IUPAC chemical nomenclature** — the systematic international naming convention for chemical compounds (e.g., "2-acetoxybenzoic acid" is aspirin, "2,4,6-trinitrotoluene" is TNT).

**Original purpose (before ToxGuard):** It was trained autoregressively on a large corpus of IUPAC names from PubChem. During pretraining the model learned to predict the *next token* in an IUPAC name. This forced it to learn:

- Chemical substructure patterns encoded in names (functional groups: "hydroxy-", "amino-", "chloro-", "nitro-")
- Systematic naming rules that encode molecular topology (ring numbering, branching positions, stereochemistry)
- Implicit structural relationships — molecules with similar names have structurally similar features

It was designed for drug discovery: molecular property prediction and de novo molecule generation.

**What we do with it:** We repurpose it as a **toxicity classifier**. Instead of predicting the next token, we extract its learned molecular representation and predict toxicity severity.

---

## 2. Pretrained Model Architecture

This is **NOT a new transformer built from scratch**. We load the existing checkpoint from:

```
iupacGPT/iupac-gpt/checkpoints/iupac/
    config.json          ← model hyperparameters
    pytorch_model.bin    ← pretrained weights (~27MB)
    model.safetensors    ← alternate weights format
```

It is a standard **GPT-2 decoder-only transformer** (no encoder, no encoder-decoder cross-attention):

| Hyperparameter        | Value                    |
|-----------------------|--------------------------|
| Model type            | GPT-2 (decoder-only)     |
| Number of layers      | 8 transformer blocks     |
| Attention heads       | 8 per layer              |
| Embedding dimension   | 256                      |
| FFN inner dimension   | 1024 (= 4 × 256)         |
| Context window        | 1280 tokens              |
| Vocabulary size       | 1491 SentencePiece tokens|
| Activation function   | GELU (new variant)       |
| Attention dropout     | 0.1                      |
| Embedding dropout     | 0.1                      |
| Residual dropout      | 0.1                      |
| Layer norm epsilon    | 1e-5                     |
| **Total backbone params** | **7,027,968 (~7M)**  |

### Per-Layer Anatomy (each of 8 layers)

```
Input (L × 256)
    │
    ├─ LayerNorm (ln_1)
    │
    ├─ Causal Self-Attention
    │     c_attn: (256 → 768)  combined Q, K, V projection     263,168 params
    │     c_proj: (256 → 256)  output projection
    │     8 heads × 32 dim/head
    │
    ├─ Residual add
    │
    ├─ LayerNorm (ln_2)
    │
    ├─ Feed-Forward Network (MLP)
    │     c_fc:   (256 → 1024) up-projection                   525,568 params
    │     c_proj: (1024 → 256) down-projection
    │     activation: GELU
    │
    └─ Residual add

LayerNorm params (ln_1 + ln_2): 1,024
Total per layer: 789,760 params
```

GPT-2 uses **Conv1D** layers internally (not `nn.Linear`). The weight matrix is transposed compared to standard Linear: shape is `(d_in, d_out)` instead of `(d_out, d_in)`. This is why `fan_in_fan_out=True` is set in the LoRA config.

---

## 3. The Tokenizer

The tokenizer is a **SentencePiece** model (T5-style subword tokenization) trained specifically on IUPAC names.

- Vocabulary: **1491 subword tokens**
- These tokens represent IUPAC chemical name fragments: `methyl`, `cyclo`, `oxy`, `chloro`, digits, punctuation, brackets, etc.
- Spaces in IUPAC names are replaced with underscores before tokenization (because SentencePiece uses spaces as word boundaries)
- Lives at: `iupacGPT/iupac-gpt/iupac_gpt/iupac_spm.model`
- Serialized version: `iupacGPT/iupac-gpt/iupac_gpt/real_iupac_tokenizer.pt`

**Why SMILES strings would NOT work here:** The tokenizer's 1491-token vocabulary is built for IUPAC nomenclature. A SMILES string like `CC(=O)Oc1ccccc1C(=O)O` would be tokenized into meaningless fragments that the model was never trained on. This is the reason step 3 resolves IUPAC names and drops any compound that cannot be resolved.

---

## 4. Full Pipeline

### Step 1 — Download Raw Data (`steps/step1_download_data.py`)
```
MoleculeNet S3   →  toxcast_raw.csv   (8,597 cpds × 617 assays)
MoleculeNet S3   →  tox21_raw.csv     (7,831 cpds × 12 assays)
DeepChem S3      →  ames_raw.csv      (5,406 cpds, binary mutagenicity)
Local T3DB CSVs  →  t3db_processed.csv(3,512 cpds, LD50-based severity)
```

### Step 2 — Preprocess & Compute Severity (`steps/step2_preprocess.py`)

**Within-dataset dedup:** Remove duplicate SMILES within each dataset.

**Cross-dataset dedup (priority: T3DB > ToxCast > Tox21 > Ames):**
1. Tox21 ∩ ToxCast → remove from Tox21
2. ToxCast ∩ T3DB → remove from ToxCast
3. Ames ∩ (ToxCast ∪ Tox21 ∪ T3DB) → remove from Ames

**Severity computation (no IUPAC names yet):**

| Dataset  | Method                              | Bins                      |
|----------|-------------------------------------|---------------------------|
| ToxCast  | Count assay positives out of 617    | [1, 5, 20, 50]            |
| Tox21    | Count assay positives out of 12     | [1, 2, 3, 4]              |
| Ames     | Binary label 0 or 1                 | 0 → sev 0, 1 → sev 2     |
| T3DB     | LD50 → GHS oral toxicity bins       | < 5 → sev 4, ..., > 5000 → sev 0 |

**Output:** `*_final.csv` files **without** IUPAC names yet.

### Step 3 — Add IUPAC Names (`steps/step3_smiles_to_iupac.py`)

For each compound in `*_final.csv`:
1. Check `iupac_name_cache.csv` (instant, offline)
2. Check extra caches (`tox21_iupac.csv`, `toxcast_iupac.csv`, `ames_iupac.csv`)
3. PubChem PUG REST API
4. ChemSpider v2 API (with API key)
5. NCI CIR API

**Stereoisomer fallback:** If all APIs fail for a molecule with stereo markers (`@`, `/`, `\`):
- Strip stereochemistry → try APIs again with canonical base structure
- If base name found → prepend `alpha-`/`beta-` or `(E)-`/`(Z)-` prefix

**Drop** any compound where no IUPAC name can be resolved.

**Output:** `*_final.csv` files updated in-place with `iupac_name` column added.

### Step 4 — Verify LoRA (`steps/step4_verify_lora.py`)
Loads the model, applies LoRA, prints parameter counts and confirms the setup is correct before training.

### Step 5 — Train (`steps/step5_train.py`)
Fine-tunes the model. Saves only the LoRA + head weights to `outputs/run_<timestamp>/lora_weights.pt`.

### Step 6 — Evaluate (`steps/step6_evaluate.py`)
Loads base model + LoRA weights, evaluates on test split. Reports AUROC, AUPRC, severity accuracy, score RMSE.

### Step 7 — Predict (`steps/step7_predict.py`)
Loads model, takes an IUPAC name or a CSV of names, outputs severity class and score.

---

## 5. LoRA: Frozen Backbone + Trainable Adapters

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique. Instead of training all 7M parameters, we:

1. **Freeze** all original GPT-2 weights
2. **Inject** small trainable low-rank matrices alongside the frozen weights

### How LoRA works mathematically

For a frozen weight matrix $W_0$, the modified output is:

$$h = W_0 x + \frac{\alpha}{r} \cdot B A x$$

Where:
- $W_0$: Original frozen weight (untouched)
- $A \in \mathbb{R}^{d_{in} \times r}$: Down-projection (initialized with Kaiming normal)
- $B \in \mathbb{R}^{r \times d_{out}}$: Up-projection (initialized to **zero** → model starts as original IUPAC-GPT)
- $r = 16$: Rank (how many "directions" of adaptation)
- $\alpha = 32$: Scaling factor → effective scale = $\alpha/r = 2.0$

Because $B$ starts at zero, $BAx = 0$ initially → the model behaves exactly like IUPAC-GPT at the start of training. Over training, $A$ and $B$ gradually learn the toxicity-relevant adaptations.

### Which layers get LoRA

LoRA is applied to 3 module types across all 8 transformer layers:

| Module   | What it does                        | LoRA matrices per layer           |
|----------|-------------------------------------|-----------------------------------|
| `c_attn` | Q/K/V combined projection (256→768) | A: 256×16, B: 16×768 = 12,288    |
| `c_proj` | Attention output (256→256)          | A: 256×16, B: 16×256 = 8,192     |
| `c_fc`   | MLP up-projection (256→1024)        | A: 256×16, B: 16×1024 = 16,384   |

Note: `c_proj` appears in both the attention and MLP sublayers per layer.

**Total LoRA layers: 32** (= 8 layers × 4 LoRA-injected modules per layer)

---

## 6. Parameter Count Breakdown

```
Component                           Params      Status
─────────────────────────────────────────────────────────
Token embeddings (1491 × 256)       381,696     FROZEN
Positional embeddings (1280 × 256)  327,680     FROZEN
8 × Transformer layer weights     6,318,080     FROZEN
─────────────────────────────────────────────────────────
FROZEN TOTAL                      7,027,968

─────────────────────────────────────────────────────────
32 LoRA adapter matrices (A+B)      524,288     TRAINABLE
Severity head (256→128→5)            33,541     TRAINABLE
Score head (256→128→1)               33,025     TRAINABLE
EGNN projection (256→256+LN)         66,304     TRAINABLE
─────────────────────────────────────────────────────────
TRAINABLE TOTAL                     657,158     (8.55% of total)

─────────────────────────────────────────────────────────
GRAND TOTAL                       7,685,126
```

**Only 657K parameters are trained** (~0.6MB checkpoint) versus the full 7M base model (~27MB). This is the key advantage of LoRA.

---

## 7. What Happens Inside Each Transformer Layer

**Input:** Token sequence $X \in \mathbb{R}^{L \times 256}$ where $L$ = number of tokens in the IUPAC name.

**Self-Attention:**

$$[Q, K, V] = X \cdot W_{attn}$$

Each of Q, K, V is 256-dim, then split into 8 heads of 32 dimensions each:

$$\text{Attention}(Q_i, K_i, V_i) = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{32}} + M_{causal}\right) V_i$$

$M_{causal}$ is the causal mask: token at position $t$ can only attend to positions $\leq t$. This was needed during pretraining (GPT-2 is autoregressive) and is preserved in fine-tuning.

All 8 head outputs are concatenated → projected back to 256-dim via `c_proj`.

**Feed-Forward Network (MLP):**

$$h = \text{GELU}(X \cdot W_{fc}) \cdot W_{proj}$$

Expands 256 → 1024 → 256. This is where most of the "computation" per layer happens.

**After all 8 layers:** We take the hidden state at the **last token** position. In a sequence `["2", "-", "methyl", "prop", "ane"]`, we take the representation at the position of `"ane"`. This final hidden state (256-dim vector) summarizes the meaning of the entire IUPAC name as learned by the transformer.

---

## 8. Why the IUPAC Name Encodes Toxicity

IUPAC names are not arbitrary labels — they are a **systematic encoding of molecular structure**. The naming convention directly encodes chemical features:

| IUPAC fragment           | Chemical meaning              | Toxicity relevance                              |
|--------------------------|-------------------------------|--------------------------------------------------|
| `2,4,6-trinitro-`        | Three -NO₂ groups at 2,4,6   | Explosive, DNA damage (mutagenic)               |
| `benz-` / `phenyl`       | Aromatic benzene ring         | Key pharmacophore, receptor binding potential   |
| `chloro-` / `bromo-`     | Halogen substitution          | Increases lipophilicity, metabolic resistance   |
| `amino-`                 | Amine group (-NH₂)            | Affects pharmacokinetics, potential bioactivity |
| `methyl-` / `ethyl-`     | Short alkyl chain             | Generally low toxicity modifier                 |
| `hydroxy-`               | -OH group                     | Increases polarity, often detoxifying           |
| `epoxy-` / `oxo-`        | Reactive oxygen groups        | Electrophilic, potential DNA alkylation         |
| `1,2-dioxo-`             | Two adjacent carbonyls        | Redox activity, potential oxidative stress      |
| `cyclohexa-`             | 6-membered aliphatic ring     | Lipophilic scaffold, CNS penetration potential  |

When IUPAC-GPT was pretrained on PubChem IUPAC names, it learned these co-occurrence patterns — "which name fragments appear together, which substructures they imply." Fine-tuning with LoRA + severity labels teaches it to map these learned representations to toxicity outcomes.

---

## 9. The Three Output Heads

All heads receive the same 256-dim last-token hidden state.

### Head 1: Severity Classification

```
256 → Linear(256, 128) → GELU → Dropout(0.1) → Linear(128, 5)
Output: 5 logits → argmax → class {0, 1, 2, 3, 4}
```

### Head 2: Toxicity Score Regression

```
256 → Linear(256, 128) → GELU → Dropout(0.1) → Linear(128, 1) → Sigmoid
Output: continuous score in [0.0, 1.0]
```

### Head 3: EGNN Projection (Phase 2, future use)

```
256 → Linear(256, 256) → LayerNorm(256)
Output: 256-dim molecular representation vector
```

This is reserved for Phase 2 of the project where this vector will be fed into an **Equivariant Graph Neural Network (EGNN)** that operates on 3D molecular structure. The IUPAC→EGNN vector bridges text-based and geometry-based toxicity reasoning.

---

## 10. Severity Label System

Aligned with **WHO/GHS acute oral toxicity** classification:

| Class | Label            | Score range | ToxCast positives | Tox21 positives | LD50 (T3DB)   |
|-------|------------------|-------------|-------------------|-----------------|---------------|
| 0     | Non-toxic        | 0.00–0.20   | 0                 | 0               | > 5000 mg/kg  |
| 1     | Slightly toxic   | 0.20–0.40   | 1–4               | 1               | 2000–5000     |
| 2     | Moderately toxic | 0.40–0.60   | 5–19              | 2               | 200–2000      |
| 3     | Toxic            | 0.60–0.80   | 20–49             | 3               | 50–200        |
| 4     | Highly toxic     | 0.80–1.00   | ≥ 50              | ≥ 4             | < 50 mg/kg    |

**Ames dataset special handling:** Binary mutagenicity label.
- `ames_label = 0` → severity 0, score 0.10
- `ames_label = 1` → severity 2, score 0.50

Rationale: Ames-positive compounds are flagged as a regulatory genotoxicity hazard (GHS Category 2 genotoxicity), mapped to Moderate severity. The test has ~70% specificity so sev-2 (not sev-3 or sev-4) is a conservative mapping.

**Severity midpoint scores** used as regression targets:
```python
SEVERITY_MIDPOINT = [0.10, 0.30, 0.50, 0.70, 0.90]
```

---

## 11. Training Loss

$$L = 1.0 \times \underbrace{\text{CE}(\text{severity\_logits},\ \text{severity\_labels})}_{\text{classification}} + 0.5 \times \underbrace{\text{MSE}(\text{predicted\_score},\ \text{target\_score})}_{\text{regression}}$$

**Class weighting:** Because severity 0 (non-toxic) dominates the dataset (~28% ToxCast, ~60% Tox21), inverse-frequency weights are computed from the training set and applied to the cross-entropy loss. This prevents the model from predicting "non-toxic" for everything.

**Metrics tracked:**
- `severity_acc`: Correct severity class prediction rate
- `auroc`: Area Under ROC curve (binary: sev ≥ 2 = "toxic")
- `auprc`: Area Under Precision-Recall curve (same binary definition)
- `score_rmse`: Root Mean Square Error of continuous score

---

## 12. At Inference Time

```
Input:  "2-acetoxybenzoic acid"    ← IUPAC name only, nothing else

    ↓ Tokenize (SentencePiece, 1491 vocab)
    [45, 231, 7, 892, 3, ...]

    ↓ Token + Positional Embeddings (256-dim)

    ↓ 8 × Transformer Layer (frozen GPT-2 + trained LoRA deltas)
      Causal self-attention: each token attends to all previous tokens
      FFN: 256 → 1024 → 256

    ↓ Extract last-token hidden state (256-dim vector)

    ↓ Severity head:  → "Class 1: Slightly toxic"
    ↓ Score head:     → 0.27
    ↓ EGNN projection → 256-dim vector (for future Phase 2)

Output: { severity: 1, score: 0.27, label: "Slightly toxic" }
```

The model at inference only needs:
- The pretrained IUPAC-GPT checkpoint
- The trained LoRA weights (`outputs/run_<timestamp>/lora_weights.pt`)
- The IUPAC SentencePiece tokenizer

---

## 13. File Map

```
ToxGaurd/
│
├── ARCHITECTURE.md              ← this file
│
├── iupacGPT/iupac-gpt/
│   ├── checkpoints/iupac/
│   │   ├── config.json          ← GPT-2 hyperparameters (8L, 8H, 256D)
│   │   ├── pytorch_model.bin    ← pretrained weights (READ-ONLY, never modified)
│   │   └── model.safetensors    ← alternate format
│   └── iupac_gpt/
│       ├── iupac_spm.model      ← SentencePiece tokenizer model
│       └── real_iupac_tokenizer.pt ← serialized tokenizer
│
├── toxguard/
│   ├── model.py                 ← ToxGuardModel, ToxGuardLitModel, heads
│   ├── lora.py                  ← LoRALayer, apply_lora_to_model, save/load
│   ├── tokenizer.py             ← ToxGuardTokenizer (T5/SentencePiece wrapper)
│   ├── data_pipeline.py         ← Dataset classes, ToxicityCollator
│   ├── inference.py             ← predict() function for deployment
│   └── cot_explainer.py         ← Chain-of-thought explanation generation
│
├── steps/
│   ├── step1_download_data.py   ← Download ToxCast, Tox21, Ames; process T3DB
│   ├── step2_preprocess.py      ← Dedup, severity computation, save without IUPAC
│   ├── step3_smiles_to_iupac.py ← Add IUPAC via cache + PubChem + ChemSpider + NCI
│   ├── step4_verify_lora.py     ← Verify LoRA setup and parameter counts
│   ├── step5_train.py           ← Training entry point
│   ├── step6_evaluate.py        ← Evaluation on test set
│   └── step7_predict.py         ← Prediction on new compounds
│
├── data/
│   ├── toxcast_raw.csv          ← 8,597 × 618 (smiles + 617 assays)
│   ├── tox21_raw.csv            ← 7,831 × 14 (smiles + 12 assays)
│   ├── ames_raw.csv             ← 5,406 × 2 (smiles, ames_label)
│   ├── t3db_processed.csv       ← 3,512 cpds with LD50-based severity
│   ├── toxcast_final.csv        ← after step 2+3: smiles, iupac_name, sev
│   ├── tox21_final.csv          ← after step 2+3: smiles, iupac_name, sev
│   ├── ames_final.csv           ← after step 2+3: smiles, iupac_name, sev
│   └── iupac_name_cache.csv     ← 19,570+ cached SMILES → IUPAC mappings
│
└── outputs/
    └── run_<timestamp>/
        ├── lora_weights.pt      ← ONLY trained LoRA + head weights (~1-5MB)
        ├── config.json          ← training hyperparameters
        └── results.json         ← evaluation metrics
```

---

## Key Design Decisions Summary

| Decision | Why |
|---|---|
| Use IUPAC names (not SMILES) | IUPAC-GPT was pretrained on IUPAC names; SMILES would produce garbage tokens |
| Decoder-only (GPT-2, no encoder) | IUPAC-GPT is GPT-2 style; using last-token hidden state as sequence representation |
| LoRA r=16, alpha=32 | Trains only 8.55% of parameters; preserves pretrained knowledge; stable at alpha=2r |
| LoRA on c_attn + c_proj + c_fc | Adapts both attention (what to focus on) and FFN (feature transformation) |
| Drop compounds without IUPAC | The tokenizer cannot handle SMILES; unknown IUPAC names would corrupt training |
| Severity 0→sev0, 1→sev2 (Ames) | GHS Category 2 genotoxicity flag; sev-2 is conservative given ~70% test specificity |
| Cross-entropy + MSE dual loss | CE gives sharp class boundaries; MSE provides ordinal grounding (sev 4 ≠ 2× sev 2) |
| Inverse-frequency class weights | Sev-0 dominates dataset; without weighting model predicts "non-toxic" for everything |
| Save only LoRA weights | 657K params vs 7M; original backbone never modified; any run is reproducible from checkpoint |
