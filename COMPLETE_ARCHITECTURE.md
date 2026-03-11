# Complete Architecture & Working of ToxGuard

---

## Table of Contents

1. [Project Concept](#1-project-concept)
2. [System Overview Diagram](#2-system-overview-diagram)
3. [The Foundation: What is IUPAC-GPT?](#3-the-foundation-what-is-iupac-gpt)
4. [The Backbone: GPT-2 Transformer Architecture](#4-the-backbone-gpt-2-transformer-architecture)
5. [The Tokenizer: Chemical Language Vocabulary](#5-the-tokenizer-chemical-language-vocabulary)
6. [LoRA: How We Fine-Tune Efficiently](#6-lora-how-we-fine-tune-efficiently)
7. [The ToxGuard Model: Full Architecture](#7-the-toxguard-model-full-architecture)
8. [Why IUPAC Names Encode Toxicity](#8-why-iupac-names-encode-toxicity)
9. [Data Pipeline: From Raw Data to Training-Ready](#9-data-pipeline-from-raw-data-to-training-ready)
10. [Training Procedure](#10-training-procedure)
11. [Inference: How a Prediction is Made](#11-inference-how-a-prediction-is-made)
12. [Complete Parameter Accounting](#12-complete-parameter-accounting)
13. [Key Design Decisions](#13-key-design-decisions)

---

## 1. Project Concept

ToxGuard answers one question: **given the name of a chemical compound, how toxic is it?**

The output is a **severity class (0–4)** aligned with WHO/GHS acute oral toxicity standards and a continuous **toxicity score (0.0–1.0)**.

The key insight driving the entire architecture is:

> **IUPAC chemical names are not arbitrary labels. They are a systematic encoding of molecular structure. A model that understands IUPAC names implicitly understands chemistry.**

For example:
- `2,4,6-trinitrotoluene` — the name tells you: benzene ring, methyl group, three nitro groups at positions 2, 4, 6. You know it's TNT before any experiment.
- `sodium chloride` — ionic salt, no functional groups, low toxicity.
- `tetrachlorodibenzo-p-dioxin` — four chlorines, two benzene rings fused with oxygen bridges. This is TCDD/dioxin, highly toxic.

The naming convention encodes functional groups, ring systems, substitution patterns, and molecular topology — all of which determine biological activity and toxicity. We exploit this by using a language model already trained to understand this chemical language.

---

## 2. System Overview Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                        TOXGUARD SYSTEM                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   RAW DATASETS                                                       ║
║   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           ║
║   │ ToxCast  │  │  Tox21   │  │   Ames   │  │  T3DB    │           ║
║   │ 8,597 cpd│  │ 7,831 cpd│  │ 5,406 cpd│  │ 3,512 cpd│           ║
║   │617 assays│  │ 12 assays│  │ binary   │  │ LD50     │           ║
║   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           ║
║        └─────────────┴──────────────┴──────────────┘                ║
║                              │ step2: dedup + severity               ║
║                              ▼                                       ║
║                   *_final.csv (smiles + severity)                    ║
║                              │ step3: SMILES → IUPAC name            ║
║                              ▼                                       ║
║                *_final.csv (smiles + iupac_name + severity)          ║
║                              │                                       ║
╠══════════════════════════════╪═══════════════════════════════════════╣
║   MODEL ARCHITECTURE         │                                       ║
║                              ▼                                       ║
║   IUPAC Name: "2-acetoxybenzoic acid"                                ║
║        │                                                             ║
║        ▼ SentencePiece Tokenizer (1491 vocab)                        ║
║   [45, 231, 7, 892, 3, 12, 8, ...]                                   ║
║        │                                                             ║
║        ▼ Token Embeddings (1491 × 256) + Positional (1280 × 256)     ║
║   [L × 256 tensor]                                                   ║
║        │                                                             ║
║   ┌────▼──────────────────────────────────────────────────────────┐  ║
║   │  FROZEN GPT-2 BACKBONE (7,027,968 params)                     │  ║
║   │                                                               │  ║
║   │  Layer 1: LN → Attn(+LoRA) → Add → LN → MLP(+LoRA) → Add    │  ║
║   │  Layer 2: LN → Attn(+LoRA) → Add → LN → MLP(+LoRA) → Add    │  ║
║   │  Layer 3: LN → Attn(+LoRA) → Add → LN → MLP(+LoRA) → Add    │  ║
║   │  Layer 4: LN → Attn(+LoRA) → Add → LN → MLP(+LoRA) → Add    │  ║
║   │  Layer 5: LN → Attn(+LoRA) → Add → LN → MLP(+LoRA) → Add    │  ║
║   │  Layer 6: LN → Attn(+LoRA) → Add → LN → MLP(+LoRA) → Add    │  ║
║   │  Layer 7: LN → Attn(+LoRA) → Add → LN → MLP(+LoRA) → Add    │  ║
║   │  Layer 8: LN → Attn(+LoRA) → Add → LN → MLP(+LoRA) → Add    │  ║
║   └────────────────────────────────┬──────────────────────────────┘  ║
║                     LoRA injects   │                                  ║
║                     524,288 params │ Extract last-token hidden state  ║
║                                    ▼                                  ║
║                              [256-dim vector]                         ║
║                                    │                                  ║
║          ┌─────────────────────────┼─────────────────────────┐       ║
║          ▼                         ▼                         ▼       ║
║   Severity Head             Score Head              EGNN Projection  ║
║   256→128→5                 256→128→1→σ             256→256+LN       ║
║   33,541 params             33,025 params           66,304 params    ║
║          │                         │                         │       ║
║          ▼                         ▼                         ▼       ║
║   severity class 0-4        score 0.0–1.0          256-dim repr      ║
║   "Moderately toxic"        0.50                   (Phase 2/EGNN)    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. The Foundation: What is IUPAC-GPT?

IUPAC-GPT is a GPT-2 language model built and pretrained by an external research group specifically for chemical compound names. It was published for drug discovery tasks: molecular property prediction and de novo molecule generation.

### What it was trained to do

It was trained **autoregressively** — given the beginning of an IUPAC name, predict the next token. For a name like:

```
"2-acetoxybenzoic acid"
```

Tokenized as:
```
["2", "-", "acetoxy", "benz", "oic", "_acid"]
```

The training objective was:
- Given `["2"]` → predict `"-"`
- Given `["2", "-"]` → predict `"acetoxy"`
- Given `["2", "-", "acetoxy"]` → predict `"benz"`
- ...and so on.

To do this correctly across millions of IUPAC names, the model was forced to learn:
- Which functional groups co-occur with which ring structures
- How substituent positions (numbers) relate to the base structure name
- Which name patterns correspond to which classes of molecules
- The grammar and rules of IUPAC nomenclature itself

This results in a model that has **implicit chemical knowledge encoded in its weights**, even though it never saw a molecular structure or SMILES string directly during training.

### Where we get it from

The checkpoint lives at:
```
iupacGPT/iupac-gpt/checkpoints/iupac/
    config.json           ← architecture hyperparameters
    pytorch_model.bin     ← pretrained weights (~27 MB, READ-ONLY)
    model.safetensors     ← alternate format
```

**This checkpoint is NEVER modified.** All training writes only to the `outputs/` directory (LoRA weights + heads). The original checkpoint is a read-only input.

### How we load it

In `toxguard/model.py`, the `from_pretrained_iupacgpt()` class method:
1. Initializes a `GPT2Model` from the config
2. Loads `pytorch_model.bin` (read-only)
3. Copies only the transformer weights — not the LM head (we don't need token prediction)
4. Applies LoRA adapters on top of the frozen transformer

---

## 4. The Backbone: GPT-2 Transformer Architecture

### Architecture configuration

| Parameter             | Value                         | What it means                                         |
|-----------------------|-------------------------------|-------------------------------------------------------|
| `model_type`          | `gpt2`                        | Standard GPT-2 decoder-only architecture              |
| `n_layer`             | `8`                           | 8 stacked transformer blocks                          |
| `n_head`              | `8`                           | Attention split into 8 parallel heads                 |
| `n_embd`              | `256`                         | Every token is represented as a 256-dim vector        |
| `n_inner`             | `1024`                        | MLP expands to 1024 before compressing back to 256    |
| `n_ctx`               | `1280`                        | Maximum 1280 tokens in a single sequence              |
| `vocab_size`          | `1491`                        | 1491 IUPAC subword tokens in the vocabulary           |
| `activation_function` | `gelu_new`                    | GELU variant used in GPT-2                            |
| `attn_pdrop`          | `0.1`                         | 10% dropout on attention weights                      |
| `embd_pdrop`          | `0.1`                         | 10% dropout on input embeddings                       |
| `resid_pdrop`         | `0.1`                         | 10% dropout on residual connections                   |
| `layer_norm_epsilon`  | `1e-5`                        | Numerical stability for layer normalization           |

### How a single transformer block works

Each of the 8 layers is identical in structure:

```
Input hidden states H  [shape: L × 256, where L = sequence length]
    │
    ├─ LayerNorm (ln_1)    normalize → stable gradients
    │       │
    │       ▼
    │   Causal Self-Attention
    │       │
    │       ├── c_attn  [256 → 768]  ← projects H into Q, K, V combined
    │       │   (first 256 cols = Q, next 256 = K, last 256 = V)
    │       │
    │       ├── Split into 8 heads, each head has dim 32
    │       │
    │       ├── For each head i:
    │       │       score_i = (Q_i · K_i^T) / √32
    │       │       mask: set score[t, t'] = -∞ if t' > t   (causal!)
    │       │       attn_i = softmax(score_i)
    │       │       out_i  = attn_i · V_i
    │       │
    │       ├── Concatenate all 8 heads → [L × 256]
    │       │
    │       └── c_proj  [256 → 256]  ← output projection
    │
    ├─ Residual connection: H = H + attention_output
    │
    ├─ LayerNorm (ln_2)    normalize again
    │       │
    │       ▼
    │   Feed-Forward Network (MLP)
    │       │
    │       ├── c_fc    [256 → 1024]   ← up-project (expand)
    │       ├── GELU activation
    │       └── c_proj  [1024 → 256]  ← down-project (compress)
    │
    └─ Residual connection: H = H + mlp_output

Output hidden states H  [shape: L × 256]
```

### The causal mask

The causal mask ensures that token at position `t` can only attend to tokens at positions `0, 1, ..., t`. It cannot look at future tokens. This was needed for autoregressive generation during pretraining.

For toxicity classification, this means the final token has seen all other tokens before it — it carries a representation of the entire IUPAC name. This is why we extract the **last-token hidden state** as the compound's representation.

### GPT-2's Conv1D convention

GPT-2 uses `Conv1D` layers instead of `nn.Linear`. The weight matrix shape is `(d_in, d_out)` — transposed from standard Linear `(d_out, d_in)`. When we implement LoRA, this is handled by setting `fan_in_fan_out=True`, which transposes the LoRA matrices accordingly.

### Parameter count per layer

```
c_attn:                 256 × 768 + bias 768        =    197,376
c_proj (attention):     256 × 256 + bias 256         =     65,792
c_fc:                   256 × 1024 + bias 1024        =    263,168
c_proj (MLP):          1024 × 256 + bias 256         =    262,400
ln_1 (weight + bias):  256 + 256                     =        512
ln_2 (weight + bias):  256 + 256                     =        512
                                               Total = 789,760 params/layer
× 8 layers                                         = 6,318,080

Token embeddings:   1491 × 256                       =    381,696
Position embeddings: 1280 × 256                      =    327,680
Final LayerNorm:    256 + 256                        =        512

BACKBONE TOTAL:                                      = 7,027,968
```

---

## 5. The Tokenizer: Chemical Language Vocabulary

### Type and vocabulary

The tokenizer is a **SentencePiece** model (same type as T5), trained from scratch on a large corpus of IUPAC names from PubChem. It has exactly **1491 subword tokens**.

Unlike general-purpose BPE tokenizers (GPT-2's original tokenizer has 50,257 tokens for English), this tokenizer's entire vocabulary is devoted to IUPAC chemical nomenclature fragments:

```
Examples of tokens in the vocabulary:
    methyl      cyclo       oxy         chloro
    benz        amino       hydroxy     fluoro
    nitro       acetyl      phenyl      ethyl
    propyl      carboxyl    sulfonyl    phospho
    1           2           3           ,
    -           (           )           [
    alpha       beta        hex         pent
    oct         ...and 1491 total tokens
```

### Preprocessing step (critical)

Before tokenization, spaces in IUPAC names are replaced with underscores:

```
"2-acetoxybenzoic acid"
         ↓
"2-acetoxybenzoic_acid"
         ↓ SentencePiece tokenizer
[45, 231, 7, 892, 3, 12, 8]
```

Reason: SentencePiece treats a space as a word boundary signal. Since IUPAC names contain spaces that are part of the name (e.g., "benzoic acid" is one compound, not two words), replacing spaces with underscores preserves this correctly.

### Implementation

```
iupacGPT/iupac-gpt/iupac_gpt/iupac_spm.model      ← the tokenizer file
iupacGPT/iupac-gpt/iupac_gpt/real_iupac_tokenizer.pt  ← serialized version
```

`ToxGuardTokenizer` in `toxguard/tokenizer.py` subclasses `T5Tokenizer` and overrides `_tokenize()` to:
1. Replace spaces with underscores
2. Tokenize using the SentencePiece model
3. Remove the first token if it is the SentencePiece non-printing leading character

### Why SMILES cannot be used

The vocabulary was built for IUPAC names, not SMILES notation. A SMILES string fed in would be tokenized into random fragments:

```
SMILES: "CC(=O)Oc1ccccc1C(=O)O"   ← aspirin
Tokenized with IUPAC vocab:
    "C"  → some token for "cyclo" prefix?
    "C(" → unknown fragment
    "=O" → splits unpredictably
```

The resulting token IDs would correspond to IUPAC name fragments at those positions, producing complete nonsense. The model has no framework to interpret SMILES character sequences. This is the fundamental reason step 3 resolves IUPAC names and any compound that cannot be resolved is dropped from training.

---

## 6. LoRA: How We Fine-Tune Efficiently

### The problem with full fine-tuning

The IUPAC-GPT backbone has 7,027,968 parameters. Training all of them on toxicity data has two problems:

1. **Catastrophic forgetting**: The model overwrites its pretrained chemical knowledge with toxicity-specific patterns. It loses the structural understanding that makes IUPAC representations valuable.
2. **Computational cost**: 7M parameters × gradients × optimizer states = large memory, slow training.

### LoRA solution

LoRA (Low-Rank Adaptation, Hu et al. 2021) freezes all original weights and injects small trainable low-rank update matrices alongside them.

### Mathematical formulation

For any frozen weight matrix $W_0 \in \mathbb{R}^{d_{in} \times d_{out}}$:

$$\text{Standard:    } h = W_0 x$$

$$\text{With LoRA:   } h = W_0 x + \frac{\alpha}{r} \cdot B A x$$

Where:
- $W_0$: Original frozen weight — gradient is never computed, never updated
- $A \in \mathbb{R}^{d_{in} \times r}$: Down-projection, initialized with Kaiming normal distribution
- $B \in \mathbb{R}^{r \times d_{out}}$: Up-projection, **initialized to all zeros**
- $r = 16$: The rank — controls how many "directions of adaptation" are learned
- $\alpha = 32$: Scaling factor — effective scale $= \alpha / r = 32/16 = 2.0$

### Why B is initialized to zero

At the start of training:
$$B A x = 0 \cdot A x = 0$$

So the model starts as exactly the original IUPAC-GPT. The adaptation starts from zero and grows as the model learns from toxicity labels. This guarantees stable training from a known starting point.

### Which modules get LoRA applied

LoRA is injected into 3 module types across all 8 transformer layers:

```
Layer k (k = 1..8):
    c_attn  [256 → 768]   ← combined Q, K, V projection
        LoRA A: [256 × 16]
        LoRA B: [16 × 768]
        Extra params: 256×16 + 16×768 = 4,096 + 12,288 = 16,384

    c_proj (attention)  [256 → 256]   ← attention output projection
        LoRA A: [256 × 16]
        LoRA B: [16 × 256]
        Extra params: 256×16 + 16×256 = 4,096 + 4,096 = 8,192

    c_fc  [256 → 1024]   ← MLP up-projection
        LoRA A: [256 × 16]
        LoRA B: [16 × 1024]
        Extra params: 256×16 + 16×1024 = 4,096 + 16,384 = 20,480

    c_proj (MLP)  [1024 → 256]   ← MLP down-projection
        LoRA A: [1024 × 16]
        LoRA B: [16 × 256]
        Extra params: 1024×16 + 16×256 = 16,384 + 4,096 = 20,480
```

Per layer LoRA params: 16,384 + 8,192 + 20,480 + 20,480 = 65,536
× 8 layers = **524,288 total LoRA params**

### What LoRA learns

- `c_attn` LoRA: Adjusts which parts of the IUPAC name the model focuses on for toxicity assessment (e.g., "pay more attention to functional group prefixes like nitro-, chloro-")
- `c_proj` LoRA: Adjusts how attention head outputs are combined
- `c_fc` LoRA: Adjusts the feature transformations in the MLP bottleneck
- `c_proj (MLP)` LoRA: Adjusts the final compression back to 256-dim

### Fan-in fan-out convention

GPT-2's Conv1D has transposed weight shapes vs standard `nn.Linear`. The LoRA implementation in `toxguard/lora.py` sets `fan_in_fan_out=True` which transposes the LoRA matrix multiplication to match:

```python
# Standard Linear: h = x @ W.T   (W is [d_out, d_in])
# GPT-2 Conv1D:    h = x @ W     (W is [d_in, d_out])
# LoRA for Conv1D: h += x @ A @ B  (with fan_in_fan_out=True)
```

### Saving and loading LoRA weights

`save_lora_weights()` in `lora.py` only saves parameters where `requires_grad=True` — this is the LoRA matrices plus the head parameters. Result: a ~1–5 MB checkpoint instead of the full 27 MB model.

At inference, `load_lora_weights()` loads this small checkpoint back into the model. The frozen backbone is loaded separately from the original (unchanged) checkpoint.

---

## 7. The ToxGuard Model: Full Architecture

### Class hierarchy

```
ToxGuardModel (nn.Module)
    │
    ├── backbone: GPT2Model (frozen, LoRA-injected)
    │       └── 8 × GPT2Block
    │               └── GPT2Attention + GPT2MLP (with LoRA layers)
    │
    └── heads: ToxGuardMultiTaskHead (trainable)
            ├── severity_head: ToxicityHead (256→128→5)
            ├── score_head: ToxicityHead (256→128→1→sigmoid)
            └── egnn_projection: Linear(256,256) + LayerNorm(256)

ToxGuardLitModel (pl.LightningModule)
    └── model: ToxGuardModel
    (Handles training loop, optimizer, learning rate schedule, metrics)
```

### Forward pass (step by step)

**Input:** batch of IUPAC name token sequences with shape `[B × L]` (B = batch size, L = max token length)

```
Step 1: Input preparation
    input_ids:      [B × L]  integer token IDs
    attention_mask: [B × L]  1 for real tokens, 0 for padding

Step 2: GPT2Model forward pass
    → Token embeddings:    input_ids   →  [B × L × 256]
    → Position embeddings: [0..L-1]   →  [B × L × 256]
    → Sum                              →  [B × L × 256]
    → Apply dropout (embd_pdrop=0.1)
    → Pass through 8 transformer layers
    → Output: all_hidden_states        →  [B × L × 256]

Step 3: Extract last-token hidden state
    For each example b in batch:
        last_real_pos = last position where attention_mask[b, :] == 1
        h_b = all_hidden_states[b, last_real_pos, :]   → [256]
    Stacked: pooled = [B × 256]

Step 4: Branch into three heads
    severity_logits = severity_head(pooled)    → [B × 5]
    toxicity_score  = score_head(pooled)       → [B × 1]  (0.0–1.0 after sigmoid)
    egnn_repr       = egnn_projection(pooled)  → [B × 256]

Step 5: Return
    {
        severity_logits: [B × 5],
        toxicity_score:  [B × 1],
        egnn_repr:       [B × 256]
    }
```

### The severity head in detail

```
ToxicityHead(input_dim=256, hidden_dim=128, output_dim=5)

    Linear(256 → 128)
        ↓  weight: [128 × 256] + bias [128]
    GELU activation
    Dropout(p=0.1)
    Linear(128 → 5)
        ↓  weight: [5 × 128] + bias [5]

    Output: raw logits [5]
    At prediction time: argmax → integer class 0, 1, 2, 3, or 4

Params: 128×256 + 128 + 5×128 + 5 = 32,768 + 128 + 640 + 5 = 33,541
```

### The score head in detail

```
ToxicityHead(input_dim=256, hidden_dim=128, output_dim=1, use_sigmoid=True)

    Linear(256 → 128)
    GELU
    Dropout(0.1)
    Linear(128 → 1)
    Sigmoid
        ↓
    Output: scalar in [0.0, 1.0]

Params: 128×256 + 128 + 1×128 + 1 = 32,768 + 128 + 128 + 1 = 33,025
```

### The EGNN projection in detail

```
Linear(256 → 256)
LayerNorm(256)
    ↓
Output: 256-dim normalized vector

Params: 256×256 + 256 + 256 + 256 = 65,536 + 256 + 512 = 66,304
```

This is a Phase 2 component. The 256-dim output of this projection is designed to be passed to an Equivariant Graph Neural Network (EGNN) that processes 3D molecular structure. The projection aligns the text-derived representation with the 3D geometry-derived representation space, enabling multi-modal toxicity reasoning.

---

## 8. Why IUPAC Names Encode Toxicity

This section explains the chemical reasoning behind the entire architecture choice.

### IUPAC names as structural description

IUPAC names follow a strict hierarchical system:
1. Parent chain / parent ring name (describes carbon skeleton)
2. Substituent prefixes (describes functional groups and their positions)
3. Suffix (describes the principal functional group)

This means every IUPAC name is a compressed description of molecular structure. The model learns to decode this compression.

### Functional group → toxicity mapping

| IUPAC Fragment       | Chemical Meaning                  | Why It Matters for Toxicity                                    |
|----------------------|-----------------------------------|----------------------------------------------------------------|
| `nitro-`             | –NO₂ group                        | Electrophilic; metabolized to reactive nitroso intermediates, DNA damage |
| `chloro-`/`bromo-`   | Halogen substitution              | Increases metabolic stability; halogenated compounds bioaccumulate |
| `amino-`             | –NH₂                              | Can form reactive intermediates; many carcinogens contain amino groups |
| `epoxy-`             | 3-membered ring with oxygen       | Highly reactive electrophile; alkylates DNA directly           |
| `benz-`/`phenyl`     | Aromatic benzene ring             | Aromatic systems bind hydrophobic pockets; PAHs are carcinogenic |
| `hydroxy-`           | –OH                               | Generally detoxifying; adds polarity for excretion             |
| `cyano-`             | –CN (nitrile)                     | Releases cyanide on hydrolysis                                 |
| `phospho-`           | Phosphorus group                  | Organophosphates inhibit acetylcholinesterase (nerve agents)   |
| `sulfonyl-`          | –SO₂–                             | Found in sulfonamide drugs, generally moderate activity        |
| `methyl-`/`ethyl-`   | Short alkyl chains                | Usually low toxicity modifiers                                 |
| `perfluoro-`         | All H replaced by F               | Extreme metabolic stability, PFAS compounds persist in environment |
| `1,2-dioxo-`         | Two adjacent carbonyls            | Redox cycling, oxidative stress                                |
| `cyclopenta-`/`-hexyl` | Ring systems                   | Lipophilicity, CNS penetration potential                       |

### How pretraining captures this

During IUPAC-GPT pretraining on millions of PubChem names, the model saw:
- `"2,4,6-trinitrophenol"` (picric acid, explosive, highly toxic)
- `"2,4,6-trinitrotoluene"` (TNT)
- `"2,4-dinitrotoluene"`
- `"nitrobenzene"`
- `"aniline"` (aminobenzene)

It learned co-occurrence: `nitro-` + benzene ring patterns are a cluster in representation space. Fine-tuning with severity labels teaches: this cluster → high severity.

Similarly:
- `sodium-, -ate, -ide` patterns cluster as ionic salts → generally low toxicity
- `organo-phosph-` patterns cluster as potential AChE inhibitors → high toxicity when confirmed

---

## 9. Data Pipeline: From Raw Data to Training-Ready

### The four source datasets

| Dataset  | Source                    | Size              | Label type                              |
|----------|---------------------------|-------------------|-----------------------------------------|
| ToxCast  | US EPA / MoleculeNet      | 8,597 compounds   | 617 in-vitro bioactivity assays (0/1)   |
| Tox21    | NIH / MoleculeNet         | 7,831 compounds   | 12 regulatory assays (0/1)             |
| Ames     | DeepChem S3               | 5,406 compounds   | Binary mutagenicity (Ames test)         |
| T3DB     | Toxin/Target Database     | 3,512 compounds   | LD50 values (mg/kg, oral, rat)          |

### Step 1: Download / Prepare (`step1_download_data.py`)

```
ToxCast  → MoleculeNet S3 URL → download + validate → data/toxcast_raw.csv
Tox21    → MoleculeNet S3 URL → download + validate → data/tox21_raw.csv
Ames     → DeepChem S3 URL    → download + validate → data/ames_raw.csv
T3DB     → local CSVs in data/t3db/ → process_local_t3db() → data/t3db_processed.csv
```

### Step 2: Deduplicate + Compute Severity (`step2_preprocess.py`)

**Within-dataset deduplication:**
Each dataset: keep only one compound per unique canonical SMILES.

**Cross-dataset deduplication (by priority):**

Priority order: T3DB > ToxCast > Tox21 > Ames

Rules:
1. Any compound in both Tox21 and ToxCast → keep in ToxCast, remove from Tox21
2. Any compound in both ToxCast and T3DB → keep in T3DB, remove from ToxCast
3. Any compound in Ames AND any of {ToxCast, Tox21, T3DB} → remove from Ames

Rationale: T3DB has direct LD50 measurements (quantitative, most informative). ToxCast has 617 assays (broadest coverage). Tox21 has regulatory endpoints. Ames is narrow (only mutagenicity).

**Severity computation:**

```
ToxCast:
    n_positive = number of assays where compound tested positive (out of 617)
    severity = pd.cut(n_positive, bins=[−1, 0, 4, 19, 49, 617], labels=[0,1,2,3,4])
    
    sev 0: n_positive = 0   (no assay positives)
    sev 1: n_positive = 1–4
    sev 2: n_positive = 5–19
    sev 3: n_positive = 20–49
    sev 4: n_positive ≥ 50

Tox21:
    n_positive = number of assays positive (out of 12)
    severity = pd.cut(n_positive, bins=[−1, 0, 1, 2, 3, 12], labels=[0,1,2,3,4])
    
    sev 0: n_positive = 0
    sev 1: n_positive = 1
    sev 2: n_positive = 2
    sev 3: n_positive = 3
    sev 4: n_positive ≥ 4

T3DB:
    GHS Acute Oral Toxicity classification from LD50 (mg/kg):
    sev 4: LD50 < 5         (GHS Category 1, Fatal)
    sev 3: LD50 5–50        (GHS Category 2, Fatal)
    sev 2: LD50 50–300      (GHS Category 3, Toxic)  
    sev 1: LD50 300–2000    (GHS Category 4, Harmful)
    sev 0: LD50 > 2000      (GHS Category 5, Minimal concern)

Ames:
    ames_label = 0 → severity 0, score 0.10  (non-mutagenic, no cause for concern)
    ames_label = 1 → severity 2, score 0.50  (mutagenic = GHS Category 2 genotoxicity)
```

**Output:** `data/toxcast_final.csv`, `data/tox21_final.csv`, `data/ames_final.csv`, `data/t3db_processed.csv` — all containing `smiles` + `severity` but NO `iupac_name` yet.

### Step 3: Resolve IUPAC Names (`step3_smiles_to_iupac.py`)

For each SMILES in each `*_final.csv`, resolve to an IUPAC name using a cascade:

```
Attempt 1: iupac_name_cache.csv          (19,570+ entries, instant)
     ↓ not found
Attempt 2: tox21_iupac.csv / toxcast_iupac.csv / ames_iupac.csv
           (pre-existing IUPAC caches from prior processing)
     ↓ not found
Attempt 3: PubChem PUG REST API
           GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON
     ↓ not found
Attempt 4: ChemSpider v2 API
           POST https://api.rsc.org/compounds/v1/filter/smiles
     ↓ not found
Attempt 5: NCI CIR (Chemical Identifier Resolver)
           GET https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name
     ↓ all failed
     
Stereoisomer fallback (if SMILES contained @, /, \):
     Strip stereo markers from SMILES → canonical base structure
     Try all 5 sources again for base structure
     If found: prepend alpha-/beta- or (E)-/(Z)- to the name
```

If all attempts fail → compound is **dropped** from the dataset.

**Why dropping is correct** (not using SMILES as fallback): The tokenizer's 1491-token vocabulary is IUPAC-only. Any SMILES string fed in would produce meaningless token sequences that corrupt model training.

---

## 10. Training Procedure

### Dataset construction

Each dataset class (`ToxCastDataset`, `Tox21Dataset`, `AmesDataset`) defined in `toxguard/data_pipeline.py`:
1. Reads the corresponding `*_final.csv` (must have `iupac_name` column)
2. Raises `ValueError` if `iupac_name` is missing (run step 3 first)
3. Tokenizes each IUPAC name → `input_ids` + `attention_mask`
4. Looks up `severity` → integer label 0–4
5. Looks up severity midpoint → `toxicity_score` ∈ {0.10, 0.30, 0.50, 0.70, 0.90}

Severity midpoints:
```python
SEVERITY_MIDPOINT = {0: 0.10, 1: 0.30, 2: 0.50, 3: 0.70, 4: 0.90}
```

### Loss function

Two simultaneous objectives:

$$L = \underbrace{1.0 \times \text{CE}(\hat{y}_{sev},\ y_{sev},\ w)}_{\text{classification loss}} + \underbrace{0.5 \times \text{MSE}(\hat{s},\ s)}_{\text{score regression loss}}$$

**Cross-entropy (CE)** with class weights $w$:

$$\text{CE} = -\sum_{c=0}^{4} w_c \cdot y_c \cdot \log(\hat{p}_c)$$

Class weights are computed from training set severity distribution using inverse frequency:

$$w_c = \frac{N_{\text{total}}}{5 \times N_c}$$

This prevents the model from predicting "non-toxic" (class 0) for everything, since sev-0 compounds are the most common class in ToxCast (~28%) and Tox21 (~60%).

**MSE** on continuous score:

$$\text{MSE} = (\hat{s} - s)^2$$

where $s$ is the severity midpoint target (0.10–0.90). This adds ordinal structure — the model learns not just class boundaries but that sev-4 is much more toxic than sev-0.

### Optimizer and schedule

- **Optimizer**: AdamW, learning rate = 2e-4, weight decay = 0.01
- **Schedule**: Cosine annealing with linear warmup
  - Warmup for first 5% of steps: LR ramps from 0 → max_lr
  - Then cosine decay from max_lr → near 0

### What gets updated vs. not

```
Updated (requires_grad=True):
    LoRA matrices A, B in 32 LoRA layers         ← backbone adaptation
    severity_head weights + biases               ← classification
    score_head weights + biases                  ← regression
    egnn_projection weights                      ← Phase 2 prep

NOT updated (requires_grad=False):
    All original GPT-2 weights (embeddings +
    LayerNorms + Q/K/V + FFN + output LN)        ← frozen backbone
```

### Saved output

At the end of training, `outputs/run_<timestamp>/` contains:
- `lora_weights.pt` — only LoRA + head params (~1–5 MB)
- `config.json` — training hyperparameters
- `results.json` — final evaluation metrics

The original checkpoint in `iupacGPT/` is untouched. Any run can be fully reproduced: load the original checkpoint + apply `lora_weights.pt`.

---

## 11. Inference: How a Prediction is Made

### Input to output, fully traced

```
───────────────────────────────────────────────────────────
INPUT: "tetrachlorodibenzo-p-dioxin"
───────────────────────────────────────────────────────────

STEP 1 — TOKENIZATION
    Replace spaces with underscores (none here)
    SentencePiece encode with vocab of 1491 tokens:
    → [32, 78, 14, 229, 5, 44, 189, 3, 22]  (example IDs)
    
    Tensor shapes:
    input_ids:      [1 × 9]
    attention_mask: [1 × 9]  (all 1s, no padding)

STEP 2 — EMBEDDING
    Token embedding lookup:   [1 × 9 × 256]
    Position embedding lookup: [1 × 9 × 256]
    Sum → dropout → [1 × 9 × 256]

STEP 3 — 8 TRANSFORMER LAYERS (frozen + LoRA delta)
    Each layer:
        LayerNorm → Self-Attention (causal) → Add residual
        LayerNorm → MLP → Add residual
    After all 8: [1 × 9 × 256]

STEP 4 — POOL (last token)
    Last real token at position 8 (no padding)
    Pooled: [1 × 256]

STEP 5 — SEVERITY HEAD
    Linear(256→128) → GELU → Dropout → Linear(128→5)
    Logits: [-2.1, -0.8, 0.3, 1.2, 3.9]
    Softmax: [0.01, 0.04, 0.12, 0.25, 0.58]
    argmax → class 4

STEP 6 — SCORE HEAD
    Linear(256→128) → GELU → Dropout → Linear(128→1) → Sigmoid
    Raw: 2.8 → Sigmoid → 0.94

STEP 7 — OUTPUT
    {
        severity_class:  4,
        severity_label:  "Highly toxic",
        toxicity_score:  0.94,
        confidence:      0.58   (softmax probability of predicted class),
        egnn_repr:       [256-dim vector]  (for Phase 2)
    }

───────────────────────────────────────────────────────────
(TCDD/Dioxin is indeed one of the most toxic known substances)
───────────────────────────────────────────────────────────
```

### Files needed at inference

| File | Purpose |
|------|---------|
| `iupacGPT/iupac-gpt/checkpoints/iupac/pytorch_model.bin` | Frozen backbone weights |
| `iupacGPT/iupac-gpt/checkpoints/iupac/config.json` | Architecture config |
| `iupacGPT/iupac-gpt/iupac_gpt/real_iupac_tokenizer.pt` | Tokenizer |
| `outputs/run_<timestamp>/lora_weights.pt` | Trained LoRA + heads |

### Inference script

```
python steps/step7_predict.py --iupac "tetrachlorodibenzo-p-dioxin"
python steps/step7_predict.py --input compounds.csv --output predictions.csv
```

---

## 12. Complete Parameter Accounting

```
╔═══════════════════════════════════════════════════════════════╗
║  COMPONENT                      PARAMS      STATUS            ║
╠═══════════════════════════════════════════════════════════════╣
║  Token Embeddings (1491 × 256)   381,696    FROZEN            ║
║  Position Embeddings (1280×256)  327,680    FROZEN            ║
║                                                               ║
║  Transformer Layers (×8):                                     ║
║    c_attn (256→768)+bias         197,376  × 8                 ║
║    c_proj attn (256→256)+bias     65,792  × 8                 ║
║    c_fc (256→1024)+bias          263,168  × 8      FROZEN     ║
║    c_proj MLP (1024→256)+bias    262,400  × 8                 ║
║    ln_1 (weight+bias)                512  × 8                 ║
║    ln_2 (weight+bias)                512  × 8                 ║
║    Subtotal per layer:           789,760                      ║
║    Total 8 layers:             6,318,080    FROZEN            ║
║  Final LayerNorm:                    512    FROZEN            ║
║─────────────────────────────────────────────────────────────  ║
║  FROZEN TOTAL                  7,027,968                      ║
╠═══════════════════════════════════════════════════════════════╣
║  LoRA c_attn  (per layer: 256×16 + 16×768)                    ║
║      A: 4,096 + B: 12,288 = 16,384 × 8      131,072           ║
║  LoRA c_proj attn (per layer: 256×16 + 16×256)                ║
║      A: 4,096 + B: 4,096  =  8,192 × 8       65,536           ║
║  LoRA c_fc    (per layer: 256×16 + 16×1024)                   ║
║      A: 4,096 + B: 16,384 = 20,480 × 8      163,840  TRAINABLE║
║  LoRA c_proj MLP (per layer: 1024×16 + 16×256)               ║
║      A: 16,384 + B: 4,096 = 20,480 × 8      163,840           ║
║  Total LoRA                    524,288    TRAINABLE           ║
║                                                               ║
║  Severity Head (256→128→5)       33,541    TRAINABLE          ║
║  Score Head (256→128→1)          33,025    TRAINABLE          ║
║  EGNN Projection (256→256+LN)    66,304    TRAINABLE          ║
║─────────────────────────────────────────────────────────────  ║
║  TRAINABLE TOTAL                657,158    (8.55%)            ║
╠═══════════════════════════════════════════════════════════════╣
║  GRAND TOTAL                  7,685,126                       ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 13. Key Design Decisions

| Decision | What it means | Why it was made |
|---|---|---|
| **IUPAC names as input** | Model takes chemical names, not SMILES or molecular graphs | IUPAC-GPT was pretrained on IUPAC; the tokenizer's 1491-token vocab is IUPAC-only |
| **GPT-2 decoder-only** | No encoder block; we use last-token pooling for sequence representation | IUPAC-GPT is GPT-2 architecture; causal attention means the last token seen all previous tokens |
| **LoRA r=16, α=32** | 32 rank-16 adapter matrices instead of full weight updates | Trains 8.55% of params; preserves pretrained chemical knowledge; stable with α=2r |
| **LoRA on c_attn + c_proj + c_fc** | Adapts attention + MLP in each layer | Covers what-to-attend (attention) and how-to-transform (MLP); omitting FFN output slightly reduces params |
| **Frozen during fine-tuning** | All 7M original GPT-2 params never change | Prevents catastrophic forgetting of IUPAC structural knowledge; smaller, reproducible checkpoints |
| **Drop compounds without IUPAC** | If step 3 can't find an IUPAC name, the compound is excluded | SMILES fed to this tokenizer would produce meaningless tokens; no valid training signal possible |
| **Cross-entropy + MSE dual loss** | Two objectives simultaneously | CE: sharp class boundaries; MSE: ordinal grounding (sev-4 is not 2× sev-2, it's much worse) |
| **Inverse-frequency class weights** | Rare severe classes get higher loss weight | Overcomes class imbalance; without this, model predicts "non-toxic" for everything |
| **T3DB > ToxCast > Tox21 > Ames** | Cross-dedup priority order | T3DB has quantitative LD50; ToxCast broadest assay count; Ames narrowest single endpoint |
| **Ames: sev-0 or sev-2 only** | No sev-1 from Ames data | Binary Ames test; label=1 → GHS Category 2 genotoxicity → moderate (sev-2); not sev-4 because Ames ~70% specificity |
| **Save LoRA weights only** | `lora_weights.pt` is ~1–5 MB vs 27 MB full model | Full model is always recoverable from original checkpoint + LoRA; checkpoint is small; any run reproducible |
| **EGNN projection head** | 256→256+LN projection output head exists but isn't used in Phase 1 | Architectural placeholder for Phase 2 multi-modal toxicity reasoning using 3D molecular geometry |
