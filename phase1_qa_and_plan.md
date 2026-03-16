# ToxGuard Phase 1 — Q&A and Finalization Plan

## Overview

ToxGuard is a multi-modal molecular toxicity prediction system.  
**Phase 1** fine-tunes IUPACGPT (GPT-2 based, trained on IUPAC names) with LoRA adapters for binary toxicity classification.  
**Phase 2** (not yet started) integrates an EGNN (E(n) Equivariant Graph Neural Network) for 3D molecular geometry.

---

## Part 1: Architecture Q&A

### Q1. What is the backbone model?

IUPACGPT — a GPT-2 architecture (8 layers, 8 heads, 256 hidden dim) pre-trained on IUPAC chemical names using a SentencePiece tokenizer (vocab size 1491). It is loaded from a checkpoint and frozen except for LoRA adapter parameters.

### Q2. What is LoRA and why use it here?

Low-Rank Adaptation (LoRA) inserts trainable rank-decomposition matrices into frozen weight matrices. For each target weight `W`, it learns `ΔW = B·A` where `A ∈ R^{r×d}`, `B ∈ R^{d×r}`, and r << d. This drastically reduces trainable parameters while preserving the pre-trained chemistry knowledge.

- **r = 32**, **alpha = 64**, **dropout = 0.2**
- **Target modules**: `c_attn`, `c_proj`, `c_fc` (attention projections + MLP feed-forward)
- **fan_in_fan_out = True** (required for GPT-2 Conv1D weight layout)
- Trainable: ~1.25M params (14.05% of ~8.17M total)

### Q3. What is the classification head?

A binary linear head (`nn.Linear(256, 1)`) attached to the last-token representation of the GPT-2 output. The head outputs a single logit; `sigmoid(logit)` gives P(toxic).

### Q4. What loss function is used and why?

**Focal Loss** with `alpha` and `gamma` parameters:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- Focal loss down-weights easy examples and focuses training on hard, misclassified samples.
- `focal_alpha = 0.45` — calibrated for the ~54% toxic imbalance in the training set (updated from 0.35 which was calibrated for the old 61.4% imbalance).
- `focal_gamma = 2.0` (standard value).

### Q5. What is temperature scaling?

Post-hoc calibration that learns a single scalar T on the validation set:

```
P_calibrated = sigmoid(logit / T)
```

T > 1 softens predictions (makes the model less confident); T < 1 sharpens them. Applied after training, does not change model weights. The best run had T = 1.237.

### Q6. What is the operative decision threshold and how is it chosen?

The threshold is swept from 0.10 to 0.90 on the validation set. Three candidates are computed:

- `best_f1_threshold` — maximises F1
- `best_mcc_threshold` — maximises Matthews Correlation Coefficient (MCC)
- `best_acc_threshold` — maximises accuracy

**Operative threshold = `best_mcc_threshold`** because MCC is the most balanced single metric for binary classification under class imbalance — it accounts for all four cells of the confusion matrix.

---

## Part 2: Dataset Q&A

### Q7. Which datasets are used for training?

| Dataset | Total | Toxic% | Notes |
|---|---|---|---|
| herg_final.csv | 11,304 | 49.7% | hERG cardiotoxicity |
| toxcast_final.csv | 6,887 | 69.9% | Multi-endpoint ToxCast |
| common_molecules_final.csv | 1,048 | 49.4% | Hand-curated common chemicals |
| tox21_final.csv | 240 | 40.4% | Tox21 challenge |
| dili_final.csv | 186 | 47.3% | Drug-induced liver injury |
| **Total** | **~19,665** | **~54%** | |

### Q8. Why are T3DB and ClinTox excluded from training?

They are held out as **external validation sets** to test generalisation on out-of-distribution data:

- **T3DB** (3,512 compounds, 99.2% toxic): A toxin database. Its near-all-toxic composition would catastrophically bias the training distribution (was causing ~61% toxic skew). Headline metric on external validation: **Recall** (sensitivity).
- **ClinTox** (648 compounds, 3.4% toxic): FDA-approved drugs + clinical trial failures. Its extreme non-toxic majority would pull the model toward over-predicting non-toxic. Headline metric: **Specificity** and **Precision**.

### Q9. How is the train/val/test split performed?

**Scaffold stratified split** (Bemis-Murcko scaffold hashing on SMILES). This ensures chemically similar molecules land in the same split, preventing over-optimistic evaluation from scaffold leakage.

- `val_split = 0.1`, `test_split = 0.1`, `seed = 42`
- Requires SMILES column in all training CSVs (common_molecules_final.csv needs regeneration with `--add_smiles`)

### Q10. What is the tautomer canonicalisation fix?

Before converting SMILES → IUPAC name (step 3), the pipeline now applies RDKit's `TautomerEnumerator.Canonicalize()` to collapse tautomers to a canonical form. This eliminates spurious duplicate entries that previously caused 535 `collision_unresolvable` errors in `data/failed_resolve.csv`.

---

## Part 3: Evaluation Q&A

### Q11. What metrics are reported for the main test set?

At both default threshold (0.5) and operative threshold (best val MCC):

- **AUC-ROC** — threshold-independent discrimination ability
- **AUC-PRC** — precision-recall tradeoff (more informative than AUC-ROC under imbalance)
- Accuracy, Precision, Recall, F1, MCC
- Full confusion matrix
- Temperature T

Best run baseline: AUC-ROC = 0.786, AUC-PRC = 0.827, MCC = 0.414

### Q12. What metrics are reported for external validation?

T3DB and ClinTox are evaluated at the same operative threshold:

- All standard binary metrics (same as main test)
- **Specificity** (TN / (TN + FP)) — computed explicitly since sklearn doesn't return it directly
- Headline metrics called out in the report:
  - T3DB: Recall (`<-- headline metric (near-all-toxic corpus)`)
  - ClinTox: Specificity (`<-- headline metric (FDA-approved drug corpus)`)

External validation results are saved in `eval_metrics.json` under `"external_validation"`.

---

## Part 4: Phase 1 Finalization Checklist

### Code changes (all complete)

- [x] `toxguard/data_pipeline.py` — T3DB + ClinTox removed from `prepare_combined_dataset()`; `load_external_validation_datasets()` added
- [x] `toxguard/model.py` — `focal_alpha` default updated: `0.25` → `0.45`
- [x] `steps/step5_train.py` — docstring updated; `--focal_alpha` CLI default updated: `0.35` → `0.45`
- [x] `steps/step3_smiles_to_iupac.py` — tautomer canonicalisation added (`TautomerEnumerator`)
- [x] `steps/build_common_molecules.py` — `--add_smiles` flag added (function + argparse)
- [x] `steps/step2_preprocess.py` — `smiles` column passthrough for `common_molecules_final.csv`
- [x] `steps/step6_evaluate.py` — external validation section added; `best_mcc_threshold` as operative threshold

### Data regeneration (user must run)

```bash
toxguard_env\Scripts\activate
python steps/build_common_molecules.py --add_smiles
python steps/step2_preprocess.py
```

Expected outputs:
- `data/common_molecules_raw.csv` — now includes `smiles` column
- `data/common_molecules_final.csv` — now includes `smiles` column (required for scaffold split)

### Retraining (user must run after data regeneration)

```bash
python steps/step5_train.py
```

Key configuration for final Phase 1 run:
- `--lora_r 32`
- `--focal_alpha 0.45`
- `--split_method scaffold`
- All other defaults are correct

### Evaluation (user must run after retraining)

```bash
python steps/step6_evaluate.py
```

The report will now include both in-distribution test metrics and external validation (T3DB + ClinTox) metrics.

---

## Part 5: Phase 2 Roadmap (EGNN Integration)

### Architecture plan

Phase 2 adds an EGNN branch that processes 3D molecular geometry (atom coordinates + atomic features). The EGNN output (a molecular-level embedding) is fused with the IUPACGPT text embedding before the classification head.

```
IUPAC name string
       |
  IUPACGPT + LoRA
       |
  text_embedding (dim=256)
                                  3D coordinates + atom features
                                          |
                                        EGNN
                                          |
                                  graph_embedding (dim=D)
       |                                 |
       +--------[FusionModule]----------+
                       |
               fused_embedding
                       |
            binary classification head
```

### Steps to implement

| Step | File | Description |
|---|---|---|
| step8 | `toxguard/egnn.py` | EGNN model (equivariant message passing on 3D graphs) |
| step8 | `toxguard/fusion.py` | Fusion module (concat / cross-attention / gated) |
| step8 | `steps/step8_build_3d.py` | Generate 3D conformers from SMILES using RDKit ETKDG |
| step9 | `steps/step9_train_phase2.py` | Phase 2 training (optionally freeze text branch) |
| step10 | `steps/step10_evaluate_phase2.py` | Phase 2 evaluation with ablation (text-only vs EGNN-only vs fused) |

### Data requirements for Phase 2

All training CSVs must have a `smiles` column (already ensured by Phase 1 data regeneration step for `common_molecules_final.csv`; other CSVs already have SMILES from step 3).

3D conformers will be generated on-the-fly or cached to disk via step8.

### Key design decisions (to be resolved in Phase 2)

1. **Fusion method**: simple concatenation → linear projection vs. cross-attention. Cross-attention is more expressive but harder to train.
2. **EGNN depth**: number of message-passing layers (typically 4–6 for drug-sized molecules).
3. **Whether to freeze the text branch** during Phase 2 training to prevent catastrophic forgetting of Phase 1 representations.
4. **Ablation study** is required for the paper: text-only, EGNN-only, and fused model must all be evaluated on the same test/external-validation splits.

---

## Part 6: Paper Outline

### Proposed title

*ToxGuard: Multi-modal Molecular Toxicity Prediction via IUPAC Name Language Models and 3D Equivariant Graph Neural Networks*

### Core contributions

1. First application of IUPAC-name-specific language model (IUPACGPT) with LoRA fine-tuning for toxicity classification.
2. Scaffold-stratified evaluation protocol that correctly separates in-distribution test from out-of-distribution external validation (T3DB, ClinTox).
3. Multi-modal fusion of textual (IUPAC name) and geometric (3D conformer) molecular representations.
4. Ablation demonstrating complementary information from each modality.

### Suggested experiments

- Phase 1 baseline: AUC-ROC, AUC-PRC, MCC on combined test set + T3DB recall + ClinTox specificity
- Phase 2: same metrics for text-only, EGNN-only, and fused model
- Calibration curves (reliability diagrams) comparing T < 1 and T > 1 regimes
- Threshold sensitivity analysis (MCC vs. threshold sweep plots)
