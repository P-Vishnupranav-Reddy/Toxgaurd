# ToxGuard

Molecular toxicity prediction using LoRA fine-tuning of the [IUPACGPT](https://github.com/iupacgpt/iupac-gpt) language model.

ToxGuard takes IUPAC chemical names as input and outputs a binary toxic / non-toxic prediction together with a confidence score. It is trained on seven toxicity datasets: ToxCast, Tox21, T3DB, ClinTox, hERG, DILI, and a curated common-molecules set (~23 800 compounds total).

```
IUPAC name  →  GPT-2 + LoRA  →  P(toxic)  →  severity label
```

**Test-set performance (AUC-ROC 0.84 | AUC-PRC 0.88 | Accuracy 0.77)**

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.9 – 3.11 |
| CUDA-enabled GPU | recommended (CPU works but is slow) |
| PyTorch | ≥ 1.13 with CUDA **— install separately, see below** |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/P-Vishnupranav-Reddy/Toxgaurd.git
cd Toxgaurd
```

### 2. Create and activate a virtual environment

```bash
python -m venv toxguard_env

# Windows
toxguard_env\Scripts\activate

# Linux / macOS
source toxguard_env/bin/activate
```

### 3. Install PyTorch with CUDA (do this first)

Visit [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) and select your OS, CUDA version, and package manager. Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Important:** Do not run `pip install torch` from `requirements.txt` — it will install the CPU-only version and overwrite your CUDA build.

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify the setup

```bash
python verify_setup.py
```

This checks that all modules import correctly and that the IUPACGPT tokenizer loads.

---

## Pipeline

Run each step from the project root in order. All steps are safe to re-run.

### Step 1 — Download raw data

```bash
python steps/step1_download_data.py
```

Downloads ToxCast, Tox21, ClinTox, hERG, and DILI raw CSV files into `data/`.

### Step 2 — Preprocess datasets

```bash
python steps/step2_preprocess.py
```

Canonicalises SMILES, binarises labels, deduplicates. Writes `data/*_final.csv` (columns: `smiles, is_toxic`).

### Step 3 — Resolve SMILES → IUPAC names

```bash
python steps/step3_smiles_to_iupac.py
```

Calls PubChem → ChemSpider → NCI CIR in cascade to resolve each unique SMILES to its IUPAC name. Results are cached in `data/step3_cache.csv` — safe to interrupt with Ctrl+C and resume.

Molecules that cannot be resolved are dropped (raw SMILES strings are not valid input for the IUPAC tokenizer).

Re-run from cache only (no API calls):
```bash
python steps/step3_smiles_to_iupac.py --cache-only
```

### Step 4 — Verify LoRA

```bash
python steps/step4_verify_lora.py
```

Injects LoRA adapters into the IUPACGPT backbone, prints a parameter summary, runs a test forward pass, and saves `outputs/lora_config.json`.

```bash
python steps/step4_verify_lora.py --rank 16   # change LoRA rank
```

### Step 5 — Train

```bash
python steps/step5_train.py
```

Fine-tunes with LoRA on all seven datasets. Saves weights to `outputs/<run_name>/lora_weights.pt`.

Common options:
```bash
python steps/step5_train.py --max_epochs 30 --batch_size 16
python steps/step5_train.py --learning_rate 1e-4 --lora_rank 16
```

TensorBoard logs are written to `outputs/<run_name>/tensorboard/`. View them with:
```bash
tensorboard --logdir outputs/
```

### Step 6 — Evaluate

```bash
python steps/step6_evaluate.py
```

Evaluates the latest trained run on the held-out test split. Prints AUC-ROC, AUC-PRC, accuracy, F1, MCC, and a confusion matrix. Saves `evaluation_report.txt` and `eval_metrics.json` to the run directory.

### Step 7 — Predict

```bash
python steps/step7_predict.py
```

Runs predictions on a default set of IUPAC names and prints a formatted results table. To predict custom molecules, pass them on the command line:

```bash
python steps/step7_predict.py "2-(acetyloxy)benzoic acid" "ethanol" "sodium azide"
```

---

## Project Structure

```
ToxGuard/
├── steps/                        # Pipeline scripts (run in order)
│   ├── step1_download_data.py
│   ├── step2_preprocess.py
│   ├── step3_smiles_to_iupac.py
│   ├── step4_verify_lora.py
│   ├── step5_train.py
│   ├── step6_evaluate.py
│   └── step7_predict.py
├── toxguard/                     # Core library
│   ├── model.py                  # ToxGuardModel (IUPACGPT + LoRA + binary head)
│   ├── lora.py                   # LoRA implementation
│   ├── tokenizer.py              # Wrapper around IUPAC SPM tokenizer
│   ├── data_pipeline.py          # Dataset classes + combined loader
│   └── inference.py              # High-level predictor API
├── iupacGPT/iupac-gpt/           # IUPACGPT pretrained model (submodule / cloned)
│   └── checkpoints/iupac/        # GPT-2 weights (not tracked by git)
├── data/                         # Raw and processed datasets
├── outputs/                      # Training runs, LoRA weights, evaluation reports
├── requirements.txt
└── verify_setup.py
```

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Backbone | IUPACGPT (GPT-2), 7.1M params |
| Layers / heads / dim | 8 / 8 / 256 |
| Tokenizer | SentencePiece, vocab 1491 |
| LoRA targets | `c_attn`, `c_proj`, `c_fc` (all 8 layers) |
| LoRA rank | 16 (configurable) |
| Trainable params | ~624K / 7.65M (8.15%) |
| Classification head | Linear(256 → 1) + sigmoid |

---

## Datasets

| Dataset | Compounds (after step 3) | Task |
|---------|--------------------------|------|
| ToxCast | 6 887 | Multi-assay binary |
| hERG | 11 304 | Cardiotoxicity |
| T3DB | 3 505 | Known toxins (all toxic) |
| ClinTox | 648 | FDA clinical toxicity |
| Tox21 | 240 | 12-assay panel |
| DILI | 186 | Drug-induced liver injury |
| Common molecules | 1 048 | Curated IUPAC set |

---

## License

This project uses the [IUPACGPT](https://github.com/iupacgpt/iupac-gpt) pretrained model. Please refer to its license before commercial use.
