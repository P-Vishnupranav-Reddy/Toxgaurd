"""Data pipeline for ToxCast, Tox21, and T3DB toxicity datasets.

Provides PyTorch Dataset classes and a unified data-loading pipeline
for training ToxGuard (IUPACGPT + LoRA) on binary toxicity prediction.

Architecture insight:
  At inference the model receives ONLY an IUPAC name -- no assay data.
  Assay results (617 ToxCast, 12 Tox21) are used solely for computing
  ground-truth binary labels during preprocessing (step 2).

Binary classification:
  ToxCast: toxic if ANY assay is positive (pos_count >= 1)
  Tox21  : toxic if ANY assay is positive (pos_count >= 1)
  T3DB   : all entries are known toxins (is_toxic = 1)

Toxicity score (regression target):
  ToxCast: fraction of positive assays = pos_count / tested_count
  Tox21  : fraction of positive assays = pos_count / tested_count
  T3DB   : derived from LD50 using WHO/GHS classification scale

Datasets (all use *_final.csv format with pre-computed labels):
  - ToxCast : data/toxcast_final.csv   (binary + score from ALL 617 assays)
  - Tox21   : data/tox21_final.csv     (binary + score from 12 assays)
  - T3DB    : data/t3db_processed.csv  (binary + score from LD50)
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Score thresholds for severity display (inference only, not training)
# ──────────────────────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = [0.20, 0.50, 0.65, 0.80]


def _ld50_to_toxicity(ld50_mg_kg: float) -> Tuple[str, float]:
    """WHO/GHS LD50 oral classification -> (class_name, score).

    Score is normalised to [0, 1] based on the LD50 band.
    Lower LD50 = more toxic = higher score.
    """
    if ld50_mg_kg <= 5:
        return "supertoxic", 0.95
    if ld50_mg_kg <= 50:
        return "extremely toxic", 0.85
    if ld50_mg_kg <= 300:
        return "very toxic", 0.70
    if ld50_mg_kg <= 2000:
        return "toxic", 0.55
    if ld50_mg_kg <= 5000:
        return "harmful", 0.35
    if ld50_mg_kg <= 15000:
        return "slightly toxic", 0.20
    return "non-toxic", 0.05


# ──────────────────────────────────────────────────────────────────────
# Local T3DB processing (from manually downloaded files)
# ──────────────────────────────────────────────────────────────────────

def process_local_t3db(t3db_dir: str = "./data/t3db",
                       save_dir: str = "./data") -> str:
    """Process locally downloaded T3DB CSV files into a clean dataset.

    Expects these files in *t3db_dir* (from www.t3db.ca bulk download):
        - toxin_structures.csv   (SMILES, IUPAC names, molecular descriptors)
        - all_toxin_data.csv     (toxicity info, LD50, health effects)
        - target_mechanisms.csv  (toxin -> target mechanism links)  [optional]
        - all_toxin_target.csv   (protein target info)              [optional]

    Processing steps:
        1. Load structures (has SMILES + JCHEM_IUPAC for ~3,500 compounds)
        2. Load toxin data (has LD50, health_effects, types for ~3,900 entries)
        3. Join on compound name (NAME <-> common_name)
        4. Parse LD50 values from free-text -> numeric mg/kg
        5. Derive toxicity_class + toxicity_score from LD50 and health_effects
        6. Deduplicate on SMILES, save to *save_dir*/t3db_processed.csv

    Returns:
        Path to the saved t3db_processed.csv
    """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "t3db_processed.csv")

    # -- 1. Load structures ---
    struct_path = os.path.join(t3db_dir, "toxin_structures.csv")
    if not os.path.exists(struct_path):
        raise FileNotFoundError(
            f"toxin_structures.csv not found in {t3db_dir}. "
            "Please download T3DB bulk data from www.t3db.ca."
        )

    structs = pd.read_csv(struct_path)
    logger.info(f"Loaded toxin_structures.csv: {len(structs)} rows")

    if "smiles" in structs.columns:
        smiles_col = "smiles"
    elif "SMILES" in structs.columns:
        smiles_col = "SMILES"
    else:
        raise ValueError("No SMILES column in toxin_structures.csv")

    structs_clean = pd.DataFrame({
        "smiles":      structs[smiles_col],
        "iupac_name":  structs.get("JCHEM_IUPAC", structs.get("JCHEM_TRADITIONAL_IUPAC", pd.Series())),
        "common_name": structs.get("NAME", pd.Series()),
        "t3db_id":     structs.get("T3DB_ID", pd.Series()),
        "cas":         structs.get("CAS", pd.Series()),
        "logp":        pd.to_numeric(structs.get("JCHEM_LOGP", pd.Series()), errors="coerce"),
        "mol_weight":  pd.to_numeric(structs.get("MOLECULAR_WEIGHT", pd.Series()), errors="coerce"),
    })

    structs_clean = structs_clean.dropna(subset=["smiles"])
    structs_clean["smiles"] = structs_clean["smiles"].astype(str).str.strip()
    structs_clean = structs_clean[structs_clean["smiles"].str.len() > 0]
    logger.info(f"  Structures with valid SMILES: {len(structs_clean)}")

    # -- 2. Load toxin data ---
    data_path = os.path.join(t3db_dir, "all_toxin_data.csv")
    if os.path.exists(data_path):
        toxdata = pd.read_csv(data_path)
        logger.info(f"Loaded all_toxin_data.csv: {len(toxdata)} rows")
        toxdata_clean = pd.DataFrame({
            "common_name_key": toxdata["common_name"].fillna("").str.lower().str.strip(),
            "toxicity_text":   toxdata.get("toxicity", pd.Series(dtype=str)),
            "lethaldose_text": toxdata.get("lethaldose", pd.Series(dtype=str)),
            "health_effects":  toxdata.get("health_effects", pd.Series(dtype=str)),
            "types_raw":       toxdata.get("types", pd.Series(dtype=str)),
            "td_smiles":       toxdata.get("moldb_smiles", pd.Series(dtype=str)),
        })
    else:
        logger.warning(f"all_toxin_data.csv not found in {t3db_dir} -- proceeding without toxicity info")
        toxdata_clean = pd.DataFrame()

    # -- 3. Join on name ---
    structs_clean["common_name_key"] = structs_clean["common_name"].fillna("").str.lower().str.strip()

    if len(toxdata_clean) > 0:
        toxdata_dedup = toxdata_clean.drop_duplicates(subset=["common_name_key"])
        merged = structs_clean.merge(toxdata_dedup, on="common_name_key", how="left")
        logger.info(f"  Merged structures + toxin data: {len(merged)} rows "
                     f"({merged['lethaldose_text'].notna().sum()} with LD50, "
                     f"{merged['health_effects'].notna().sum()} with health effects)")
    else:
        merged = structs_clean.copy()

    # -- 4. Parse LD50 ---
    merged["ld50_mg_kg"] = merged.get("lethaldose_text", pd.Series(dtype=str)).apply(_parse_ld50_text)

    if "toxicity_text" in merged.columns:
        still_missing = merged["ld50_mg_kg"].isna()
        n_before = still_missing.sum()
        merged.loc[still_missing, "ld50_mg_kg"] = (
            merged.loc[still_missing, "toxicity_text"].apply(_parse_ld50_text)
        )
        n_recovered = n_before - merged["ld50_mg_kg"].isna().sum()
        if n_recovered > 0:
            logger.info(f"  LD50 fallback: recovered {n_recovered} values from toxicity_text")

    # -- 5. Derive toxicity labels ---
    tox_classes = []
    tox_scores = []

    for _, row in merged.iterrows():
        ld50_val = row.get("ld50_mg_kg")
        health   = str(row.get("health_effects", "")) if pd.notna(row.get("health_effects")) else ""
        tox_text = str(row.get("toxicity_text", ""))   if pd.notna(row.get("toxicity_text"))   else ""

        if pd.notna(ld50_val) and ld50_val > 0:
            cls, score = _ld50_to_toxicity(ld50_val)
            tox_classes.append(cls)
            tox_scores.append(score)
            continue

        combined_text = (tox_text + " " + health).lower()
        cls, score = _text_to_toxicity(combined_text)
        tox_classes.append(cls)
        tox_scores.append(score)

    merged["toxicity_class"] = tox_classes
    merged["toxicity_score"] = tox_scores
    # All T3DB entries are known toxins — is_toxic is always 1
    merged["is_toxic"] = 1
    merged["source"] = "t3db_local"

    # -- 6. Select columns, deduplicate ---
    final_cols = [
        "smiles", "iupac_name", "common_name", "t3db_id", "cas",
        "toxicity_class", "toxicity_score", "is_toxic",
        "ld50_mg_kg", "logp", "mol_weight", "source",
    ]
    final_cols = [c for c in final_cols if c in merged.columns]
    result = merged[final_cols].copy()

    before = len(result)
    result = result.drop_duplicates(subset=["smiles"])
    logger.info(f"  Deduplicated: {before} -> {len(result)} unique SMILES")

    result.to_csv(output_path, index=False)
    logger.info(f"  Saved {output_path} ({len(result)} compounds)")

    return output_path


def _parse_ld50_text(text) -> Optional[float]:
    """Extract LD50 in mg/kg from free-text strings.

    Returns the *lowest* LD50 value found (most conservative / most toxic).
    Returns None if no parseable LD50 is found.
    """
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return None

    text = text.lower()
    values = []

    for m in re.finditer(r'ld50[:\s=]*(\d+(?:\.\d+)?)\s*mg/kg', text):
        values.append(float(m.group(1)))
    for m in re.finditer(r'ld50[:\s=]*(\d+(?:\.\d+)?)\s*g/kg', text):
        values.append(float(m.group(1)) * 1000)
    for m in re.finditer(r'ld50[:\s=]*(\d+(?:\.\d+)?)\s*(?:ug|mcg)/kg', text):
        values.append(float(m.group(1)) / 1000)
    for m in re.finditer(r'lethal\s+(?:oral\s+)?dose[^.]*?(\d+(?:\.\d+)?)\s*mg/kg', text):
        values.append(float(m.group(1)))
    if not values:
        m = re.search(r'(\d+(?:\.\d+)?)\s*mg/kg', text)
        if m:
            values.append(float(m.group(1)))
    if not values:
        m = re.search(r'(\d+(?:\.\d+)?)\s*mg\b(?:\s+for\s+an?\s+adult)', text)
        if m:
            values.append(float(m.group(1)) / 70.0)

    return min(values) if values else None


def _text_to_toxicity(text: str) -> Tuple[str, float]:
    """Infer toxicity class + score from free-text descriptions."""
    if not text or len(text.strip()) == 0:
        return "toxic", 0.55

    text = text.lower()

    fatal_kw = ["fatal", "lethal", "death", "die ", "dies ", "kill",
                 "organ failure", "cardiac arrest"]
    if any(kw in text for kw in fatal_kw):
        return "very toxic", 0.80

    cancer_kw = ["carcinogen", "mutagenic", "genotoxic", "tumori",
                 "cancer", "neoplasm"]
    if any(kw in text for kw in cancer_kw):
        return "toxic", 0.65

    severe_kw = ["kidney damage", "liver damage", "renal failure",
                 "hepatotoxic", "nephrotoxic", "neurotoxic",
                 "brain damage", "nerve damage", "respiratory failure",
                 "pulmonary edema", "seizure", "convulsion", "coma"]
    if any(kw in text for kw in severe_kw):
        return "toxic", 0.60

    moderate_kw = ["irritat", "nausea", "vomit", "headache", "dizziness",
                   "dermatitis", "rash", "burn", "corrosive"]
    if any(kw in text for kw in moderate_kw):
        return "harmful", 0.40

    mild_kw = ["mild", "slight", "minor", "low toxicity"]
    if any(kw in text for kw in mild_kw):
        return "slightly toxic", 0.25

    return "toxic", 0.55


# ──────────────────────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────────────────────

class MoleculeDataset(Dataset):
    """Generic PyTorch Dataset for any toxicity CSV with IUPAC names.

    Consolidates ToxCast, Tox21, ClinTox, hERG, DILI, and Common Molecules
    datasets — all share identical loading and tokenisation logic.

    Required columns: iupac_name (configurable via ``iupac_col``), is_toxic

    Args:
        csv_path: Path to the CSV file.
        tokenizer: ToxGuardTokenizer instance for IUPAC tokenisation.
        max_length: Maximum token length (default 1024).
        iupac_col: Column name containing IUPAC names (default "iupac_name").
        dataset_name: Label used in log messages (defaults to CSV filename stem).
        all_toxic: If True, forces all binary_labels to 1 (used by T3DB subclass).
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 1024,
        iupac_col: str = "iupac_name",
        dataset_name: str = "",
        all_toxic: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        name_tag = dataset_name or os.path.splitext(os.path.basename(csv_path))[0]

        df = pd.read_csv(csv_path)
        if iupac_col not in df.columns:
            raise ValueError(
                f"Column '{iupac_col}' not found in {csv_path}.\n"
                "Run step 3 first: python steps/step3_smiles_to_iupac.py"
            )
        df = df.dropna(subset=[iupac_col])
        self.iupac_names = df[iupac_col].values

        if all_toxic:
            self.binary_labels = np.ones(len(df), dtype=np.int64)
            logger.info(f"{name_tag}: {len(self)} compounds (all toxic)")
        else:
            self.binary_labels = df["is_toxic"].values.astype(np.int64)
            logger.info(f"{name_tag}: {len(self)} compounds from {csv_path}")
            logger.info(f"  Toxic: {int(self.binary_labels.sum())} | "
                        f"Non-toxic: {int((self.binary_labels == 0).sum())}")

    def __len__(self):
        return len(self.iupac_names)

    def __getitem__(self, idx):
        name = self.iupac_names[idx]
        tokenized = self.tokenizer(str(name))
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)

        bos = torch.tensor([self.tokenizer._convert_token_to_id(self.tokenizer.unk_token)])
        input_ids = torch.cat([bos, input_ids])
        attention_mask = torch.ones(input_ids.numel(), dtype=torch.long)

        return {
            "input_ids": input_ids[:self.max_length],
            "attention_mask": attention_mask[:self.max_length],
            "binary_labels": torch.tensor(self.binary_labels[idx], dtype=torch.float),
        }


# Backward-compatible aliases — existing code using the old class names continues to work.
ToxCastDataset         = MoleculeDataset
Tox21Dataset           = MoleculeDataset
ClinToxDataset         = MoleculeDataset
HErgDataset            = MoleculeDataset
DILIDataset            = MoleculeDataset
CommonMoleculesDataset = MoleculeDataset


class T3DBDataset(MoleculeDataset):
    """Dataset for T3DB — all entries are known toxins (is_toxic forced to 1).

    Thin subclass of MoleculeDataset with ``all_toxic=True`` so no ``is_toxic``
    column is required in the CSV.
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int = 1024,
                 iupac_col: str = "iupac_name", dataset_name: str = "T3DBDataset"):
        super().__init__(
            csv_path, tokenizer, max_length, iupac_col,
            dataset_name=dataset_name, all_toxic=True,
        )


class ToxicityDataset(Dataset):
    """Combined dataset merging ToxCast, Tox21, T3DB, ClinTox, and Common Molecules."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                if i == 0:
                    return self.datasets[i][idx]
                else:
                    return self.datasets[i][idx - self.cumulative_sizes[i - 1]]
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")


# ──────────────────────────────────────────────────────────────────────
# Collator
# ──────────────────────────────────────────────────────────────────────

class ToxicityCollator:
    """Collator that pads input_ids and attention_mask, stacks labels."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, records: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}

        # Pad variable-length sequences
        batch["input_ids"] = pad_sequence(
            [r["input_ids"] for r in records],
            batch_first=True, padding_value=self.pad_token_id
        )
        batch["attention_mask"] = pad_sequence(
            [r["attention_mask"] for r in records],
            batch_first=True, padding_value=0
        )

        # Stack fixed-size tensors
        batch["binary_labels"] = torch.stack([r["binary_labels"] for r in records])

        return batch


# ──────────────────────────────────────────────────────────────────────
# Data Preparation Pipeline
# ──────────────────────────────────────────────────────────────────────

def prepare_combined_dataset(
    data_dir: str = "./data",
    tokenizer=None,
    max_length: int = 1024,
    val_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Full pipeline: load all datasets -> create DataLoaders.

    Dataset sources (all with pre-computed binary labels from step 2):
      - ToxCast          : data/toxcast_final.csv       (binary from 617 assays)
      - Tox21            : data/tox21_final.csv         (binary from 12 assays)
      - T3DB             : data/t3db_processed.csv      (all toxic)
      - ClinTox          : data/clintox_final.csv       (clinical-trial toxicity)
      - hERG             : data/herg_final.csv          (cardiotoxicity — hERG blocker)
      - DILI             : data/dili_final.csv          (drug-induced liver injury)
      - Common Molecules : data/common_molecules_final.csv (curated short IUPAC)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Dataset file paths
    toxcast_path   = os.path.join(data_dir, "toxcast_final.csv")
    tox21_path     = os.path.join(data_dir, "tox21_final.csv")
    t3db_path      = os.path.join(data_dir, "t3db_processed.csv")
    clintox_path   = os.path.join(data_dir, "clintox_final.csv")
    herg_path      = os.path.join(data_dir, "herg_final.csv")
    dili_path      = os.path.join(data_dir, "dili_final.csv")
    common_path    = os.path.join(data_dir, "common_molecules_final.csv")

    # Create datasets
    datasets_list = []

    for name, path, cls in [
        ("ToxCast",          toxcast_path, ToxCastDataset),
        ("Tox21",            tox21_path,   Tox21Dataset),
        ("T3DB",             t3db_path,    T3DBDataset),
        ("ClinTox",          clintox_path, ClinToxDataset),
        ("hERG",             herg_path,    HErgDataset),
        ("DILI",             dili_path,    DILIDataset),
        ("CommonMolecules",  common_path,  CommonMoleculesDataset),
    ]:
        if os.path.exists(path):
            try:
                ds = cls(path, tokenizer, max_length, dataset_name=name)
                datasets_list.append(ds)
                logger.info(f"Loaded {name}: {len(ds)} compounds")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
        else:
            logger.warning(f"{name} not found at {path}. Run steps 1-3 first.")

    if not datasets_list:
        raise RuntimeError(
            "No datasets could be loaded! Run steps 1-3 to produce the "
            "required CSV files. See steps/step1_download_data.py."
        )

    combined = ToxicityDataset(datasets_list)

    # -- Stratified split by binary label + source dataset ---
    total = len(combined)
    strat_labels = []
    for idx in range(total):
        sample = combined[idx]
        binary = int(sample["binary_labels"].item())
        for di, size in enumerate(combined.cumulative_sizes):
            if idx < size:
                ds_idx = di
                break
        strat_labels.append(f"{ds_idx}:{binary}")
    strat_labels = np.array(strat_labels)

    # First split: train+val vs test
    test_frac = test_split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
    trainval_idx, test_idx = next(sss1.split(np.zeros(total), strat_labels))

    # Second split: train vs val (from train+val portion)
    val_frac_of_trainval = val_split / (1.0 - test_split)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=42)
    trainval_strat = strat_labels[trainval_idx]
    train_sub_idx, val_sub_idx = next(sss2.split(np.zeros(len(trainval_idx)), trainval_strat))

    train_indices = trainval_idx[train_sub_idx]
    val_indices = trainval_idx[val_sub_idx]
    test_indices = test_idx

    train_ds = Subset(combined, train_indices.tolist())
    val_ds   = Subset(combined, val_indices.tolist())
    test_ds  = Subset(combined, test_indices.tolist())

    # DataLoaders
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
        logger.warning(f"pad_token_id is None -- falling back to {pad_id}")
    collator = ToxicityCollator(pad_id)

    # persistent_workers keeps worker processes alive between epochs
    # pin_memory speeds up CPU->GPU transfer
    _persistent = num_workers > 0
    _pin_memory  = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers,
        persistent_workers=_persistent, pin_memory=_pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers,
        persistent_workers=_persistent, pin_memory=_pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers,
        persistent_workers=_persistent, pin_memory=_pin_memory,
    )

    logger.info(f"Combined: {total} compounds | "
                f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Compute class distribution for weighting
    train_labels = [combined[i]["binary_labels"].item() for i in train_indices.tolist()]
    n_pos = sum(1 for l in train_labels if l >= 0.5)
    n_neg = len(train_labels) - n_pos
    logger.info(f"Training class distribution: {n_pos} toxic, {n_neg} non-toxic "
                f"(ratio {n_pos / max(n_neg, 1):.2f}:1)")

    return train_loader, val_loader, test_loader, {"n_positive": n_pos, "n_negative": n_neg}
