#!/usr/bin/env python3
"""
STEP 2 -- Preprocess & Compute Binary Labels
==============================================
Reads raw datasets, checks unique SMILES, performs cross-dataset dedup,
computes binary toxic/non-toxic labels, and saves final CSVs (without IUPAC names).

Inputs:
  data/toxcast_raw.csv        (8,597 cpds x 617 assays)
  data/tox21_raw.csv          (7,831 cpds x 12 assays)
  data/t3db_processed.csv     (3,512 cpds, LD50-based labels)
  data/clintox_raw.csv        (1,484 cpds, FDA approval + clinical tox)
  data/herg_raw.csv           (~13,445 cpds, cardiac ion channel blocking)
  data/dili_raw.csv           (~475 cpds, drug-induced liver injury)
  data/common_molecules_raw.csv  (curated ~1000-1200 short-IUPAC molecules)

Outputs:
  data/toxcast_final.csv        (smiles, is_toxic)
  data/tox21_final.csv          (smiles, is_toxic)
  data/clintox_final.csv        (smiles, is_toxic)
  data/herg_final.csv           (smiles, is_toxic)
  data/dili_final.csv           (smiles, is_toxic)
  data/common_molecules_final.csv (iupac_name, is_toxic)

NOTE: These files do NOT contain IUPAC names yet (except common_molecules_final
      which already has them). Step 3 adds IUPAC names via cache + API and
      drops compounds without them.

Binary classification:
  ToxCast  (617 assays): toxic = any assay positive (pos_count >= 1)
  Tox21    (12 assays) : toxic = any assay positive (pos_count >= 1)
  ClinTox              : toxic = CT_TOX == 1 (failed clinical trials due to toxicity)
                         non-toxic = CT_TOX == 0 AND FDA_APPROVED == 1
  hERG                 : toxic = is_herg_blocker == 1 (blocks cardiac ion channel)
  DILI                 : toxic = is_dili == 1 (causes drug-induced liver injury)
  Common molecules     : curated toxic/non-toxic labels from authoritative sources

Cross-dataset deduplication (by canonical SMILES):
  Priority: T3DB > ToxCast > hERG > Tox21 > ClinTox > DILI
  1. ToxCast ∩ T3DB     -> remove from ToxCast
  2. Tox21 ∩ ToxCast    -> remove from Tox21
  3. Tox21 ∩ T3DB       -> remove from Tox21
  4. hERG  ∩ (T3DB+TC)  -> remove from hERG
  5. ClinTox ∩ (all)    -> remove from ClinTox
  6. DILI  ∩ (all)      -> remove from DILI

Run from project root:
  python steps/step2_preprocess.py
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "./data"

# ── RDKit canonicalisation ────────────────────────────────────────────
try:
    from rdkit import Chem, RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available -- dedup will use raw SMILES strings")


def canonicalize_smiles(smi: str) -> str | None:
    """Return canonical isomeric SMILES (largest fragment, stereo preserved).

    Returns None if RDKit cannot parse the SMILES.
    Strips salts by keeping the fragment with the most heavy atoms.
    """
    if not RDKIT_AVAILABLE:
        return smi
    if not smi or not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if frags and len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def canonicalize_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Replace SMILES column with canonical SMILES; drop unparseable rows."""
    original = len(df)
    df = df.copy()
    df["smiles"] = df["smiles"].apply(canonicalize_smiles)
    n_failed = df["smiles"].isna().sum()
    if n_failed:
        logger.warning(f"  [{name}] {n_failed} SMILES failed RDKit parse -- dropping")
        df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    logger.info(f"  [{name}] Canonicalised {original} → {len(df)} rows (dropped {n_failed} unparseable)")
    return df


# ── Within-dataset unique SMILES check ────────────────────────────────

def check_unique_smiles(df: pd.DataFrame, name: str, smiles_col: str = "smiles") -> pd.DataFrame:
    """Canonicalise and deduplicate SMILES within a single dataset.

    For rows with conflicting labels on the same canonical SMILES,
    keeps the toxic label (conservative: any positive = toxic).
    """
    total = len(df)

    # Step 1: canonicalise
    df = canonicalize_df(df, name)

    # Step 2: resolve label conflicts on same canonical SMILES (keep toxic)
    if "is_toxic" in df.columns:
        before_conflict = len(df)
        df = (
            df.groupby(smiles_col, sort=False)
            .agg({"is_toxic": "max", **{c: "first" for c in df.columns if c not in [smiles_col, "is_toxic"]}})
            .reset_index()
        )
        n_conflicts = before_conflict - len(df)
        if n_conflicts > 0:
            logger.info(f"  [{name}] Resolved {n_conflicts} within-dataset canonical SMILES duplicates (kept toxic label)")
    else:
        df = df.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)

    n_unique = len(df)
    n_dups = total - n_unique
    logger.info(f"  [{name}] Total rows: {total} | Unique canonical SMILES: {n_unique} | Collapsed: {n_dups}")

    return df


# ── Load raw datasets ─────────────────────────────────────────────────

def load_toxcast_raw() -> pd.DataFrame:
    """Load raw ToxCast and clean SMILES."""
    path = os.path.join(DATA_DIR, "toxcast_raw.csv")
    df = pd.read_csv(path)
    logger.info(f"ToxCast raw: {len(df)} rows, {len(df.columns)} columns")

    smiles_col = next((c for c in ["smiles", "SMILES"] if c in df.columns), None)
    df = df.dropna(subset=[smiles_col])
    df["smiles"] = df[smiles_col].astype(str).str.strip()
    df = df[df["smiles"].str.len() > 0]
    df = check_unique_smiles(df, "ToxCast")

    return df


def load_tox21_raw() -> pd.DataFrame:
    """Load raw Tox21 and clean SMILES."""
    path = os.path.join(DATA_DIR, "tox21_raw.csv")
    df = pd.read_csv(path)
    logger.info(f"Tox21 raw: {len(df)} rows, {len(df.columns)} columns")

    smiles_col = next((c for c in ["smiles", "SMILES"] if c in df.columns), None)
    df = df.dropna(subset=[smiles_col])
    df["smiles"] = df[smiles_col].astype(str).str.strip()
    df = df[df["smiles"].str.len() > 0]
    df = check_unique_smiles(df, "Tox21")

    return df


def load_t3db_smiles() -> set:
    """Load T3DB canonical SMILES for cross-dedup (T3DB itself stays as-is)."""
    path = os.path.join(DATA_DIR, "t3db_processed.csv")
    if not os.path.exists(path):
        logger.warning("t3db_processed.csv not found -- skipping T3DB dedup")
        return set()

    df = pd.read_csv(path, usecols=["smiles"])
    raw_smiles = df["smiles"].dropna().astype(str).str.strip().tolist()
    canonical = {c for c in (canonicalize_smiles(s) for s in raw_smiles) if c}
    logger.info(f"T3DB: {len(canonical)} unique canonical SMILES for cross-dedup")
    return canonical


# ── Cross-dataset deduplication ──────────────────────────────────────

def cross_dedup(toxcast_df: pd.DataFrame, tox21_df: pd.DataFrame,
                t3db_smiles: set) -> tuple:
    """Remove cross-dataset canonical SMILES overlaps.

    Priority: T3DB > ToxCast > Tox21
      1. ToxCast ∩ T3DB   -> remove from ToxCast
      2. Tox21 ∩ ToxCast  -> remove from Tox21  (AFTER ToxCast dedup)
      3. Tox21 ∩ T3DB     -> remove from Tox21

    NOTE: Both DataFrames must already have canonical SMILES in the
    'smiles' column (done by check_unique_smiles → canonicalize_df).
    t3db_smiles must also be canonical (from load_t3db_smiles).
    """
    tc_smiles = set(toxcast_df["smiles"].values)

    # 1. Remove ToxCast compounds already in T3DB
    overlap_tc_t3db = tc_smiles & t3db_smiles
    if overlap_tc_t3db:
        toxcast_df = toxcast_df[~toxcast_df["smiles"].isin(overlap_tc_t3db)].reset_index(drop=True)
        logger.info(f"  Dedup: removed {len(overlap_tc_t3db)} ToxCast compounds in T3DB")

    # 2. Remove Tox21 compounds already in (remaining) ToxCast
    remaining_tc = set(toxcast_df["smiles"].values)
    t21_smiles = set(tox21_df["smiles"].values)
    overlap_t21_tc = t21_smiles & remaining_tc
    if overlap_t21_tc:
        tox21_df = tox21_df[~tox21_df["smiles"].isin(overlap_t21_tc)].reset_index(drop=True)
        logger.info(f"  Dedup: removed {len(overlap_t21_tc)} Tox21 compounds in ToxCast")

    # 3. Remove Tox21 compounds already in T3DB
    remaining_t21 = set(tox21_df["smiles"].values)
    overlap_t21_t3db = remaining_t21 & t3db_smiles
    if overlap_t21_t3db:
        tox21_df = tox21_df[~tox21_df["smiles"].isin(overlap_t21_t3db)].reset_index(drop=True)
        logger.info(f"  Dedup: removed {len(overlap_t21_t3db)} Tox21 compounds in T3DB")

    logger.info(f"  After cross-dedup: ToxCast={len(toxcast_df)} | Tox21={len(tox21_df)}")
    return toxcast_df, tox21_df


# ── Binary + Score computation ────────────────────────────────────────

def compute_toxcast_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binary toxic label from ALL 617 ToxCast assays.

    Binary: toxic = any assay positive (pos_count >= 1)

    Returns DataFrame with: smiles, is_toxic
    """
    smiles_col = next((c for c in ["smiles", "SMILES"] if c in df.columns), "smiles")
    assay_cols = [c for c in df.columns if c not in [smiles_col, "smiles"]]

    assay_matrix = df[assay_cols].values.astype(float)

    # Count positives and tested per compound
    pos_count = np.nansum(assay_matrix == 1.0, axis=1).astype(int)

    # Binary: toxic if any assay positive
    is_toxic = (pos_count >= 1).astype(int)

    logger.info(f"  Assays used: {len(assay_cols)}")
    n_toxic = int(is_toxic.sum())
    n_nontoxic = int((is_toxic == 0).sum())
    logger.info(f"  Toxic: {n_toxic} ({100*n_toxic/len(df):.1f}%) | "
                f"Non-toxic: {n_nontoxic} ({100*n_nontoxic/len(df):.1f}%)")

    return pd.DataFrame({
        "smiles": df["smiles"].values,
        "is_toxic": is_toxic,
    })


def compute_tox21_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binary toxic label from 12 Tox21 assays.

    Binary: toxic = any assay positive (pos_count >= 1)

    Returns DataFrame with: smiles, is_toxic
    """
    smiles_col = next((c for c in ["smiles", "SMILES"] if c in df.columns), "smiles")
    skip_cols = {smiles_col, "smiles", "mol_id", "Unnamed: 0"}
    assay_cols = [c for c in df.columns if c not in skip_cols]

    assay_matrix = df[assay_cols].values.astype(float)
    pos_count = np.nansum(assay_matrix == 1.0, axis=1).astype(int)

    is_toxic = (pos_count >= 1).astype(int)

    logger.info(f"  Assays used: {len(assay_cols)}")
    n_toxic = int(is_toxic.sum())
    n_nontoxic = int((is_toxic == 0).sum())
    logger.info(f"  Toxic: {n_toxic} ({100*n_toxic/len(df):.1f}%) | "
                f"Non-toxic: {n_nontoxic} ({100*n_nontoxic/len(df):.1f}%)")

    return pd.DataFrame({
        "smiles": df["smiles"].values,
        "is_toxic": is_toxic,
    })


# ── Main ──────────────────────────────────────────────────────────────

def load_clintox_raw() -> pd.DataFrame:
    """Load raw ClinTox and clean SMILES."""
    path = os.path.join(DATA_DIR, "clintox_raw.csv")
    if not os.path.exists(path):
        logger.warning("clintox_raw.csv not found -- skipping ClinTox")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"ClinTox raw: {len(df)} rows, {len(df.columns)} columns")

    smiles_col = next((c for c in ["smiles", "SMILES"] if c in df.columns), None)
    df = df.dropna(subset=[smiles_col])
    df["smiles"] = df[smiles_col].astype(str).str.strip()
    df = df[df["smiles"].str.len() > 0]
    df = check_unique_smiles(df, "ClinTox")

    return df


def load_herg_raw() -> pd.DataFrame:
    """Load raw hERG dataset and clean SMILES."""
    path = os.path.join(DATA_DIR, "herg_raw.csv")
    if not os.path.exists(path):
        logger.warning("herg_raw.csv not found -- run step 1 first")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"hERG raw: {len(df)} rows")

    smiles_col = next((c for c in ["smiles", "Drug", "SMILES"] if c in df.columns), None)
    if smiles_col is None:
        logger.warning("hERG: no SMILES column found -- skipping")
        return pd.DataFrame()

    df = df.dropna(subset=[smiles_col])
    df["smiles"] = df[smiles_col].astype(str).str.strip()
    df = df[df["smiles"].str.len() > 0]
    df = check_unique_smiles(df, "hERG")
    return df


def load_dili_raw() -> pd.DataFrame:
    """Load raw DILI dataset and clean SMILES."""
    path = os.path.join(DATA_DIR, "dili_raw.csv")
    if not os.path.exists(path):
        logger.warning("dili_raw.csv not found -- run step 1 first")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"DILI raw: {len(df)} rows")

    smiles_col = next((c for c in ["smiles", "Drug", "SMILES"] if c in df.columns), None)
    if smiles_col is None:
        logger.warning("DILI: no SMILES column found -- skipping")
        return pd.DataFrame()

    df = df.dropna(subset=[smiles_col])
    df["smiles"] = df[smiles_col].astype(str).str.strip()
    df = df[df["smiles"].str.len() > 0]
    df = check_unique_smiles(df, "DILI")
    return df


def compute_herg_labels(df: pd.DataFrame, existing_smiles: set) -> pd.DataFrame:
    """Binary label for hERG: 1 = hERG blocker (cardiotoxic), 0 = non-blocker.

    Removes compounds already in higher-priority datasets.
    """
    if df.empty:
        return df

    before = len(df)
    df = df[~df["smiles"].isin(existing_smiles)].reset_index(drop=True)
    n_removed = before - len(df)
    if n_removed:
        logger.info(f"  Dedup: removed {n_removed} hERG compounds already in higher-priority datasets")

    label_col = next((c for c in ["is_herg_blocker", "Y"] if c in df.columns), None)
    if label_col is None:
        logger.error("hERG: no label column found")
        return pd.DataFrame()

    is_toxic = df[label_col].fillna(0).astype(int).values
    n_toxic   = int(is_toxic.sum())
    n_nontoxic = int((is_toxic == 0).sum())
    logger.info(f"  hERG: Toxic (blocker)={n_toxic} ({100*n_toxic/max(len(df),1):.1f}%) | "
                f"Non-toxic={n_nontoxic} ({100*n_nontoxic/max(len(df),1):.1f}%)")

    return pd.DataFrame({"smiles": df["smiles"].values, "is_toxic": is_toxic})


def compute_dili_labels(df: pd.DataFrame, existing_smiles: set) -> pd.DataFrame:
    """Binary label for DILI: 1 = liver injury concern, 0 = no concern.

    Removes compounds already in higher-priority datasets.
    """
    if df.empty:
        return df

    before = len(df)
    df = df[~df["smiles"].isin(existing_smiles)].reset_index(drop=True)
    n_removed = before - len(df)
    if n_removed:
        logger.info(f"  Dedup: removed {n_removed} DILI compounds already in higher-priority datasets")

    label_col = next((c for c in ["is_dili", "Y"] if c in df.columns), None)
    if label_col is None:
        logger.error("DILI: no label column found")
        return pd.DataFrame()

    is_toxic = df[label_col].fillna(0).astype(int).values
    n_toxic    = int(is_toxic.sum())
    n_nontoxic = int((is_toxic == 0).sum())
    logger.info(f"  DILI: Toxic={n_toxic} ({100*n_toxic/max(len(df),1):.1f}%) | "
                f"Non-toxic={n_nontoxic} ({100*n_nontoxic/max(len(df),1):.1f}%)")

    return pd.DataFrame({"smiles": df["smiles"].values, "is_toxic": is_toxic})


def compute_clintox_labels(df: pd.DataFrame, existing_smiles: set) -> pd.DataFrame:
    """Compute binary toxic label from ClinTox data.

    Classification logic:
      CT_TOX == 1                        -> toxic   (failed clinical trials due to toxicity)
      CT_TOX == 0 AND FDA_APPROVED == 1  -> non-toxic (FDA-approved, no toxicity failure)
      CT_TOX == 0 AND FDA_APPROVED == 0  -> excluded (ambiguous — not approved, but not toxic either)

    Also deduplicates against all existing dataset canonical SMILES.

    Returns DataFrame with: smiles, is_toxic
    """
    # Remove overlap with existing datasets (existing_smiles are already canonical)
    before = len(df)
    df = df[~df["smiles"].isin(existing_smiles)].reset_index(drop=True)
    n_removed = before - len(df)
    if n_removed > 0:
        logger.info(f"  Dedup: removed {n_removed} ClinTox compounds already in other datasets")

    # Exclude ambiguous cases: not FDA-approved AND not toxic
    ambiguous = (df["CT_TOX"] == 0) & (df["FDA_APPROVED"] == 0)
    n_ambiguous = int(ambiguous.sum())
    if n_ambiguous > 0:
        df = df[~ambiguous].reset_index(drop=True)
        logger.info(f"  Excluded {n_ambiguous} ambiguous (CT_TOX=0, FDA_APPROVED=0)")

    # Binary: toxic if CT_TOX == 1
    is_toxic = (df["CT_TOX"] == 1).astype(int).values

    n_toxic = int(is_toxic.sum())
    n_nontoxic = int((is_toxic == 0).sum())
    logger.info(f"  Toxic: {n_toxic} ({100*n_toxic/len(df):.1f}%) | "
                f"Non-toxic: {n_nontoxic} ({100*n_nontoxic/len(df):.1f}%)")

    return pd.DataFrame({
        "smiles": df["smiles"].values,
        "is_toxic": is_toxic,
    })


def main():
    print("\n" + "=" * 60)
    print("  STEP 2 -- Preprocess & Compute Binary Labels")
    print("=" * 60)

    # Check inputs exist
    for name, path in [
        ("ToxCast raw", os.path.join(DATA_DIR, "toxcast_raw.csv")),
        ("Tox21 raw",   os.path.join(DATA_DIR, "tox21_raw.csv")),
    ]:
        if not os.path.exists(path):
            print(f"\n[ERROR] {path} not found.")
            print("  -> Run step 1 first: python steps/step1_download_data.py")
            sys.exit(1)

    # ── 1. Load raw datasets ──
    print("\n-- Loading raw datasets --")
    toxcast_raw = load_toxcast_raw()
    tox21_raw   = load_tox21_raw()
    t3db_smiles = load_t3db_smiles()
    clintox_raw = load_clintox_raw()
    herg_raw    = load_herg_raw()
    dili_raw    = load_dili_raw()

    # ── 2. Cross-dataset deduplication ──
    print("\n-- Cross-dataset deduplication --")
    toxcast_raw, tox21_raw = cross_dedup(toxcast_raw, tox21_raw, t3db_smiles)

    # ── 3. Compute binary labels ──
    print("\n-- Computing ToxCast labels (617 assays) --")
    toxcast_final = compute_toxcast_labels(toxcast_raw)

    print("\n-- Computing Tox21 labels (12 assays) --")
    tox21_final = compute_tox21_labels(tox21_raw)

    # hERG: dedup against T3DB + ToxCast (higher priority)
    herg_final = pd.DataFrame()
    if len(herg_raw) > 0:
        print("\n-- Computing hERG labels (cardiac ion channel) --")
        herg_existing = (
            t3db_smiles |
            set(toxcast_final["smiles"].values)
        )
        herg_final = compute_herg_labels(herg_raw, herg_existing)

    # ClinTox: dedup against T3DB + ToxCast + hERG + Tox21
    clintox_final = pd.DataFrame()
    if len(clintox_raw) > 0:
        print("\n-- Computing ClinTox labels (FDA approval + clinical tox) --")
        existing_smiles = (
            set(toxcast_final["smiles"].values) |
            set(tox21_final["smiles"].values) |
            set(herg_final["smiles"].values if len(herg_final) > 0 else []) |
            t3db_smiles
        )
        clintox_final = compute_clintox_labels(clintox_raw, existing_smiles)

    # DILI: dedup against everything (lowest priority)
    dili_final = pd.DataFrame()
    if len(dili_raw) > 0:
        print("\n-- Computing DILI labels (drug-induced liver injury) --")
        dili_existing = (
            t3db_smiles |
            set(toxcast_final["smiles"].values) |
            set(tox21_final["smiles"].values) |
            set(herg_final["smiles"].values if len(herg_final) > 0 else []) |
            set(clintox_final["smiles"].values if len(clintox_final) > 0 else [])
        )
        dili_final = compute_dili_labels(dili_raw, dili_existing)

    # Common molecules: just validate if the raw file exists
    common_path = os.path.join(DATA_DIR, "common_molecules_raw.csv")
    common_final = pd.DataFrame()
    if os.path.exists(common_path):
        print("\n-- Processing common molecules (curated short-IUPAC) --")
        cm_df = pd.read_csv(common_path)
        # Dedup against all SMILES-based datasets (common molecules use IUPAC names)
        n_before = len(cm_df)
        logger.info(f"  Common molecules raw: {n_before} compounds")
        # Keep only iupac_name + is_toxic (drop token_count and any extra columns)
        common_final = cm_df[["iupac_name", "is_toxic"]].copy()
        n_toxic = int(common_final["is_toxic"].sum())
        n_nontoxic = len(common_final) - n_toxic
        logger.info(f"  Toxic: {n_toxic} ({100*n_toxic/len(common_final):.1f}%) | "
                    f"Non-toxic: {n_nontoxic} ({100*n_nontoxic/len(common_final):.1f}%)")
    else:
        logger.info("common_molecules_raw.csv not found -- skipping (will be created by step2)")

    # ── 4. Save final CSVs (WITHOUT IUPAC names) ──
    print("\n-- Saving final datasets (without IUPAC -- step 3 adds them) --")

    toxcast_out = os.path.join(DATA_DIR, "toxcast_final.csv")
    toxcast_final.to_csv(toxcast_out, index=False)
    logger.info(f"  Saved {toxcast_out}: {len(toxcast_final)} compounds")

    tox21_out = os.path.join(DATA_DIR, "tox21_final.csv")
    tox21_final.to_csv(tox21_out, index=False)
    logger.info(f"  Saved {tox21_out}: {len(tox21_final)} compounds")

    if len(clintox_final) > 0:
        clintox_out = os.path.join(DATA_DIR, "clintox_final.csv")
        clintox_final.to_csv(clintox_out, index=False)
        logger.info(f"  Saved {clintox_out}: {len(clintox_final)} compounds")

    if len(herg_final) > 0:
        herg_out = os.path.join(DATA_DIR, "herg_final.csv")
        herg_final.to_csv(herg_out, index=False)
        logger.info(f"  Saved {herg_out}: {len(herg_final)} compounds")

    if len(dili_final) > 0:
        dili_out = os.path.join(DATA_DIR, "dili_final.csv")
        dili_final.to_csv(dili_out, index=False)
        logger.info(f"  Saved {dili_out}: {len(dili_final)} compounds")

    if len(common_final) > 0:
        common_out = os.path.join(DATA_DIR, "common_molecules_final.csv")
        common_final.to_csv(common_out, index=False)
        logger.info(f"  Saved {common_out}: {len(common_final)} compounds")

    # ── Summary ──
    pieces = [toxcast_final, tox21_final]
    if len(clintox_final) > 0: pieces.append(clintox_final)
    if len(herg_final) > 0:    pieces.append(herg_final)
    if len(dili_final) > 0:    pieces.append(dili_final)
    if len(common_final) > 0:  pieces.append(common_final)
    total = sum(len(p) for p in pieces)

    print("\n" + "-" * 60)
    print("  Step 2 complete. Final datasets (binary labels, no IUPAC yet):")
    print(f"    toxcast_final.csv         : {len(toxcast_final):>6} compounds (binary from 617 assays)")
    print(f"    tox21_final.csv           : {len(tox21_final):>6} compounds (binary from 12 assays)")
    if len(clintox_final) > 0:
        print(f"    clintox_final.csv         : {len(clintox_final):>6} compounds (FDA + clinical tox)")
    if len(herg_final) > 0:
        print(f"    herg_final.csv            : {len(herg_final):>6} compounds (cardiac ion channel)")
    if len(dili_final) > 0:
        print(f"    dili_final.csv            : {len(dili_final):>6} compounds (drug-induced liver injury)")
    print(f"    t3db_processed.csv        : (unchanged, all toxic)")
    if len(common_final) > 0:
        print(f"    common_molecules_final.csv: {len(common_final):>6} compounds (curated short-IUPAC)")
    print(f"    Total                     : {total:>6} compounds (+ T3DB)")
    print()

    all_toxic = np.concatenate([p["is_toxic"].values for p in pieces])
    n_toxic    = int(all_toxic.sum())
    n_nontoxic = int((all_toxic == 0).sum())
    print("  Combined binary distribution (excluding T3DB):")
    print(f"    Toxic     : {n_toxic:>6} ({100*n_toxic/len(all_toxic):.1f}%)")
    print(f"    Non-toxic : {n_nontoxic:>6} ({100*n_nontoxic/len(all_toxic):.1f}%)")
    print()
    print("  Next -> run:  python steps/step3_smiles_to_iupac.py")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
