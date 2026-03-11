#!/usr/bin/env python3
"""
STEP 1 -- Download & Prepare Raw Datasets
==========================================
Downloads all raw datasets needed for ToxGuard training:
  1. ToxCast  -> data/toxcast_raw.csv     (8,597 cpds x 617 assays, MoleculeNet)
  2. Tox21    -> data/tox21_raw.csv       (7,831 cpds x 12 assays,  MoleculeNet)
  3. T3DB     -> data/t3db_processed.csv  (processed from local T3DB bulk CSVs)
  4. ClinTox  -> data/clintox_raw.csv     (1,484 cpds, FDA approval + toxicity)
  5. Common Molecules -> data/common_molecules_raw.csv  (curated ~1000-1200 short IUPAC)

Run from project root:
  python steps/step1_download_data.py
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
T3DB_DIR = "./data/t3db"

# ── Download URLs ──────────────────────────────────────────────────────
TOXCAST_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz"
TOX21_URL   = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
CLINTOX_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"


# ── Download helpers ───────────────────────────────────────────────────

def _download_gz(url: str, output_path: str, name: str) -> str:
    """Download a .csv.gz file from URL and decompress."""
    import urllib.request
    import gzip
    import shutil

    if os.path.exists(output_path):
        df = pd.read_csv(output_path, nrows=5)
        if len(df) > 0:
            logger.info(f"[SKIP] {name} already exists at {output_path}")
            return output_path

    gz_path = output_path + ".gz"
    logger.info(f"Downloading {name}...")
    logger.info(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        df = pd.read_csv(output_path)
        logger.info(f"[OK] {name} downloaded -- {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        # Clean up failed partial downloads
        for f in [gz_path, output_path]:
            if os.path.exists(f):
                os.remove(f)
        raise RuntimeError(f"Download failed for {name}: {e}")
    return output_path


def _download_plain(url: str, output_path: str, name: str) -> str:
    """Download a plain CSV from URL."""
    import urllib.request

    if os.path.exists(output_path):
        df = pd.read_csv(output_path, nrows=5)
        if len(df) > 0:
            logger.info(f"[SKIP] {name} already exists at {output_path}")
            return output_path

    logger.info(f"Downloading {name}...")
    logger.info(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        df = pd.read_csv(output_path)
        logger.info(f"[OK] {name} downloaded -- {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Download failed for {name}: {e}")
    return output_path


# ── Dataset preparation ──────────────────────────────────────────────

def prepare_toxcast_raw():
    """Download raw ToxCast CSV (8,597 cpds x 617 assays)."""
    path = os.path.join(DATA_DIR, "toxcast_raw.csv")
    path = _download_gz(TOXCAST_URL, path, "ToxCast raw")

    df = pd.read_csv(path)
    smiles_col = next((c for c in ["smiles", "SMILES"] if c in df.columns), None)
    if smiles_col is None:
        logger.error("ToxCast raw CSV has no SMILES column!")
        sys.exit(1)

    assay_cols = [c for c in df.columns if c != smiles_col]
    logger.info(f"[OK] toxcast_raw.csv -- {len(df)} compounds, {len(assay_cols)} assays")
    logger.info(f"     Unique SMILES: {df[smiles_col].nunique()}")
    return path


def prepare_tox21_raw():
    """Download raw Tox21 CSV (7,831 cpds x 12 assays)."""
    path = os.path.join(DATA_DIR, "tox21_raw.csv")
    path = _download_gz(TOX21_URL, path, "Tox21 raw")

    df = pd.read_csv(path)
    smiles_col = next((c for c in ["smiles", "SMILES"] if c in df.columns), None)
    if smiles_col is None:
        logger.error("Tox21 raw CSV has no SMILES column!")
        sys.exit(1)

    assay_cols = [c for c in df.columns
                  if c not in [smiles_col, "mol_id", "Unnamed: 0"]]
    logger.info(f"[OK] tox21_raw.csv -- {len(df)} compounds, {len(assay_cols)} assays")
    logger.info(f"     Unique SMILES: {df[smiles_col].nunique()}")
    return path


def prepare_t3db():
    """Process T3DB from local bulk CSVs -> t3db_processed.csv."""
    output_path = os.path.join(DATA_DIR, "t3db_processed.csv")

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        if len(df) >= 200:
            logger.info(f"[OK] t3db_processed.csv -- {len(df)} compounds (already exists)")
            if "toxicity_class" in df.columns:
                logger.info("     Toxicity class distribution:")
                for cls, count in df["toxicity_class"].value_counts().items():
                    logger.info(f"       {cls:<22} {count:>5}")
            return output_path

    # Try to rebuild from raw T3DB CSVs
    required = ["toxin_structures.csv", "all_toxin_data.csv"]
    for fname in required:
        fpath = os.path.join(T3DB_DIR, fname)
        if not os.path.exists(fpath):
            logger.error(f"{fname} not found at {fpath}")
            logger.error("Please download T3DB bulk data from www.t3db.ca")
            sys.exit(1)

    from toxguard.data_pipeline import process_local_t3db
    result_path = process_local_t3db(t3db_dir=T3DB_DIR, save_dir=DATA_DIR)

    df = pd.read_csv(result_path)
    logger.info(f"[OK] t3db_processed.csv -- {len(df)} compounds (rebuilt from raw)")
    return result_path


def prepare_clintox():
    """Download ClinTox CSV (1,484 compounds: FDA approval + clinical trial toxicity).

    Columns:
      smiles       - SMILES string
      FDA_APPROVED - 1 = FDA-approved drug, 0 = not approved
      CT_TOX       - 1 = failed clinical trials due to toxicity, 0 = no toxicity failure
    """
    path = os.path.join(DATA_DIR, "clintox_raw.csv")
    path = _download_gz(CLINTOX_URL, path, "ClinTox raw")

    df = pd.read_csv(path)
    smiles_col = next((c for c in ["smiles", "SMILES"] if c in df.columns), None)
    if smiles_col is None:
        logger.error("ClinTox raw CSV has no SMILES column!")
        sys.exit(1)

    logger.info(f"[OK] clintox_raw.csv -- {len(df)} compounds")
    logger.info(f"     CT_TOX=1 (toxic): {(df['CT_TOX']==1).sum()}")
    logger.info(f"     FDA_APPROVED=1: {(df['FDA_APPROVED']==1).sum()}")
    logger.info(f"     Unique SMILES: {df[smiles_col].nunique()}")
    return path


def prepare_herg():
    """Download hERG cardiotoxicity dataset via PyTDC (Karim et al., 13,445 compounds).

    hERG (human Ether-a-go-go Related Gene) encodes a cardiac ion channel.
    Blocking it causes QT interval prolongation -> potentially fatal arrhythmia.
    This is one of the leading causes of drug withdrawal from market.

    Label: 1 = hERG blocker (cardiotoxic), 0 = non-blocker
    Source: Karim et al. 2021, curated from ChEMBL + literature, binary threshold IC50 <= 10uM

    Output: data/herg_raw.csv  columns: Drug_ID, smiles, is_herg_blocker
    """
    path = os.path.join(DATA_DIR, "herg_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) > 100:
            logger.info(f"[SKIP] herg_raw.csv already exists ({len(df)} compounds)")
            return path

    try:
        from tdc.single_pred import Tox
    except ImportError:
        logger.error("PyTDC not installed. Run:  pip install PyTDC")
        sys.exit(1)

    logger.info("Downloading hERG (Karim) dataset via PyTDC...")
    data = Tox(name='hERG_Karim')
    df = data.get_data()
    df = df.rename(columns={"Drug": "smiles", "Y": "is_herg_blocker"})
    df.to_csv(path, index=False)

    n_block = int((df["is_herg_blocker"] == 1).sum())
    n_safe  = int((df["is_herg_blocker"] == 0).sum())
    logger.info(f"[OK] herg_raw.csv -- {len(df)} compounds | blocker={n_block} non-blocker={n_safe}")
    return path


def prepare_dili():
    """Download DILI (Drug-Induced Liver Injury) dataset via PyTDC (475 compounds).

    DILIrank is an FDA-curated dataset of liver injury risk for approved drugs.
    Liver toxicity is the #1 cause of post-market drug withdrawal and failed trials.

    Label: 1 = DILI concern (liver toxic), 0 = no DILI concern
    Source: Chen et al. 2016 (DILIrank), FDA Liver Toxicity Knowledge Base

    Output: data/dili_raw.csv  columns: Drug_ID, smiles, is_dili
    """
    path = os.path.join(DATA_DIR, "dili_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) > 100:
            logger.info(f"[SKIP] dili_raw.csv already exists ({len(df)} compounds)")
            return path

    try:
        from tdc.single_pred import Tox
    except ImportError:
        logger.error("PyTDC not installed. Run:  pip install PyTDC")
        sys.exit(1)

    logger.info("Downloading DILI dataset via PyTDC...")
    data = Tox(name='DILI')
    df = data.get_data()
    df = df.rename(columns={"Drug": "smiles", "Y": "is_dili"})
    df["is_dili"] = df["is_dili"].astype(int)
    df.to_csv(path, index=False)

    n_dili   = int((df["is_dili"] == 1).sum())
    n_nodili = int((df["is_dili"] == 0).sum())
    logger.info(f"[OK] dili_raw.csv -- {len(df)} compounds | dili={n_dili} no-dili={n_nodili}")
    return path


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  STEP 1 -- Download & Prepare Raw Datasets")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n-- 1/3  ToxCast --")
    toxcast_path = prepare_toxcast_raw()

    print("\n-- 2/3  Tox21 --")
    tox21_path = prepare_tox21_raw()

    print("\n-- 3/6  T3DB --")
    t3db_path = prepare_t3db()

    print("\n-- 4/6  ClinTox --")
    clintox_path = prepare_clintox()

    print("\n-- 5/6  hERG --")
    herg_path = prepare_herg()

    print("\n-- 6/6  DILI --")
    dili_path = prepare_dili()

    # Common molecules (built by a separate script, not downloaded)
    common_path = os.path.join(DATA_DIR, "common_molecules_raw.csv")
    if os.path.exists(common_path):
        n = len(pd.read_csv(common_path))
        print(f"\n  Common Molecules: {common_path} ({n} molecules, already built)")
    else:
        print(f"\n  Common Molecules: not yet built.")
        print(f"    Run:  python steps/build_common_molecules.py")

    print("\n" + "-" * 60)
    print("  Step 1 complete. Raw datasets ready:")
    print(f"    {toxcast_path}")
    print(f"    {tox21_path}")
    print(f"    {t3db_path}")
    print(f"    {clintox_path}")
    print(f"    {herg_path}")
    print(f"    {dili_path}")
    print(f"    {common_path}")
    print("  Next -> run:  python steps/step2_preprocess.py")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
