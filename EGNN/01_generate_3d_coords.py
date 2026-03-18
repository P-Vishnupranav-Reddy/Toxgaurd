#!/usr/bin/env python3
"""
==============================================================================
ToxGuard Phase 2 — EGNN Pipeline
Step 01: 3D Conformer Generation from SMILES
==============================================================================

Purpose:
    Generate 3D molecular conformers from SMILES strings using RDKit.
    For each molecule, we:
      1. Parse SMILES → RDKit Mol object
      2. Add explicit hydrogens (important for 3D geometry)
      3. Generate 3D conformer using ETKDGv3
      4. Energy-minimise with MMFF94 (fallback to UFF)
      5. Extract per-atom features and 3D coordinates
      6. Build molecular graph (edge index from covalent bonds)
      7. Cache results as individual .pt files

Input:
    - data/*_final.csv files (columns: smiles, iupac_name, is_toxic)
    - data/t3db_processed.csv (columns include: smiles, is_toxic)

Output:
    - final_egnn_datasets/<dataset_name>/ containing per-molecule .pt files
    - Each .pt file is a dict with:
        node_features : Tensor (N_atoms, F)  — per-atom feature vectors
        coordinates   : Tensor (N_atoms, 3)  — 3D Cartesian coordinates (Å)
        edge_index    : Tensor (2, E)        — covalent bond graph
        edge_attr     : Tensor (E, D_e)      — bond features
        label         : int (0 or 1)
        smiles        : str
        num_atoms     : int

Usage:
    python EGNN/01_generate_3d_coords.py [--data_dir data] [--output_dir final_egnn_datasets]

Author: ToxGuard Team
==============================================================================
"""

import os
import sys
import argparse
import logging
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atom featurization
# ---------------------------------------------------------------------------
# Following best practices from OGB (Open Graph Benchmark) and MoleculeNet
# for molecular property prediction with GNNs.

# Allowable atom feature sets (expanded for better expressiveness)
ATOM_TYPES = [
    1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 26, 29, 30, 33, 34, 35, 53, 0
]  # H, B, C, N, O, F, Si, P, S, Cl, Fe, Cu, Zn, As, Se, Br, I, other

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

FORMAL_CHARGE_RANGE = [-3, -2, -1, 0, 1, 2, 3]
NUM_HS_RANGE = [0, 1, 2, 3, 4]
DEGREE_RANGE = [0, 1, 2, 3, 4, 5, 6]


def one_hot(value, allowable_set):
    """One-hot encode a value against an allowable set, with a catch-all."""
    encoding = [0] * (len(allowable_set) + 1)  # +1 for unknown
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    else:
        encoding[-1] = 1  # unknown category
    return encoding


def get_atom_features(atom):
    """
    Extract a rich feature vector for a single atom.
    
    Features (total ~74 dimensions):
        - Atomic number (one-hot, 18+1 = 19)
        - Degree (one-hot, 7+1 = 8)
        - Formal charge (one-hot, 7+1 = 8)
        - Number of Hs (one-hot, 5+1 = 6)
        - Hybridization (one-hot, 6+1 = 7)
        - Is aromatic (1)
        - Is in ring (1)
        - Ring size membership 3-8 (6)
        - Atomic mass (1, normalized)
        - Van der Waals radius (1, normalized)
        - Electronegativity (1, from Gasteiger)
        - Number of radical electrons (1)
        - Is donor / is acceptor (estimated, 2) 
    
    Returns:
        list[float]: Feature vector
    """
    features = []
    
    # 1. Atomic number one-hot (19 dim)
    features += one_hot(atom.GetAtomicNum(), ATOM_TYPES)
    
    # 2. Degree one-hot (8 dim)
    features += one_hot(atom.GetDegree(), DEGREE_RANGE)
    
    # 3. Formal charge one-hot (8 dim)
    features += one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_RANGE)
    
    # 4. Number of Hs one-hot (6 dim)
    features += one_hot(int(atom.GetTotalNumHs()), NUM_HS_RANGE)
    
    # 5. Hybridization one-hot (7 dim)
    features += one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)
    
    # 6. Is aromatic (1 dim)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    
    # 7. Is in ring (1 dim)
    features.append(1.0 if atom.IsInRing() else 0.0)
    
    # 8. Ring size membership 3-8 (6 dim)
    for ring_size in range(3, 9):
        features.append(1.0 if atom.IsInRingSize(ring_size) else 0.0)
    
    # 9. Normalized atomic mass (1 dim) — divide by 100 for scale
    features.append(atom.GetMass() / 100.0)
    
    # 10. Number of radical electrons (1 dim)
    features.append(float(atom.GetNumRadicalElectrons()))
    
    return features


# ---------------------------------------------------------------------------
# Bond featurization
# ---------------------------------------------------------------------------
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def get_bond_features(bond):
    """
    Extract bond feature vector.
    
    Features (total ~12 dimensions):
        - Bond type (one-hot, 4+1 = 5)
        - Is conjugated (1)
        - Is in ring (1)
        - Stereo (one-hot, 4+1 = 5)
    
    Returns:
        list[float]: Feature vector
    """
    features = []
    features += one_hot(bond.GetBondType(), BOND_TYPES)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    features += one_hot(bond.GetStereo(), STEREO_TYPES)
    return features


# ---------------------------------------------------------------------------
# 3D conformer generation
# ---------------------------------------------------------------------------
def generate_3d_conformer(mol, max_attempts=3):
    """
    Generate a 3D conformer for an RDKit molecule.
    
    Strategy:
        1. Try ETKDGv3 (state-of-the-art distance geometry)
        2. Energy minimise with MMFF94
        3. Fallback to UFF if MMFF94 fails
        4. Retry with random coordinates if embedding fails
    
    Args:
        mol: RDKit Mol object (with explicit Hs)
        max_attempts: Number of retry attempts
    
    Returns:
        mol with 3D conformer, or None if all attempts fail
    """
    for attempt in range(max_attempts):
        try:
            # ETKDGv3 parameters
            params = AllChem.ETKDGv3()
            params.randomSeed = 42 + attempt
            params.numThreads = 0  # use all cores
            params.maxIterations = 500
            
            if attempt > 0:
                # On retry, use random coordinates as starting point
                params.useRandomCoords = True
            
            status = AllChem.EmbedMolecule(mol, params)
            
            if status == -1:
                # Embedding failed, try next attempt
                continue
            
            # Energy minimisation
            try:
                # Try MMFF94 first (more accurate for drug-like molecules)
                mmff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                if mmff_result == -1:
                    raise RuntimeError("MMFF failed")
            except Exception:
                # Fallback to UFF
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
                except Exception:
                    pass  # Keep unoptimised geometry
            
            return mol
            
        except Exception:
            continue
    
    return None


# ---------------------------------------------------------------------------
# Process a single molecule
# ---------------------------------------------------------------------------
def process_molecule(smiles, label, mol_idx):
    """
    Full pipeline for one molecule: SMILES → graph + 3D coords.
    
    Args:
        smiles: SMILES string
        label: Binary toxicity label (0 or 1)
        mol_idx: Index for logging
    
    Returns:
        dict with node_features, coordinates, edge_index, edge_attr, label, smiles, num_atoms
        or None if processing fails
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add explicit hydrogens (critical for accurate 3D geometry)
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformer
    mol_3d = generate_3d_conformer(mol)
    if mol_3d is None:
        return None
    
    # Get conformer
    conf = mol_3d.GetConformer()
    num_atoms = mol_3d.GetNumAtoms()
    
    # Extract 3D coordinates
    coords = np.zeros((num_atoms, 3), dtype=np.float32)
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
    
    # Center coordinates (translation invariance)
    coords -= coords.mean(axis=0, keepdims=True)
    
    # Extract atom features
    atom_features = []
    for atom in mol_3d.GetAtoms():
        atom_features.append(get_atom_features(atom))
    atom_features = np.array(atom_features, dtype=np.float32)
    
    # Build edge index and edge features from covalent bonds
    # Undirected: add both (i→j) and (j→i) for each bond
    edge_src, edge_dst = [], []
    edge_feats = []
    
    for bond in mol_3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_feats.extend([bf, bf])  # same features for both directions
    
    if len(edge_src) == 0:
        # Molecule has no bonds (shouldn't happen for valid molecules)
        return None
    
    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    edge_attr = np.array(edge_feats, dtype=np.float32)
    
    return {
        "node_features": torch.tensor(atom_features, dtype=torch.float32),
        "coordinates": torch.tensor(coords, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "label": int(label),
        "smiles": smiles,
        "num_atoms": num_atoms,
    }


# ---------------------------------------------------------------------------
# Process a dataset
# ---------------------------------------------------------------------------
def process_dataset(csv_path, output_dir, dataset_name, skip_existing=True):
    """
    Process all molecules in a CSV file and save as .pt files.
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Base output directory (final_egnn_datasets/)
        dataset_name: Name for the subdirectory
        skip_existing: If True, skip molecules that already have a .pt file
    
    Returns:
        dict with statistics
    """
    logger.info(f"{'='*60}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"  Source: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    assert "smiles" in df.columns, f"Missing 'smiles' column in {csv_path}"
    assert "is_toxic" in df.columns, f"Missing 'is_toxic' column in {csv_path}"
    
    # Drop rows with missing SMILES
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    
    # Create output directory
    ds_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(ds_output_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(ds_output_dir) if f.endswith(".pt")]

    if skip_existing and len(existing_files) >= len(df):
        logger.info(f"Skipping {dataset_name} (already processed)")
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": len(existing_files),
            "toxic": 0,
            "non_toxic": 0,
            "avg_atoms": 0.0,
        }

    stats = {
        "total": len(df),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "toxic": 0,
        "non_toxic": 0,
        "avg_atoms": 0.0,
    }
    
    atom_counts = []
    failed_indices = []
    
    logger.info(f"  Total molecules: {len(df)}")
    
    for idx in tqdm(range(len(df)), desc=f"  {dataset_name}", ncols=80):
        pt_path = os.path.join(ds_output_dir, f"mol_{idx:06d}.pt")
        
        # Resume logic: check if file already exists
        if skip_existing and os.path.exists(pt_path):
            try:
                # Load existing file to maintain statistics (toxic/non-toxic counts)
                res = torch.load(pt_path, weights_only=False)
                stats["skipped"] += 1
                stats["success"] += 1
                atom_counts.append(res.get("num_atoms", 0))
                if int(res["label"]) == 1:
                    stats["toxic"] += 1
                else:
                    stats["non_toxic"] += 1
                continue
            except Exception:
                # If file is corrupted, re-process it
                pass

        smiles = str(df.loc[idx, "smiles"]).strip()
        label = int(df.loc[idx, "is_toxic"])
        
        result = process_molecule(smiles, label, idx)
        
        if result is not None:
            # Save as .pt file
            torch.save(result, pt_path)
            
            stats["success"] += 1
            atom_counts.append(result["num_atoms"])
            
            if label == 1:
                stats["toxic"] += 1
            else:
                stats["non_toxic"] += 1
        else:
            stats["failed"] += 1
            failed_indices.append(idx)
    
    stats["avg_atoms"] = np.mean(atom_counts) if atom_counts else 0.0
    
    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "source_csv": str(csv_path),
        "stats": stats,
        "failed_indices": failed_indices,
        "node_feature_dim": None,
        "edge_feature_dim": None,
    }
    
    # Get feature dimensions from first successful molecule
    pt_files = sorted(Path(ds_output_dir).glob("mol_*.pt"))
    if pt_files:
        sample = torch.load(pt_files[0], weights_only=False)
        metadata["node_feature_dim"] = sample["node_features"].shape[1]
        metadata["edge_feature_dim"] = sample["edge_attr"].shape[1]
    
    torch.save(metadata, os.path.join(ds_output_dir, "metadata.pt"))
    
    # Log statistics
    logger.info(f"  Results for {dataset_name}:")
    logger.info(f"    Success: {stats['success']}/{stats['total']} "
                f"({100*stats['success']/max(stats['total'],1):.1f}%)")
    if stats["skipped"] > 0:
        logger.info(f"    Skipped (existing): {stats['skipped']}")
    logger.info(f"    Failed:  {stats['failed']}")
    logger.info(f"    Toxic:   {stats['toxic']} | Non-toxic: {stats['non_toxic']}")
    logger.info(f"    Avg atoms per molecule: {stats['avg_atoms']:.1f}")
    if metadata["node_feature_dim"]:
        logger.info(f"    Node feature dim: {metadata['node_feature_dim']}")
        logger.info(f"    Edge feature dim: {metadata['edge_feature_dim']}")
    
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Step 01: Generate 3D conformers and molecular graphs from SMILES"
    )
    parser.add_argument(
    "--skip_datasets",
    nargs="*",
    default=[],
    help="List of dataset names to skip (e.g. toxcast tox21)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing *_final.csv and t3db_processed.csv"
    )
    parser.add_argument(
        "--output_dir", type=str, default="final_egnn_datasets",
        help="Output directory for .pt files"
    )
    args = parser.parse_args()
    skip_datasets = set(args.skip_datasets)
    
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    
    logger.info("=" * 60)
    logger.info("ToxGuard Phase 2 — EGNN 3D Conformer Generation")
    logger.info("=" * 60)
    logger.info(f"Data directory:   {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Define datasets to process
    # Training datasets
    training_datasets = [
        ("toxcast_final.csv", "toxcast"),
        ("tox21_final.csv", "tox21"),
        ("herg_final.csv", "herg"),
        ("dili_final.csv", "dili"),
        ("common_molecules_final.csv", "common_molecules"),
    ]
    
    # External validation datasets
    eval_datasets = [
        ("t3db_processed.csv", "t3db"),
        ("clintox_final.csv", "clintox"),
    ]
    
    all_stats = {}
    
    # Process training datasets
    logger.info("\n>>> Processing TRAINING datasets...")
    for csv_name, ds_name in training_datasets:
        if ds_name in skip_datasets:
            logger.info(f"Skipping dataset: {ds_name}")
            continue

        csv_path = data_dir / csv_name
        if csv_path.exists():
            stats = process_dataset(str(csv_path), str(output_dir), ds_name)
            all_stats[ds_name] = stats
        else:
            logger.warning(f"  {csv_name} not found at {csv_path}, skipping.")
    
    # Process evaluation datasets
    logger.info("\n>>> Processing EVALUATION datasets (T3DB, ClinTox)...")
    for csv_name, ds_name in eval_datasets:
        if ds_name in skip_datasets:
            logger.info(f"Skipping dataset: {ds_name}")
            continue
        csv_path = data_dir / csv_name
        if csv_path.exists():
            stats = process_dataset(str(csv_path), str(output_dir), ds_name)
            all_stats[ds_name] = stats
        else:
            logger.warning(f"  {csv_name} not found at {csv_path}, skipping.")
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    total_success = sum(s["success"] for s in all_stats.values())
    total_failed = sum(s["failed"] for s in all_stats.values())
    
    for ds_name, stats in all_stats.items():
        logger.info(
            f"  {ds_name:25s}: {stats['success']:6d} success | "
            f"{stats['failed']:4d} failed | "
            f"toxic={stats['toxic']} nontoxic={stats['non_toxic']}"
        )
    
    logger.info(f"\n  Total molecules processed: {total_success}")
    logger.info(f"  Total failures: {total_failed}")
    logger.info(f"  Time elapsed: {elapsed:.1f}s")
    logger.info(f"  Output saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
