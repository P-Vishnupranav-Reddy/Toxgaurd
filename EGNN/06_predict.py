#!/usr/bin/env python3
"""
==============================================================================
ToxGuard Phase 2 — EGNN Pipeline
Step 06: Inference — Predict Toxicity from SMILES
==============================================================================

Purpose:
    Single-molecule or batch prediction of toxicity using the trained
    ToxEGNN model. Accepts SMILES strings and outputs:
      - Toxicity probability (0.0 to 1.0)
      - Binary prediction (Toxic / Non-toxic)
      - Severity label (Non-toxic, Unlikely toxic, Likely toxic,
        Moderately toxic, Highly toxic)
      - Confidence level

Input:
    - Trained model checkpoint (outputs_egnn/<run_id>/best_model.pt)
    - SMILES string(s) via CLI or file

Usage:
    # Single molecule
    python EGNN/06_predict.py --model outputs_egnn/<run_id>/best_model.pt \
                              --smiles "CCO"

    # Batch from file (one SMILES per line)
    python EGNN/06_predict.py --model outputs_egnn/<run_id>/best_model.pt \
                              --input_file molecules.txt \
                              --output_file predictions.csv

Author: ToxGuard Team
==============================================================================
"""

import os
import sys
import argparse
import logging
import csv
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
# Severity labels (matching Phase 1)
# ---------------------------------------------------------------------------
def get_severity_label(prob):
    """Map probability to severity label."""
    if prob < 0.20:
        return "Non-toxic"
    elif prob < 0.50:
        return "Unlikely toxic"
    elif prob < 0.65:
        return "Likely toxic"
    elif prob < 0.80:
        return "Moderately toxic"
    else:
        return "Highly toxic"


def get_confidence(prob):
    """Confidence = distance from decision boundary (0.5)."""
    return abs(prob - 0.5) * 2  # ranges from 0 (uncertain) to 1 (confident)


# ---------------------------------------------------------------------------
# SMILES to PyG Data
# ---------------------------------------------------------------------------
def smiles_to_pyg(smiles):
    """
    Convert a SMILES string to a PyG Data object.
    Uses the same featurisation as Step 01.
    """
    # Import from step 01
    from importlib import import_module
    step01 = import_module("01_generate_3d_coords")
    
    result = step01.process_molecule(smiles, label=0, mol_idx=0)
    if result is None:
        return None
    
    data = Data(
        x=result["node_features"],
        pos=result["coordinates"],
        edge_index=result["edge_index"],
        edge_attr=result["edge_attr"],
        y=torch.tensor([0.0]),  # dummy label
        smiles=smiles,
    )
    return data


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------
class ToxEGNNPredictor:
    """Toxicity predictor using trained ToxEGNN model."""
    
    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: Path to best_model.pt checkpoint
            device: torch device (auto-detected if None)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, weights_only=False,
                                map_location=device)
        config = checkpoint["config"]
        
        # Create model
        egnn_module = import_module("03_egnn_model")
        self.model = egnn_module.create_model(
            node_feat_dim=config["data"]["node_feat_dim"],
            edge_feat_dim=config["data"]["edge_feat_dim"],
            config=config["model"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"  Epoch: {checkpoint['epoch']}, "
                    f"Val AUROC: {checkpoint['val_auroc']:.4f}")
    
    @torch.no_grad()
    def predict(self, smiles):
        """
        Predict toxicity for a single SMILES string.
        
        Args:
            smiles: SMILES string
        
        Returns:
            dict with: probability, prediction, severity, confidence
            or None if the molecule could not be processed
        """
        data = smiles_to_pyg(smiles)
        if data is None:
            logger.warning(f"Could not process SMILES: {smiles}")
            return None
        
        batch = Batch.from_data_list([data]).to(self.device)
        logit = self.model(batch)
        prob = torch.sigmoid(logit).item()
        
        return {
            "smiles": smiles,
            "probability": round(prob, 4),
            "prediction": "Toxic" if prob >= 0.5 else "Non-toxic",
            "severity": get_severity_label(prob),
            "confidence": round(get_confidence(prob), 4),
        }
    
    @torch.no_grad()
    def predict_batch(self, smiles_list):
        """
        Predict toxicity for multiple SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
        
        Returns:
            List of prediction dicts
        """
        results = []
        valid_data = []
        valid_indices = []
        
        # Process molecules
        for i, smiles in enumerate(smiles_list):
            data = smiles_to_pyg(smiles)
            if data is not None:
                valid_data.append(data)
                valid_indices.append(i)
            else:
                results.append({
                    "smiles": smiles,
                    "probability": None,
                    "prediction": "ERROR",
                    "severity": "Could not process molecule",
                    "confidence": 0.0,
                })
        
        # Batch predict
        if valid_data:
            batch = Batch.from_data_list(valid_data).to(self.device)
            logits = self.model(batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            valid_results = []
            for j, prob in enumerate(probs):
                prob = float(prob)
                valid_results.append({
                    "smiles": smiles_list[valid_indices[j]],
                    "probability": round(prob, 4),
                    "prediction": "Toxic" if prob >= 0.5 else "Non-toxic",
                    "severity": get_severity_label(prob),
                    "confidence": round(get_confidence(prob), 4),
                })
            
            # Merge results in original order
            final_results = [None] * len(smiles_list)
            error_idx = 0
            valid_idx = 0
            for i in range(len(smiles_list)):
                if i in valid_indices:
                    final_results[i] = valid_results[valid_idx]
                    valid_idx += 1
                else:
                    final_results[i] = results[error_idx]
                    error_idx += 1
            
            return final_results
        
        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Step 06: ToxEGNN Inference — Predict toxicity from SMILES"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--smiles", type=str, default=None,
                        help="Single SMILES string to predict")
    parser.add_argument("--input_file", type=str, default=None,
                        help="File with one SMILES per line")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output CSV file for batch predictions")
    args = parser.parse_args()
    
    if args.smiles is None and args.input_file is None:
        parser.error("Provide either --smiles or --input_file")
    
    # Load model
    predictor = ToxEGNNPredictor(args.model)
    
    if args.smiles:
        # Single prediction
        result = predictor.predict(args.smiles)
        
        if result is None:
            print("ERROR: Could not process molecule")
            return
        
        print("\n" + "=" * 50)
        print("ToxEGNN Prediction")
        print("=" * 50)
        print(f"  SMILES:      {result['smiles']}")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Prediction:  {result['prediction']}")
        print(f"  Severity:    {result['severity']}")
        print(f"  Confidence:  {result['confidence']:.4f}")
        print("=" * 50)
    
    elif args.input_file:
        # Batch prediction
        with open(args.input_file, "r") as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(smiles_list)} molecules...")
        results = predictor.predict_batch(smiles_list)
        
        # Output
        if args.output_file:
            with open(args.output_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["smiles", "probability", "prediction",
                               "severity", "confidence"],
                )
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            logger.info(f"Predictions saved to {args.output_file}")
        else:
            print(f"\n{'SMILES':<50} {'Prob':>6} {'Prediction':<12} {'Severity':<20}")
            print("-" * 95)
            for r in results:
                prob_str = f"{r['probability']:.4f}" if r['probability'] else "ERROR"
                print(f"{r['smiles']:<50} {prob_str:>6} "
                      f"{r['prediction']:<12} {r['severity']:<20}")


if __name__ == "__main__":
    from importlib import import_module
    main()
