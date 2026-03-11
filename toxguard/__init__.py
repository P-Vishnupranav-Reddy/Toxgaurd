"""ToxGuard: IUPAC-GPT Fine-tuned for Toxicity Prediction with LoRA.

Toxicity prediction from IUPAC names using fine-tuned IUPACGPT
with LoRA adapters on ToxCast + Tox21 + T3DB + ClinTox + hERG + DILI datasets.

Architecture:
  IUPAC Name -> GPT-2 + LoRA -> Binary Head (toxic/non-toxic) + Score Head (regression)
  Severity labels are derived from score at inference time.
"""

__version__ = "0.4.0"

from .data_pipeline import (
    MoleculeDataset,
    ToxCastDataset,
    Tox21Dataset,
    T3DBDataset,
    ClinToxDataset,
    HErgDataset,
    DILIDataset,
    CommonMoleculesDataset,
    ToxicityDataset,
    ToxicityCollator,
    process_local_t3db,
    prepare_combined_dataset,
)
from .tokenizer import ToxGuardTokenizer, get_tokenizer
from .model import (
    ToxGuardModel,
    ToxGuardMultiTaskHead,
    ToxGuardLitModel,
    SEVERITY_LABELS,
    NUM_SEVERITY_CLASSES,
    SEVERITY_THRESHOLDS,
    score_to_severity,
    score_to_severity_label,
)
from .lora import apply_lora_to_model, LoRAConfig
from .inference import ToxGuardPredictor

__all__ = [
    "MoleculeDataset",
    "ToxCastDataset",
    "Tox21Dataset",
    "T3DBDataset",
    "ClinToxDataset",
    "HErgDataset",
    "DILIDataset",
    "CommonMoleculesDataset",
    "ToxicityDataset",
    "ToxicityCollator",
    "process_local_t3db",
    "prepare_combined_dataset",
    "ToxGuardTokenizer",
    "get_tokenizer",
    "ToxGuardModel",
    "ToxGuardMultiTaskHead",
    "ToxGuardLitModel",
    "SEVERITY_LABELS",
    "NUM_SEVERITY_CLASSES",
    "SEVERITY_THRESHOLDS",
    "score_to_severity",
    "score_to_severity_label",
    "apply_lora_to_model",
    "LoRAConfig",
    "ToxGuardPredictor",
]
