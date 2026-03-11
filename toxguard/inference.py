"""Unified inference engine for ToxGuard.

Provides a single-call interface:
    predictor = ToxGuardPredictor.from_checkpoint("path/to/checkpoint")
    result = predictor.predict("formonitrile")
    print(result.summary())
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ToxGuardPrediction:
    """Complete prediction result from ToxGuard."""
    iupac_name: str
    is_toxic: bool                                  # binary classification
    toxicity_score: float                           # P(toxic) = sigmoid(binary_logit)
    severity_label: str                             # derived from P(toxic): "Non-toxic" .. "Highly toxic"
    confidence: float                               # same as toxicity_score = P(toxic)
    egnn_vector: Optional[List[float]]              # 256-dim vector for Phase 2

    def summary(self) -> str:
        """One-line summary."""
        toxic_str = "TOXIC" if self.is_toxic else "Non-toxic"
        return (f"{self.iupac_name}: {toxic_str} "
                f"(P(toxic)={self.toxicity_score:.3f}, severity={self.severity_label})")


class ToxGuardPredictor:
    """High-level predictor for ToxGuard — model inference over IUPAC names.

    Usage:
        predictor = ToxGuardPredictor(model, tokenizer)
        result = predictor.predict("formonitrile")
        print(result.summary())

        # Batch prediction
        results = predictor.predict_batch(["formonitrile", "oxidane", "nitrobenzene"])
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        """
        Args:
            model: ToxGuardModel instance
            tokenizer: ToxGuardTokenizer instance
            device: 'cpu' or 'cuda'
            threshold: Binary classification threshold (default 0.5)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        lora_weights_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",
    ) -> "ToxGuardPredictor":
        """Load a complete ToxGuard predictor from saved checkpoint.

        Args:
            checkpoint_dir: Path to IUPACGPT base checkpoint
            lora_weights_path: Path to saved LoRA adapter weights
            tokenizer_path: Path to tokenizer (iupac_spm.model or .pt)
            device: 'cpu' or 'cuda'
        """
        from .model import ToxGuardModel
        from .tokenizer import get_tokenizer
        from .lora import apply_lora_to_model, load_lora_weights, LoRAConfig

        # Load tokenizer
        tokenizer = get_tokenizer(
            vocab_path=tokenizer_path,
            iupacgpt_dir=checkpoint_dir,
        )

        # Load model
        model = ToxGuardModel.from_pretrained_iupacgpt(checkpoint_dir)
        model.config.pad_token_id = tokenizer.pad_token_id

        # Apply LoRA structure
        model, _ = apply_lora_to_model(model, LoRAConfig())

        # Load trained LoRA weights
        if lora_weights_path:
            model = load_lora_weights(model, lora_weights_path)

        return cls(model, tokenizer, device)

    def predict(
        self,
        iupac_name: str,
        return_egnn_vector: bool = True,
    ) -> ToxGuardPrediction:
        """Predict toxicity for a single molecule by IUPAC name.

        Args:
            iupac_name: IUPAC name of the molecule (e.g., "formonitrile")
            return_egnn_vector: Whether to include the 256-dim EGNN input vector

        Returns:
            ToxGuardPrediction with binary label, score, and severity
        """
        # Tokenize
        tokenized = self.tokenizer(iupac_name)
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)

        # Prepend BOS token
        bos = torch.tensor([self.tokenizer._convert_token_to_id(self.tokenizer.unk_token)])
        input_ids = torch.cat([bos, input_ids]).unsqueeze(0).to(self.device)

        attention_mask = torch.ones_like(input_ids).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden=return_egnn_vector,
            )

        # Binary prediction from binary_logits
        binary_prob = torch.sigmoid(output.binary_logits[0]).item()
        is_toxic = binary_prob >= self.threshold

        tox_score = binary_prob
        confidence = binary_prob

        from .model import score_to_severity_label
        severity_label = score_to_severity_label(tox_score)

        # EGNN vector
        egnn_vec = None
        if return_egnn_vector and output.hidden_state is not None:
            egnn_vec = output.hidden_state[0].cpu().tolist()

        return ToxGuardPrediction(
            iupac_name=iupac_name,
            is_toxic=is_toxic,
            toxicity_score=tox_score,
            severity_label=severity_label,
            confidence=confidence,
            egnn_vector=egnn_vec,
        )

    def predict_batch(
        self, iupac_names: List[str], return_egnn_vector: bool = True
    ) -> List[ToxGuardPrediction]:
        """Predict toxicity for multiple molecules.

        Args:
            iupac_names: List of IUPAC names
            return_egnn_vector: Whether to include EGNN vectors

        Returns:
            List of ToxGuardPrediction objects
        """
        return [
            self.predict(name, return_egnn_vector=return_egnn_vector)
            for name in iupac_names
        ]

    def get_egnn_vectors(self, iupac_names: List[str]) -> torch.Tensor:
        """Extract EGNN input vectors for a batch of molecules.

        This is the primary interface for Phase 2 (EGNN integration).

        Args:
            iupac_names: List of IUPAC names

        Returns:
            (N, 256) tensor of molecular representations for EGNN
        """
        vectors = []
        for name in iupac_names:
            pred = self.predict(name, return_egnn_vector=True)
            if pred.egnn_vector is not None:
                vectors.append(torch.tensor(pred.egnn_vector))

        return torch.stack(vectors) if vectors else torch.empty(0, 256)
