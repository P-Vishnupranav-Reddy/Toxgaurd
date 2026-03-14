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
        self, iupac_names: List[str], return_egnn_vector: bool = True,
        batch_size: int = 32,
    ) -> List[ToxGuardPrediction]:
        """Predict toxicity for multiple molecules with GPU-batched inference.

        Tokenizes all inputs, pads them into batches, and runs forward passes
        on full batches for 10-100× speedup vs sequential prediction.

        Args:
            iupac_names: List of IUPAC names
            return_egnn_vector: Whether to include EGNN vectors
            batch_size: Number of molecules per batch

        Returns:
            List of ToxGuardPrediction objects
        """
        from .model import score_to_severity_label
        from torch.nn.utils.rnn import pad_sequence

        results = []

        # Process in batches
        for start in range(0, len(iupac_names), batch_size):
            batch_names = iupac_names[start:start + batch_size]

            # Tokenize all molecules in this batch
            all_input_ids = []
            for name in batch_names:
                tokenized = self.tokenizer(name)
                ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
                bos = torch.tensor([self.tokenizer._convert_token_to_id(
                    self.tokenizer.unk_token)])
                ids = torch.cat([bos, ids])
                all_input_ids.append(ids)

            # Pad to same length
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
            padded_ids = pad_sequence(all_input_ids, batch_first=True,
                                      padding_value=pad_id).to(self.device)
            attention_mask = (padded_ids != pad_id).long().to(self.device)

            # Single batched forward pass
            with torch.no_grad():
                output = self.model(
                    input_ids=padded_ids,
                    attention_mask=attention_mask,
                    return_hidden=return_egnn_vector,
                )

            # Extract per-molecule results
            binary_probs = torch.sigmoid(output.binary_logits).cpu()
            for i, name in enumerate(batch_names):
                prob = binary_probs[i].item()
                is_toxic = prob >= self.threshold

                egnn_vec = None
                if return_egnn_vector and output.hidden_state is not None:
                    egnn_vec = output.hidden_state[i].cpu().tolist()

                results.append(ToxGuardPrediction(
                    iupac_name=name,
                    is_toxic=is_toxic,
                    toxicity_score=prob,
                    severity_label=score_to_severity_label(prob),
                    confidence=prob,
                    egnn_vector=egnn_vec,
                ))

        return results

    def get_egnn_vectors(self, iupac_names: List[str],
                         batch_size: int = 32) -> torch.Tensor:
        """Extract EGNN input vectors for a batch of molecules.

        Uses batched inference for efficient extraction.

        Args:
            iupac_names: List of IUPAC names
            batch_size: Number of molecules per batch

        Returns:
            (N, 256) tensor of molecular representations for EGNN
        """
        preds = self.predict_batch(iupac_names, return_egnn_vector=True,
                                   batch_size=batch_size)
        vectors = [torch.tensor(p.egnn_vector) for p in preds
                   if p.egnn_vector is not None]
        return torch.stack(vectors) if vectors else torch.empty(0, 256)
