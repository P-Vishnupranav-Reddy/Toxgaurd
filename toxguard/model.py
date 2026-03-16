"""ToxGuard Model: IUPACGPT adapted for binary toxicity prediction.

The model takes ONLY an IUPAC name as input and predicts toxicity.
Assay data (ToxCast/Tox21) is used only for computing ground-truth
labels during preprocessing — it is NOT a model input.

Architecture outputs:
  1. Binary classification: toxic / non-toxic  (BCE loss)
  2. Toxicity score = P(toxic) = sigmoid(binary_logit)  (no separate head)
  3. EGNN embedding vector: (256,)              (for Phase 2)

Severity labels are derived FROM P(toxic) at inference time only.
  The 0.5 boundary aligns exactly with the binary is_toxic decision:

  P(toxic) < 0.20  -> "Non-toxic"       (very confident non-toxic)
  P(toxic) < 0.50  -> "Unlikely toxic"  (leans non-toxic)
  P(toxic) < 0.65  -> "Likely toxic"    (leans toxic, lower confidence)
  P(toxic) < 0.80  -> "Moderately toxic" (moderately confident toxic)
  P(toxic) >= 0.80 -> "Highly toxic"    (very confident toxic)

Architecture:
  +-----------------------------+
  |  IUPAC Name (tokenized)     |
  +-------------+---------------+
                v
  +-----------------------------+
  |  GPT-2 Transformer (frozen) |  8 layers, 8 heads, 256 dim
   |  + LoRA adapters (trainable)|  rank-32 on c_attn, c_proj, c_fc
  +-------------+---------------+
                v
  +-----------------------------+
  |  Last-token hidden state    |  dim = 256
  +----------+---------+--------+
             |         |
             v         v
          +------+  +------+
          |Binary|  | EGNN |
          |Head  |  | Proj |
          +--+---+  +--+---+
             |          |
             v          v
           {0,1}     (256,)
  toxicity_score = sigmoid(binary_logit)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, AveragePrecision
from transformers import GPT2Config, GPT2Model

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

SEVERITY_LABELS = [
    "Non-toxic",        # 0  P < 0.20  (very confident non-toxic)
    "Unlikely toxic",   # 1  P < 0.50  (leans non-toxic — below binary threshold)
    "Likely toxic",     # 2  P < 0.65  (leans toxic, lower confidence)
    "Moderately toxic", # 3  P < 0.80  (moderately confident toxic)
    "Highly toxic",     # 4  P >= 0.80 (very confident toxic)
]
NUM_SEVERITY_CLASSES = len(SEVERITY_LABELS)

# Thresholds anchored to the 0.5 binary decision boundary:
#   bands 0-1 (P < 0.5) -> model says non-toxic
#   bands 2-4 (P >= 0.5) -> model says toxic
SEVERITY_THRESHOLDS = [0.20, 0.50, 0.65, 0.80]


def score_to_severity(score: float) -> int:
    """Convert continuous toxicity score [0,1] -> severity class {0..4}.

    Used at inference time only — not during training.
    """
    for i, t in enumerate(SEVERITY_THRESHOLDS):
        if score < t:
            return i
    return NUM_SEVERITY_CLASSES - 1


def score_to_severity_label(score: float) -> str:
    """Convert score to human-readable severity label."""
    return SEVERITY_LABELS[score_to_severity(score)]


# ──────────────────────────────────────────────────────────────────────
# Output container
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ToxGuardOutput:
    """Output from ToxGuard model."""
    loss: Optional[torch.Tensor] = None
    binary_logits: Optional[torch.Tensor] = None        # (B,) raw logit
    toxicity_score: Optional[torch.Tensor] = None       # (B,) P(toxic) = sigmoid(binary_logit)
    hidden_state: Optional[torch.Tensor] = None         # (B, 256) for EGNN Phase 2
    target_binary: Optional[torch.Tensor] = None        # (B,) ground truth {0, 1}


# ──────────────────────────────────────────────────────────────────────
# Head modules
# ──────────────────────────────────────────────────────────────────────

class ToxicityHead(nn.Module):
    """Two-layer FFN classification/regression head.

    Architecture: Linear(d_in, d_mid) -> GELU -> Dropout -> Linear(d_mid, d_out)
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(intermediate_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.out_proj(x)


class ToxGuardMultiTaskHead(nn.Module):
    """Toxicity prediction head — binary classification + EGNN projection.

    Outputs:
      1. Binary classification logit (toxic vs non-toxic)
      2. Toxicity score = P(toxic) = sigmoid(binary_logit)  — no separate head
      3. Raw hidden state vector for Phase 2 EGNN input
    """

    def __init__(self, hidden_size: int = 256,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        intermediate = hidden_size // 2  # 128 for 256-dim model

        # Head 1: Binary toxic/non-toxic classification
        self.binary_head = ToxicityHead(
            hidden_size, intermediate, 1, dropout
        )

        # Projection for EGNN input (Phase 2)
        self.egnn_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_state: (B, hidden_size) last-token representation

        Returns:
            dict with binary_logits, toxicity_score (= sigmoid(binary_logits)), egnn_vector
        """
        binary_logits = self.binary_head(hidden_state)        # (B, 1)

        # Toxicity score is simply P(toxic) — no separate regression head
        tox_score = torch.sigmoid(binary_logits)              # (B, 1)

        # EGNN feature vector
        egnn_vector = self.egnn_projection(hidden_state)      # (B, 256)

        return {
            "binary_logits": binary_logits.squeeze(-1),       # (B,)
            "toxicity_score": tox_score.squeeze(-1),          # (B,)
            "egnn_vector": egnn_vector,                       # (B, 256)
        }


# ──────────────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────────────

class ToxGuardModel(nn.Module):
    """ToxGuard: IUPACGPT backbone + LoRA + Binary Toxicity Head.

    This wraps a pre-trained GPT2Model (from IUPACGPT checkpoints) with
    a binary toxicity head (toxic/non-toxic) + score regression head.
    LoRA adapters are applied separately via apply_lora_to_model().
    """

    def __init__(self, config: GPT2Config, **kwargs):
        super().__init__()
        self.config = config

        # GPT-2 transformer backbone (loaded from IUPACGPT checkpoint)
        self.transformer = GPT2Model(config)

        # Toxicity prediction head (binary + score + EGNN projection)
        self.toxicity_head = ToxGuardMultiTaskHead(
            hidden_size=config.n_embd,
            dropout=config.embd_pdrop,
        )

        # Loss weight
        self.binary_loss_weight = 1.0

        # Label smoothing for BCE (Bug fix #4)
        self.label_smoothing = 0.1

        # Class imbalance compensation
        self.pos_weight: Optional[torch.Tensor] = None  # set via set_class_weights()
        self.use_focal_loss = False
        self.focal_gamma = 2.0
        self.focal_alpha = 0.45  # calibrated for ~54% toxic training balance

    @classmethod
    def from_pretrained_iupacgpt(
        cls, checkpoint_dir: str, **kwargs
    ) -> "ToxGuardModel":
        """Load from an IUPACGPT checkpoint.

        Args:
            checkpoint_dir: Path to IUPACGPT checkpoint (contains config.json)
        """
        config = GPT2Config.from_pretrained(checkpoint_dir)
        model = cls(config)

        # Load pretrained transformer weights
        from transformers import GPT2LMHeadModel
        pretrained = GPT2LMHeadModel.from_pretrained(checkpoint_dir, config=config)

        # Copy only the transformer (not the LM head)
        model.transformer.load_state_dict(pretrained.transformer.state_dict())

        logger.info(f"Loaded IUPACGPT transformer from {checkpoint_dir}")
        logger.info(f"Config: n_layer={config.n_layer}, n_head={config.n_head}, "
                     f"n_embd={config.n_embd}, vocab_size={config.vocab_size}")

        del pretrained  # Free memory
        return model

    def _get_last_token_hidden(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Extract the hidden state at the last non-padding token position."""
        batch_size = input_ids.shape[0]

        if self.config.pad_token_id is not None:
            sequence_lengths = torch.ne(
                input_ids, self.config.pad_token_id
            ).sum(-1) - 1
        else:
            sequence_lengths = torch.full(
                (batch_size,), input_ids.shape[1] - 1,
                dtype=torch.long, device=input_ids.device
            )

        last_hidden = hidden_states[range(batch_size), sequence_lengths]
        return last_hidden  # (B, n_embd)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        binary_labels: Optional[torch.Tensor] = None,
        return_hidden: bool = True,
        **kwargs,
    ) -> ToxGuardOutput:
        """
        Args:
            input_ids: (B, L) tokenized IUPAC name
            attention_mask: (B, L)
            binary_labels: (B,) binary toxic/non-toxic {0, 1}
            return_hidden: whether to return hidden state for EGNN
        """
        # Forward through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = transformer_outputs.last_hidden_state  # (B, L, 256)

        # Extract last-token representation
        last_hidden = self._get_last_token_hidden(hidden_states, input_ids)  # (B, 256)

        # Multi-task prediction heads
        head_outputs = self.toxicity_head(last_hidden)

        # Compute loss
        loss = None
        if binary_labels is not None:
            loss = self._compute_loss(head_outputs, binary_labels)

        return ToxGuardOutput(
            loss=loss,
            binary_logits=head_outputs["binary_logits"],
            toxicity_score=head_outputs["toxicity_score"],
            hidden_state=head_outputs["egnn_vector"] if return_hidden else None,
            target_binary=binary_labels,
        )

    def set_class_weights(self, n_positive: int, n_negative: int):
        """Set pos_weight for class-weighted BCE from dataset statistics.

        pos_weight = n_negative / n_positive compensates for class imbalance
        by penalizing false negatives (missing toxic compounds) more.
        """
        weight = n_negative / max(n_positive, 1)
        self.pos_weight = torch.tensor([weight])
        logger.info(f"Class weights set: pos_weight={weight:.4f} "
                    f"(n_pos={n_positive}, n_neg={n_negative})")

    def _compute_loss(
        self,
        head_outputs: Dict[str, torch.Tensor],
        binary_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss with label smoothing + optional class weighting / focal loss.

        Supports three modes:
          1. Standard BCE (default)
          2. Class-weighted BCE (when pos_weight is set via set_class_weights)
          3. Focal loss (when use_focal_loss=True) — down-weights easy examples

        Label smoothing: target 1 -> (1 - eps/2), target 0 -> eps/2
        """
        eps = self.label_smoothing
        smoothed_targets = binary_labels.float() * (1.0 - eps) + eps / 2.0
        logits = head_outputs["binary_logits"]

        if self.use_focal_loss:
            # Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            # Reduces contribution of easy-to-classify examples
            bce = F.binary_cross_entropy_with_logits(
                logits, smoothed_targets, reduction="none",
            )
            p = torch.sigmoid(logits)
            p_t = p * smoothed_targets + (1 - p) * (1 - smoothed_targets)
            focal_weight = (1 - p_t) ** self.focal_gamma

            # Alpha balancing: alpha for positives, (1-alpha) for negatives
            alpha_t = (self.focal_alpha * smoothed_targets
                       + (1 - self.focal_alpha) * (1 - smoothed_targets))

            binary_loss = (alpha_t * focal_weight * bce).mean()
        else:
            # Standard or class-weighted BCE
            pw = None
            if self.pos_weight is not None:
                pw = self.pos_weight.to(logits.device)
            binary_loss = F.binary_cross_entropy_with_logits(
                logits, smoothed_targets, pos_weight=pw,
            )

        return self.binary_loss_weight * binary_loss

    def get_egnn_input_vector(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract the representation vector for EGNN input (Phase 2).

        Returns:
            (B, 256) tensor — the molecular representation from the transformer,
            projected through egnn_projection layer.
        """
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden=True,
            )
        return output.hidden_state


# ──────────────────────────────────────────────────────────────────────
# PyTorch Lightning Training Module
# ──────────────────────────────────────────────────────────────────────

class ToxGuardLitModel(pl.LightningModule):
    """PyTorch Lightning wrapper for ToxGuard training.

    Handles:
      - Training with loss (binary BCE, label smoothing ε=0.1)
      - Validation/test with AUC-ROC, AUC-PRC, accuracy metrics
      - Learning rate scheduling (cosine annealing)
      - Gradient clipping
    """

    def __init__(
        self,
        model: ToxGuardModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 10000,
        scheduler_type: str = "cosine",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        # Binary classification metrics
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.train_auprc = AveragePrecision(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.test_auprc = AveragePrecision(task="binary")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _shared_step(self, batch, batch_idx, auroc_metric, auprc_metric, prefix):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            binary_labels=batch["binary_labels"],
        )

        # Log loss
        self.log(f"{prefix}_loss", outputs.loss, prog_bar=True, on_step=(prefix == "train"),
                 on_epoch=True, batch_size=batch["input_ids"].shape[0])

        # Binary probability from logit
        binary_prob = torch.sigmoid(outputs.binary_logits)  # (B,)
        binary_targets = outputs.target_binary.long()

        # Update AUC metrics
        auroc_metric.update(binary_prob, binary_targets)
        auprc_metric.update(binary_prob, binary_targets)

        # Log binary accuracy
        pred_binary = (binary_prob >= 0.5).long()
        binary_acc = (pred_binary == binary_targets).float().mean()
        self.log(f"{prefix}_acc", binary_acc, prog_bar=(prefix != "train"),
                 on_epoch=True, batch_size=batch["input_ids"].shape[0])

        return {"loss": outputs.loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.train_auroc, self.train_auprc, "train")

    def on_train_epoch_end(self):
        try:
            self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
            self.log("train_auprc", self.train_auprc.compute(), prog_bar=False)
        except (ValueError, RuntimeError):
            self.log("train_auroc", 0.0, prog_bar=True)
            self.log("train_auprc", 0.0, prog_bar=False)
        self.train_auroc.reset()
        self.train_auprc.reset()

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.val_auroc, self.val_auprc, "val")

    def on_validation_epoch_end(self):
        try:
            self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
            self.log("val_auprc", self.val_auprc.compute(), prog_bar=True)
        except (ValueError, RuntimeError):
            self.log("val_auroc", 0.0, prog_bar=True)
            self.log("val_auprc", 0.0, prog_bar=True)
        self.val_auroc.reset()
        self.val_auprc.reset()

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.test_auroc, self.test_auprc, "test")

    def on_test_epoch_end(self):
        try:
            self.log("test_auroc", self.test_auroc.compute(), prog_bar=True)
            self.log("test_auprc", self.test_auprc.compute(), prog_bar=True)
        except (ValueError, RuntimeError):
            self.log("test_auroc", 0.0, prog_bar=True)
            self.log("test_auprc", 0.0, prog_bar=True)
        self.test_auroc.reset()
        self.test_auprc.reset()

    def configure_optimizers(self):
        # Only optimize trainable parameters (LoRA + heads)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if self.hparams.scheduler_type == "none":
            return optimizer

        warmup_steps = max(self.hparams.warmup_steps, 1)
        main_steps = max(self.hparams.max_steps - warmup_steps, 1)

        # Phase 1: Linear warmup from 0 -> peak LR
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Phase 2: Cosine or exponential decay after warmup
        if self.hparams.scheduler_type == "cosine":
            decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=main_steps,
                eta_min=self.hparams.learning_rate * 0.01,
            )
        elif self.hparams.scheduler_type == "exponential":
            decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.99
            )
        else:
            return optimizer

        # Combine: warmup first, then decay
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        # Override to avoid deprecated epoch parameter warning from PyTorch Lightning
        scheduler.step()
