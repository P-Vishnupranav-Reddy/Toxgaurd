"""LoRA (Low-Rank Adaptation) implementation for IUPACGPT.

Replaces the Pfeiffer adapter approach in the original IUPACGPT with LoRA,
which inserts low-rank decomposition matrices into the attention layers,
achieving parameter-efficient fine-tuning with better performance.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
"""

import math
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.
    
    Attributes:
        r: Rank of the low-rank decomposition (default: 32)
        alpha: Scaling factor (default: 64.0). Effective scale = alpha / r.
               Best practice: keep alpha = 2 × r for stable scaling.
        dropout: Dropout probability on LoRA path (default: 0.2)
        target_modules: Which linear layers to apply LoRA to.
            For IUPACGPT (GPT-2 style, 8 layers, 256 dim):
              c_attn  — combined Q/K/V projection  (8 modules)
              c_proj  — attention output + MLP out  (16 modules)
              c_fc    — MLP first linear (FFN up)   (8 modules)
            All three targets give ~1.05M LoRA params (~14.0% of 7.1M base)
        merge_weights: Whether to merge LoRA weights into base at inference
        fan_in_fan_out: Set True for Conv1D layers (GPT-2 uses Conv1D, not Linear)
    """
    r: int = 32
    alpha: float = 64.0
    dropout: float = 0.2
    target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "c_fc"]
    )
    merge_weights: bool = False
    fan_in_fan_out: bool = True  # GPT-2 uses Conv1D which is transposed


class LoRALayer(nn.Module):
    """A single LoRA adapter layer that wraps a linear/Conv1D layer.
    
    Implements: h = W₀x + (α/r) · B·A·x
    Where:
        W₀: Original frozen weight matrix (d_out × d_in)
        A: Low-rank down-projection (r × d_in), initialized with Kaiming
        B: Low-rank up-projection (d_out × r), initialized to zero
        α/r: Scaling factor
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        fan_in_fan_out: bool = True,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.fan_in_fan_out = fan_in_fan_out
        
        # Determine dimensions
        weight = original_layer.weight
        if fan_in_fan_out:
            # Conv1D: weight shape is (d_in, d_out) — transposed
            self.d_in = weight.shape[0]
            self.d_out = weight.shape[1]
        else:
            # Regular Linear: weight shape is (d_out, d_in)
            self.d_in = weight.shape[1]
            self.d_out = weight.shape[0]
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(self.d_in, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.d_out))
        
        # Dropout on the LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        result = self.original_layer(x)
        
        if not self.merged:
            # LoRA path: x @ A @ B * scaling
            lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
            result = result + lora_out
        
        return result
    
    def merge(self):
        """Merge LoRA weights into the base model for faster inference."""
        if not self.merged:
            delta_w = (self.lora_A @ self.lora_B * self.scaling)
            if self.fan_in_fan_out:
                self.original_layer.weight.data += delta_w
            else:
                self.original_layer.weight.data += delta_w.T
            self.merged = True
    
    def unmerge(self):
        """Reverse merge for continued training."""
        if self.merged:
            delta_w = (self.lora_A @ self.lora_B * self.scaling)
            if self.fan_in_fan_out:
                self.original_layer.weight.data -= delta_w
            else:
                self.original_layer.weight.data -= delta_w.T
            self.merged = False
    
    @property
    def num_trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig = None,
) -> Tuple[nn.Module, Dict[str, int]]:
    """Apply LoRA adapters to specified modules in a GPT-2 model.
    
    This replaces the Pfeiffer adapter approach from original IUPACGPT with LoRA.
    Instead of inserting bottleneck adapters after FFN layers, LoRA injects
    low-rank matrices into the attention projections — more parameter-efficient
    and generally better-performing.
    
    Args:
        model: The GPT2ForSequenceClassification or GPT2LMHeadModel
        config: LoRA configuration
    
    Returns:
        (modified_model, stats_dict) where stats_dict contains parameter counts
    """
    if config is None:
        config = LoRAConfig()
    
    # First: freeze ALL parameters in the base model
    for param in model.parameters():
        param.requires_grad = False
    
    lora_layers = {}
    num_lora_params = 0
    
    # Walk through all named modules and replace target modules with LoRA versions
    for name, module in model.named_modules():
        for target_name in config.target_modules:
            if name.endswith(target_name):
                # Get the parent module
                parent_name = ".".join(name.split(".")[:-1])
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                
                attr_name = name.split(".")[-1]
                original_layer = getattr(parent, attr_name)
                
                # Create LoRA wrapper
                lora_layer = LoRALayer(
                    original_layer,
                    r=config.r,
                    alpha=config.alpha,
                    dropout=config.dropout,
                    fan_in_fan_out=config.fan_in_fan_out,
                )
                
                # Replace in parent
                setattr(parent, attr_name, lora_layer)
                lora_layers[name] = lora_layer
                num_lora_params += lora_layer.num_trainable_params
                
                logger.debug(f"Applied LoRA to {name}: "
                             f"{lora_layer.d_in}×{config.r} + {config.r}×{lora_layer.d_out} "
                             f"= {lora_layer.num_trainable_params} params")
    
    # Also unfreeze the output/classification head
    _unfreeze_output_head(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "lora_params": num_lora_params,
        "num_lora_layers": len(lora_layers),
        "trainable_pct": 100.0 * trainable_params / total_params,
    }
    
    logger.info(
        f"LoRA applied to {len(lora_layers)} layers | "
        f"Total: {total_params:,} | Trainable: {trainable_params:,} "
        f"({stats['trainable_pct']:.2f}%) | LoRA: {num_lora_params:,}"
    )
    
    return model, stats


def _unfreeze_output_head(model: nn.Module):
    """Unfreeze the classification/toxicity head parameters."""
    # Unfreeze 'output' (ClassificationHead) if it exists
    if hasattr(model, "output"):
        for param in model.output.parameters():
            param.requires_grad = True
    
    # Unfreeze any module named 'toxicity_head' or 'tox_head' or 'severity'
    for name, module in model.named_modules():
        if any(keyword in name.lower() for keyword in ["toxicity", "tox_head", "score_head", "severity"]):
            for param in module.parameters():
                param.requires_grad = True


def save_lora_weights(model: nn.Module, save_path: str):
    """Save only the LoRA adapter weights and trainable head weights.
    
    This produces a compact checkpoint (~100KB-1MB vs ~30MB for full model).
    """
    lora_state_dict = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state_dict[name] = param.data.clone()
    
    torch.save(lora_state_dict, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024) if os.path.exists(save_path) else 0
    logger.info(f"Saved LoRA weights ({len(lora_state_dict)} tensors, {size_mb:.2f} MB) to {save_path}")


def load_lora_weights(model: nn.Module, load_path: str) -> nn.Module:
    """Load saved LoRA weights into a model that already has LoRA applied."""
    lora_state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
    
    model_state = model.state_dict()
    for key, value in lora_state_dict.items():
        if key in model_state:
            model_state[key] = value
        else:
            logger.warning(f"Key {key} not found in model state dict, skipping")
    
    model.load_state_dict(model_state)
    logger.info(f"Loaded LoRA weights from {load_path}")
    return model
