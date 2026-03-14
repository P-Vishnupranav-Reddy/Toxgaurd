"""Temperature scaling for ToxGuard model calibration.

Post-training calibration that learns a single temperature parameter T
on the validation set so that P(toxic) = sigmoid(logit / T) outputs
calibrated probabilities.

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.

Usage:
    scaler = TemperatureScaler()
    scaler.calibrate(model, val_loader, device)
    print(f"Optimal temperature: {scaler.temperature.item():.4f}")

    # Apply during inference
    calibrated_prob = scaler.scale(logit)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for probability calibration.

    Learns a single scalar temperature T that divides logits before sigmoid:
        P_calibrated(toxic) = sigmoid(logit / T)

    When T > 1: softens predictions (less confident)
    When T < 1: sharpens predictions (more confident)
    When T = 1: no change (identity)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw binary_logits from the model (B,)

        Returns:
            Calibrated probabilities (B,)
        """
        return torch.sigmoid(logits / self.temperature)

    def calibrate(
        self,
        model: nn.Module,
        val_loader,
        device: torch.device,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> float:
        """Learn optimal temperature on validation set.

        Minimizes negative log-likelihood (NLL) on the validation set
        with respect to the single temperature parameter.

        Args:
            model: Trained ToxGuardModel (eval mode)
            val_loader: Validation DataLoader
            device: Device to run on
            max_iter: Max optimization iterations
            lr: Learning rate for temperature optimization

        Returns:
            Optimal temperature value
        """
        model.eval()
        self.to(device)

        # Collect all logits and labels from validation set
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["binary_labels"]

                output = model(input_ids=input_ids,
                               attention_mask=attention_mask)
                all_logits.append(output.binary_logits.cpu())
                all_labels.append(labels)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        logger.info(f"Calibrating temperature on {len(logits)} validation samples...")

        # Optimize temperature using L-BFGS
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        logits_device = logits.to(device)
        labels_device = labels.to(device)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_device / self.temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels_device)
            loss.backward()
            return loss

        optimizer.step(closure)

        optimal_temp = self.temperature.item()
        logger.info(f"Optimal temperature: {optimal_temp:.4f}")

        # Report calibration improvement
        with torch.no_grad():
            before_nll = F.binary_cross_entropy_with_logits(
                logits_device, labels_device).item()
            after_nll = F.binary_cross_entropy_with_logits(
                logits_device / self.temperature, labels_device).item()
            logger.info(f"NLL before calibration: {before_nll:.4f}")
            logger.info(f"NLL after calibration:  {after_nll:.4f}")

        return optimal_temp

    def save(self, path: str):
        """Save temperature parameter."""
        torch.save({"temperature": self.temperature.data}, path)
        logger.info(f"Saved temperature ({self.temperature.item():.4f}) to {path}")

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        """Load saved temperature parameter."""
        scaler = cls()
        state = torch.load(path, map_location="cpu", weights_only=True)
        scaler.temperature.data = state["temperature"]
        logger.info(f"Loaded temperature ({scaler.temperature.item():.4f}) from {path}")
        return scaler
