"""
The Guillotine Hook: Runtime intervention on SAE features.

Works with SAELens pretrained SAEs (Gemma Scope, etc.)

Core formula (error term restoration):
    x_new = decode(clamp(encode(x))) + (x - decode(encode(x)))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)


@dataclass
class InterventionStats:
    """Statistics from intervention execution."""
    features_clamped: int = 0
    activations_modified: int = 0
    latency_ms: float = 0.0
    triggered_features: list[int] = field(default_factory=list)


class GuillotineHook:
    """
    SAE-based feature suppression with error term restoration.

    Works with any SAE that has encode() and decode() methods.

    x_new = decode(clamp(encode(x))) + (x - decode(encode(x)))

    Args:
        sae: SAE with encode/decode methods (SAELens or custom)
        target_features: Feature indices to suppress
        mode: "hard" (zero), "soft" (scale by 0.5)
        threshold: Only intervene if activation > threshold

    Example:
        >>> hook = GuillotineHook(sae, target_features=[100, 200, 300])
        >>> handle = model.model.layers[12].register_forward_hook(hook)
    """

    def __init__(
        self,
        sae: Any,
        target_features: list[int],
        mode: Literal["hard", "soft"] = "hard",
        threshold: float = 0.0,
    ):
        self.sae = sae
        self.target_features = target_features
        self.mode = mode
        self.threshold = threshold
        self.enabled = True
        self.stats = InterventionStats()

    def __call__(self, module, input, output):
        """Hook function for register_forward_hook."""
        if not self.enabled:
            return output

        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Apply intervention
        modified = self._intervene(hidden)

        # Return modified output
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def _intervene(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the guillotine intervention."""
        start_time = time.perf_counter()

        original_dtype = hidden.dtype
        device = hidden.device

        with torch.no_grad():
            # Convert to float32 for SAE operations
            x = hidden.float()

            # Step 1: Encode to get features
            features = self.sae.encode(x)

            # Step 2: Compute error term BEFORE modification
            recon_before = self.sae.decode(features)
            error_term = x - recon_before

            # Step 3: Clamp target features
            features_clamped = 0
            activations_modified = 0
            triggered = []

            for feat_idx in self.target_features:
                if feat_idx < 0 or feat_idx >= features.shape[-1]:
                    continue

                feat_acts = features[:, :, feat_idx]
                mask = feat_acts > self.threshold

                if mask.any():
                    features_clamped += 1
                    activations_modified += mask.sum().item()
                    triggered.append(feat_idx)

                    if self.mode == "hard":
                        features[:, :, feat_idx] = torch.where(
                            mask,
                            torch.zeros_like(feat_acts),
                            feat_acts,
                        )
                    elif self.mode == "soft":
                        features[:, :, feat_idx] = torch.where(
                            mask,
                            feat_acts * 0.5,
                            feat_acts,
                        )

            # Step 4: Decode + error term restoration
            recon_after = self.sae.decode(features)
            modified = recon_after + error_term

            # Convert back to original dtype
            modified = modified.to(original_dtype)

        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.features_clamped = features_clamped
        self.stats.activations_modified = activations_modified
        self.stats.latency_ms = elapsed_ms
        self.stats.triggered_features = triggered

        return modified

    def reset_stats(self):
        """Reset intervention statistics."""
        self.stats = InterventionStats()


def create_intervention_hook(
    sae: Any,
    target_features: list[int],
    mode: Literal["hard", "soft"] = "hard",
    threshold: float = 0.0,
) -> GuillotineHook:
    """
    Create a configured intervention hook.

    Args:
        sae: SAE with encode/decode methods
        target_features: Feature indices to suppress
        mode: "hard" or "soft"
        threshold: Activation threshold

    Returns:
        Configured GuillotineHook
    """
    return GuillotineHook(
        sae=sae,
        target_features=target_features,
        mode=mode,
        threshold=threshold,
    )


def register_hook(model, layer_idx: int, hook: GuillotineHook):
    """
    Register hook on a model layer.

    Args:
        model: HuggingFace model
        layer_idx: Which layer to hook
        hook: GuillotineHook instance

    Returns:
        Hook handle (call .remove() to unregister)
    """
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)
    return handle
