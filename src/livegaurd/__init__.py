"""
LiveGaurd: Runtime intervention for LLMs using Sparse Autoencoders.

Suppress sycophancy and other behaviors by clamping SAE features at inference.

Quick Start (Colab/RunPod):
    >>> from livegaurd import load_model_and_sae, GuillotineHook, register_hook
    >>> model, tokenizer, sae, layer = load_model_and_sae("gemma-2-2b")
    >>> hook = GuillotineHook(sae, target_features=[100, 200, 300])
    >>> handle = register_hook(model, layer, hook)
"""

__version__ = "0.1.0"

from livegaurd.sae_loader import (
    load_sae_from_saelens,
    load_gemma_sae,
    load_deepseek_distill_sae,
    load_model_and_sae,
    get_model_config,
    SUPPORTED_MODELS,
)

from livegaurd.hooks import (
    GuillotineHook,
    InterventionStats,
    create_intervention_hook,
    register_hook,
)

__all__ = [
    # SAE loading
    "load_sae_from_saelens",
    "load_gemma_sae",
    "load_deepseek_distill_sae",
    "load_model_and_sae",
    "get_model_config",
    "SUPPORTED_MODELS",
    # Hooks
    "GuillotineHook",
    "InterventionStats",
    "create_intervention_hook",
    "register_hook",
]
