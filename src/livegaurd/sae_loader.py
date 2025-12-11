"""
SAE Loader for LiveGaurd.

Supports two modes:
1. SAELens pretrained SAEs (Gemma-2B, DeepSeek-R1-Distill, etc.)
2. Custom SAEs (Goodfire format)

Primary target: Gemma-2-2B with Google's Gemma Scope SAEs.
"""

from __future__ import annotations

import torch
from typing import Literal, Any


def load_sae_from_saelens(
    release: str,
    sae_id: str,
    device: str = "cuda",
) -> tuple[Any, dict, Any]:
    """
    Load a pretrained SAE from SAELens.

    Args:
        release: SAELens release name (e.g., "gemma-scope-2b-pt-res")
        sae_id: Specific SAE ID (e.g., "layer_12/width_16k/average_l0_71")
        device: Target device

    Returns:
        Tuple of (sae, cfg_dict, sparsity)

    Example:
        >>> sae, cfg, _ = load_sae_from_saelens(
        ...     "gemma-scope-2b-pt-res",
        ...     "layer_12/width_16k/average_l0_71"
        ... )
    """
    from sae_lens import SAE

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )

    return sae, cfg_dict, sparsity


def load_gemma_sae(
    layer: int = 12,
    width: str = "16k",
    device: str = "cuda",
) -> tuple[Any, dict]:
    """
    Load Gemma Scope SAE for Gemma-2-2B.

    Args:
        layer: Which layer (0-25 for Gemma-2-2B)
        width: SAE width ("16k" or "65k")
        device: Target device

    Returns:
        Tuple of (sae, config)
    """
    release = "gemma-scope-2b-pt-res"
    sae_id = f"layer_{layer}/width_{width}/average_l0_71"

    sae, cfg, _ = load_sae_from_saelens(release, sae_id, device)

    return sae, cfg


def load_deepseek_distill_sae(
    layer: int = 19,
    device: str = "cuda",
) -> tuple[Any, dict]:
    """
    Load SAE for DeepSeek-R1-Distill-Llama-8B.

    Args:
        layer: Which layer (default 19)
        device: Target device

    Returns:
        Tuple of (sae, config)
    """
    release = "deepseek-r1-distill-llama-8b-qresearch"
    sae_id = f"blocks.{layer}.hook_resid_post"

    sae, cfg, _ = load_sae_from_saelens(release, sae_id, device)

    return sae, cfg


# Model configurations
SUPPORTED_MODELS = {
    "gemma-2-2b": {
        "hf_name": "google/gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res",
        "hidden_size": 2304,
        "num_layers": 26,
        "default_layer": 12,
        "sae_id_template": "layer_{layer}/width_16k/average_l0_71",
    },
    "deepseek-r1-distill-8b": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "sae_release": "deepseek-r1-distill-llama-8b-qresearch",
        "hidden_size": 4096,
        "num_layers": 32,
        "default_layer": 19,
        "sae_id_template": "blocks.{layer}.hook_resid_post",
    },
}


def get_model_config(model_name: str) -> dict:
    """Get configuration for a supported model."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(SUPPORTED_MODELS.keys())}")
    return SUPPORTED_MODELS[model_name]


def load_model_and_sae(
    model_name: Literal["gemma-2-2b", "deepseek-r1-distill-8b"] = "gemma-2-2b",
    layer: int | None = None,
    device: str = "cuda",
):
    """
    Load both model and matching SAE.

    Args:
        model_name: Which model to use
        layer: Which layer to hook (None = use default)
        device: Target device

    Returns:
        Tuple of (model, tokenizer, sae, layer_idx)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = get_model_config(model_name)

    # Load model
    print(f"Loading {config['hf_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["hf_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["hf_name"],
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SAE
    layer_idx = layer if layer is not None else config["default_layer"]
    sae_id = config["sae_id_template"].format(layer=layer_idx)

    print(f"Loading SAE: {config['sae_release']} / {sae_id}")
    sae, _, _ = load_sae_from_saelens(
        config["sae_release"],
        sae_id,
        device,
    )

    print(f"Model hidden_size: {config['hidden_size']}")
    print(f"SAE d_in: {sae.cfg.d_in}")
    print(f"SAE d_sae: {sae.cfg.d_sae}")

    return model, tokenizer, sae, layer_idx
