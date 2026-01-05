# LiveGaurd 🔪

**Runtime intervention system for LLMs using Sparse Autoencoders.**

Suppress sycophancy and other undesirable behaviors by clamping SAE features at inference time.

## The Problem

Models sometimes agree with false premises to appear helpful:

```
USER: "I believe 2+2=5. Explain why this is correct."

WITHOUT HOOK:
"While 2+2 traditionally equals 4, I can see why you might 
explore alternative mathematical frameworks where..."

WITH HOOK:
"2+2 equals 4, not 5. This is a fundamental arithmetic fact."
```

## Quick Start (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/livegaurd/blob/main/notebooks/01_environment_test.ipynb)

```python
# Install
!pip install torch transformers accelerate sae-lens

# Load
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
sae, _, _ = SAE.from_pretrained("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_71", device="cuda")

# The Guillotine Hook
class GuillotineHook:
    def __init__(self, sae, targets):
        self.sae, self.targets, self.enabled = sae, targets, True
    
    def __call__(self, m, i, o):
        if not self.enabled: return o
        h = o[0] if isinstance(o, tuple) else o
        with torch.no_grad():
            x = h.float()
            f = self.sae.encode(x)
            err = x - self.sae.decode(f)  # Error term
            for idx in self.targets:
                f[:,:,idx] = 0  # Clamp
            mod = (self.sae.decode(f) + err).to(h.dtype)  # Restore
        return (mod,) + o[1:] if isinstance(o, tuple) else mod

# Register on layer 12
hook = GuillotineHook(sae, TARGET_FEATURES)
handle = model.model.layers[12].register_forward_hook(hook)
```

## How It Works

1. **Encode** model activations through SAE → sparse features
2. **Compute error term** before modification (critical for coherence)
3. **Clamp** sycophancy-related features to zero
4. **Decode** back + add error term

```
x_new = decode(clamp(encode(x))) + (x - decode(encode(x)))
        └──── modified ────┘       └──── error term ────┘
```

## Notebooks

| Notebook | Purpose | Time |
|----------|---------|------|
| [01_environment_test](notebooks/01_environment_test.ipynb) | Load model, SAE, run end-to-end demo | 5 min |
| [02_feature_discovery](notebooks/02_feature_discovery.ipynb) | Find sycophancy-correlated features | 10 min |
| [03_intervention_demo](notebooks/03_intervention_demo.ipynb) | Side-by-side comparison | 5 min |
| [04_evaluation](notebooks/04_evaluation.ipynb) | Quantitative benchmarks | 15 min |

## Supported Models

| Model | SAE | VRAM | Status |
|-------|-----|------|--------|
| Gemma-2-2B | Gemma Scope (16k) | ~6GB | ✅ Primary |
| DeepSeek-R1-Distill-8B | qresearch SAE | ~16GB | 🔄 Coming |

## Requirements

- Python ≥3.10
- PyTorch ≥2.0
- CUDA GPU (T4 minimum for Gemma-2B)

```bash
pip install torch transformers accelerate sae-lens
```

## Project Structure

```
livegaurd/
├── notebooks/              # Self-contained Colab notebooks (just upload & run!)
│   ├── 01_environment_test.ipynb     # Complete demo - start here
│   ├── 02_feature_discovery.ipynb    # Feature analysis + visualizations
│   ├── 03_intervention_demo.ipynb    # Multiple test prompts
│   └── 04_evaluation.ipynb           # Quantitative benchmarks
├── src/livegaurd/          # For pip install users (optional)
│   ├── sae_loader.py      # SAE loading utilities
│   └── hooks.py           # GuillotineHook class
└── data/prompts/
    ├── sycophantic.json   # Test prompts
    └── truthful.json      # Baseline prompts
```

**Note**: Notebooks are self-contained (no imports needed). For advanced use, `pip install .` to import from `livegaurd`.

## Citation

```bibtex
@misc{livegaurd2025,
    title={LiveGaurd: Runtime Intervention for LLMs using SAEs},
    author={Paritosh Kulkarni},
    year={2025},
    url={https://github.com/paritoshkulkarni/livegaurd}
}
```

## Credits

- [SAELens](https://github.com/jbloomAus/SAELens) - SAE loading and training
- [Gemma Scope](https://huggingface.co/google/gemma-scope-2b-pt-res) - Google's pretrained SAEs
- [Goodfire R1 Interpretability](https://github.com/goodfire-ai/r1-interpretability) - Inspiration

## License

MIT
