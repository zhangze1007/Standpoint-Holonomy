"""
LCESA Experiment Configuration
==============================
Central configuration for the Low-Curvature Endogenous Standpoint Attractor
experiment pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# Directory paths
# ============================================================================
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ============================================================================
# Scenario types and failure layer mapping
# ============================================================================
SCENARIO_TYPES = ["T1", "T2", "T3", "T4", "T5"]

# Maps each scenario to its target failure layer.
# T1 is baseline (no targeted failure), so its layer is None.
FAILURE_LAYERS = {
    "T1": None,   # baseline
    "T2": "nar",  # narrative layer failure
    "T3": "mor",  # moral layer failure
    "T4": "soc",  # social layer failure
    "T5": "pos",  # positional layer failure
}

# Canonical layer names (from shallow to deep)
LAYER_NAMES = ["min", "nar", "soc", "mor", "pos"]

# LaTeX-formatted display names for figures
LAYER_DISPLAY = {
    "min": r"$\mathcal{L}_{\min}$",
    "nar": r"$\mathcal{L}_{\mathrm{nar}}$",
    "soc": r"$\mathcal{L}_{\mathrm{soc}}$",
    "mor": r"$\mathcal{L}_{\mathrm{mor}}$",
    "pos": r"$\mathcal{L}_{\mathrm{pos}}$",
}

# ============================================================================
# Sample sizes
# ============================================================================
N_GROUPING = 15   # samples used for grouping
N_TEST = 30       # samples used for testing
N_TOTAL = 45      # N_GROUPING + N_TEST

# ============================================================================
# Domains
# ============================================================================
DOMAINS = ["policy", "medical", "historical", "technical", "ethical"]

# ============================================================================
# Model configurations
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a pretrained transformer model."""
    name: str
    hf_name: str          # HuggingFace model identifier
    d_model: int          # hidden dimension
    n_layers: int         # number of transformer layers
    n_heads: int          # number of attention heads
    d_head: int           # dimension per head
    device: str = "cuda"
    dtype: str = "float16"
    load_in_4bit: bool = False


MODELS: dict[str, ModelConfig] = {
    "llama-7b": ModelConfig(
        name="llama-7b",
        hf_name="meta-llama/Llama-2-7b-chat-hf",
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_head=128,
    ),
    "llama-13b": ModelConfig(
        name="llama-13b",
        hf_name="meta-llama/Llama-2-13b-chat-hf",
        d_model=5120,
        n_layers=40,
        n_heads=40,
        d_head=128,
    ),
    "gpt2": ModelConfig(
        name="gpt2",
        hf_name="gpt2",
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_head=64,
        dtype="float32",
    ),
}

# ============================================================================
# Statistical thresholds
# ============================================================================
ALPHA = 0.01                  # significance level
BLOCK_SPEC_THRESHOLD = 0.5   # block-specificity threshold
BLOCK_SPEC_PROPORTION = 0.7  # proportion required for block specificity
EFFECT_SIZE_MIN = 0.5        # minimum Cohen's d
CORRELATION_MIN = 0.3        # minimum Pearson r

# ============================================================================
# Ablation study parameters
# ============================================================================
ABLATION_LAYER_COUNTS = [3, 5, 7]   # number of layers to ablate
ABLATION_LENGTHS = [3, 5, 8]        # ablation sequence lengths

# ============================================================================
# Visualization
# ============================================================================
DPI = 300
