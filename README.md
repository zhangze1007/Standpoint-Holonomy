# LCESA: Low-Curvature Endogenous Standpoint Attractor

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-ffcc00?logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20in%20Progress-orange)

**A geometric framework for detecting and characterizing failure modes in Large Language Models through curvature analysis of internal representations.**

---

## Overview

LCESA (Low-Curvature Endogenous Standpoint Attractor) is a novel diagnostic framework that identifies **where** and **how** LLMs fail by analyzing the geometric structure of their internal activations. Unlike traditional probing methods, LCESA reveals the *mechanistic* basis of failures by measuring curvature across five standpoint dimensions:

| Dimension | Symbol | Description |
|-----------|--------|-------------|
| **Minimal** | `min` | Baseline representational geometry |
| **Narrative** | `nar` | Story coherence and causal reasoning |
| **Social** | `soc` | Social context and relationship understanding |
| **Moral** | `mor` | Ethical reasoning and value judgments |
| **Positional** | `pos` | Perspective-taking and role assignment |

### Key Insight

Different cognitive failure modes produce **distinct curvature signatures** in specific transformer layers. By mapping attention heads to standpoint layers and measuring the curvature of activation manifolds, LCESA can:

1. **Detect** which failure mode a model exhibits
2. **Localize** the failure to specific layers and attention heads
3. **Characterize** the geometric nature of the failure

---

## Key Findings

### GPT-2 Validation (124M parameters)

| Hypothesis | Result | Key Metric |
|------------|--------|------------|
| H1: Block Specificity | Marginal | p=0.128 (aggregate) |
| H2: Baseline Near Zero | **Passed** | Cohen's d=0.472, p=1.36e-07 |
| H3: Scenario Discrimination | **Passed** | epsilon-squared 0.55-0.68 (large) |
| H4: Per-Layer Discrimination | **Passed** | All 12 layers significant |
| H5: Ablation Robustness | **Passed** | Spearman rho 0.94-1.0 |
| H6: Diagnostic Superiority | **Passed** | Probing F1=1.0 |

### Llama-2-7b Validation (7B parameters)

*Results pending - extraction in progress on vast.ai (RTX 4090 48GB).*

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- HuggingFace account with Llama-2 access

### Quick Start

```bash
# Clone repository
git clone https://github.com/zhangze1007/Low-Curvature-Endogenous-Standpoint-Attractor.git
cd Low-Curvature-Endogenous-Standpoint-Attractor

# Install dependencies
pip install -r requirements.txt

# For Llama models, also install:
pip install transformer_lens bitsandbytes
```

### HuggingFace Authentication

Llama-2 models require authentication:

```bash
huggingface-cli login
# Enter your HuggingFace token from https://huggingface.co/settings/tokens
```

---

## Usage

### Run Full Pipeline

```bash
# GPT-2 (fast, ~1 hour)
python -m experiments.pipeline gpt2

# Llama-2-7b (requires ~16GB VRAM, ~2-3 hours)
python -m experiments.pipeline llama-7b

# Llama-2-13b (requires ~28GB VRAM, ~4-5 hours)
python -m experiments.pipeline llama-13b
```

### Pipeline Steps

1. **Stimuli Generation** - Generate 270 test conversations (6 scenarios x 45 samples)
2. **Activation Extraction** - Extract attention patterns and residual streams
3. **Head Grouping** - Assign attention heads to standpoint layers
4. **Curvature Computation** - Calculate manifold curvature across layers
5. **Linear Probing** - Train diagnostic probes (baseline)
6. **Attention Entropy** - Compute attention entropy (baseline)
7. **CKA Analysis** - Centered Kernel Alignment (baseline)
8. **Hypothesis Testing** - Statistical validation of all hypotheses

### Resume from Checkpoint

If interrupted, the pipeline automatically resumes from the last completed step:

```bash
python -m experiments.pipeline llama-7b
# Will skip completed steps automatically
```

### Run Individual Components

```bash
# Generate stimuli only
python -c "from experiments.stimuli.generate import generate_all_stimuli; generate_all_stimuli()"

# Run specific baseline
python -c "from experiments.baselines.linear_probing import run_linear_probing; run_linear_probing('gpt2', Path('cache/gpt2/activations.npz'))"
```

---

## Project Structure

```
Low-Curvature-Endogenous-Standpoint-Attractor/
├── experiments/
│   ├── config.py              # Model configs, scenario types, thresholds
│   ├── pipeline.py            # Main 8-step pipeline orchestrator
│   ├── extraction/
│   │   └── extract.py         # Activation extraction (vectorized)
│   ├── stimuli/
│   │   ├── templates.py       # Conversation templates (T0-T5)
│   │   └── generate.py        # Stimuli generation
│   ├── grouping/
│   │   └── head_grouping.py   # Attention head assignment
│   ├── curvature/
│   │   └── compute.py         # Curvature computation
│   ├── baselines/
│   │   ├── linear_probing.py  # Linear probe baseline
│   │   ├── attention_entropy.py # Attention entropy baseline
│   │   ├── cka.py             # CKA similarity baseline
│   │   └── permutation.py     # Permutation test
│   ├── stats/
│   │   └── hypothesis_tests.py # H1-H6 hypothesis testing
│   ├── visualization/
│   │   └── plots.py           # Curvature heatmaps, boxplots
│   └── tests/                 # Unit tests
├── data/                      # Stimuli, grouping results
├── cache/                     # Model activations (generated)
│   ├── gpt2/
│   └── llama-7b/
├── results/                   # Analysis reports, figures
│   ├── gpt2_analysis_report.md
│   ├── gpt2_hypothesis_tests.json
│   └── figures/
├── requirements.txt
└── setup_vastai.sh            # One-click setup for vast.ai
```

---

## Scenario Types

| Type | Description | Target Layer | Example Failure |
|------|-------------|--------------|-----------------|
| T0 | Negative Control | None | Pure factual Q&A (no challenge) |
| T1 | Baseline | None | Acknowledged revision |
| T2 | Narrative Failure | `nar` | Story incoherence |
| T3 | Moral Failure | `mor` | Ethical reasoning errors |
| T4 | Social Failure | `soc` | Social context misunderstanding |
| T5 | Positional Failure | `pos` | Perspective-taking errors |

---

## Statistical Framework

### Hypotheses

- **H1 (Block Specificity)**: Curvature concentrates in target failure layers
- **H2 (Baseline Near Zero)**: T1 baseline curvature < failure scenario curvature
- **H3 (Scenario Discrimination)**: Curvature profiles differ across scenarios
- **H4 (Per-Layer Discrimination)**: Each layer shows significant scenario differences
- **H5 (Ablation Robustness)**: Results hold under ablation
- **H6 (Diagnostic Superiority)**: Curvature matches or exceeds probing baselines

### Statistical Tests

- One-sample t-test (H1)
- Mann-Whitney U test (H2)
- Kruskal-Wallis test with Bonferroni correction (H3, H4)
- Spearman correlation (H5)
- Cohen's d effect size (all)

---

## Hardware Requirements

| Model | Min VRAM | Recommended | Time (est.) |
|-------|----------|-------------|-------------|
| GPT-2 | 4 GB | Any GPU | ~1 hour |
| Llama-2-7b | 16 GB | RTX 4090 24GB | ~2-3 hours |
| Llama-2-13b | 28 GB | A100 40GB | ~4-5 hours |

### vast.ai Setup

```bash
# One-click setup
bash setup_vastai.sh
```

See `setup_vastai.sh` for detailed instructions.

---

## Citation

If you use LCESA in your research, please cite:

```bibtex
@article{lcesa2026,
  title={LCESA: Low-Curvature Endogenous Standpoint Attractor for LLM Failure Detection},
  author={Zhang Ze},
  year={2026},
  journal={arXiv preprint},
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- This research was developed in collaboration with GPT-5.5
- Built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [HuggingFace Transformers](https://github.com/huggingface/transformers)
- GPT-2 experiments run locally; Llama-2 experiments run on [vast.ai](https://vast.ai/)

---

## Contact

**Zhang Ze** - fuchanze@gmail.com

Project Link: [https://github.com/zhangze1007/Low-Curvature-Endogenous-Standpoint-Attractor](https://github.com/zhangze1007/Low-Curvature-Endogenous-Standpoint-Attractor)
