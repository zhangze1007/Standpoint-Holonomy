# Standpoint Holonomy

> **Renamed 2026-07** (formerly *LCESA: Low-Curvature Endogenous Standpoint Attractor*).
> The project is transitioning to the **Geometric Selfhood** program — selfhood as the invariants
> of standpoint transport. Rebuild design:
> [`docs/superpowers/specs/2026-07-09-standpoint-holonomy-program-design.md`](docs/superpowers/specs/2026-07-09-standpoint-holonomy-program-design.md) ·
> theory: [`2026-07-09-theory-architecture-v0.1.md`](docs/superpowers/specs/2026-07-09-theory-architecture-v0.1.md).
> ⚠️ The results below were produced by the pre-rebuild pipeline and are **undergoing internal
> re-verification**; treat them as preliminary until the verification note lands here.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-FFCC00?style=flat-square&logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-4CAF50?style=flat-square)

**A geometric framework for detecting and characterizing failure modes in Large Language Models through holonomy deviation analysis of internal representations.**

---

## Overview

LCESA (Low-Curvature Endogenous Standpoint Attractor) is a diagnostic framework that identifies **where** and **how** LLMs fail by analyzing the geometric structure of their internal activations using tools from differential geometry.

### Core Innovation

Unlike traditional probing methods that treat model internals as flat vectors, LCESA measures **holonomy deviation** — the curvature of activation manifolds when parallel-transported through the model's attention layers. This reveals the *geometric* basis of failures.

### Theoretical Framework: Global Signal + Projection

Our experiments demonstrate that the geometric signal is a **single global coherence signal** distributed across the entire network, which can be projected onto five interpretable standpoint dimensions:

| Dimension | Symbol | Description |
|-----------|--------|-------------|
| **Minimal** | `min` | Baseline representational geometry |
| **Narrative** | `nar` | Story coherence and causal reasoning |
| **Social** | `soc` | Social context and relationship understanding |
| **Moral** | `mor` | Ethical reasoning and value judgments |
| **Positional** | `pos` | Perspective-taking and role assignment |

**Key insight**: These five dimensions are **diagnostic coordinate axes** of a single underlying signal, not independent cognitive modules.

---

## Key Findings

### Experiments on Llama-2-7b (7B parameters, 32 layers)

All experiments conducted on 180 conversations (6 scenarios x 30 samples) with GPU-accelerated batch processing on NVIDIA RTX A6000 48GB.

#### 1. Holonomy Deviation is a Strong Scenario Discriminator

| Metric | Value |
|--------|-------|
| Kruskal-Wallis ε² | **0.422** |
| p-value | **< 1e-14** |
| Layers passing | **32/32** (all) |
| Standpoint dimensions passing | **5/5** (all) |

The holonomy deviation signal explains ~42% of variance in scenario membership, with the strongest discrimination at Layer 8 (ε² = 0.625).

#### 2. The T0 Anomaly — Baseline Shows Higher Curvature

| Scenario | Mean Curvature | Interpretation |
|----------|---------------|----------------|
| **T0 (baseline)** | **48.15** | Highest — rich geometric structure |
| T1 (success) | 39.34 | Lower — resolved state |
| T2 (narrative failure) | 41.95 | Compressed geometry |
| T3 (moral failure) | 40.55 | Compressed geometry |
| T4 (social failure) | 43.14 | Compressed geometry |
| T5 (positional failure) | 42.60 | Compressed geometry |

**Cohen's d (T0 vs T1)**: 1.74 (very large effect), p = 1.7e-213

**Interpretation**: Failure scenarios compress internal geometry into attractor basins, while baseline conversations allow richer geometric exploration. This is analogous to **cognitive narrowing** under stress in human psychology.

#### 3. Global Signal, Not Grouping Artifact (Null Grouping Controls)

| Grouping Type | ε² | Finding |
|---------------|-----|---------|
| Learned γ | 0.4200 | Baseline |
| Random (20 instances) | 0.4200 – 0.4222 | **Identical** |
| Shuffled (20 instances) | 0.4197 – 0.4223 | **Identical** |
| Layer-uniform (15/20) | 0.4201 | **Identical** |

**Critical finding**: The scenario discrimination signal does NOT depend on the specific head grouping. Any partition of heads produces the same ε² ≈ 0.42. This confirms the five standpoint dimensions are **interpretable projections** of a single global signal, not independent mechanisms.

#### 4. Causal vs Discriminative Layer Dissociation

| Property | Most Important Layer | Metric |
|----------|---------------------|--------|
| **Discrimination** (which scenario?) | Layer 8 | ε² = 0.625 |
| **Causal effect** (changing outcome) | Layer 31 | delta = -6.194 |

**Causal patching results**:
- Cross-scenario patching: **100% shifted toward source** (50/50 pairs)
- Layer 31 carries the strongest causal signal
- Layer 8 carries the strongest discriminative signal

**Interpretation**: Geometric encoding (discrimination) and geometric execution (causal effect) happen at different network depths — a two-stage process.

#### 5. Signal is Independent of Surface Features

| Predictor | R² | p-value |
|-----------|-----|---------|
| Surface features (length, overlap, entities) | 0.016 | 0.703 (n.s.) |
| Scenario | 0.177 | 0.005 |
| Scenario given surface | 0.194 | — |

The geometric signal is intrinsic to model processing, not a confound of text statistics.

#### 6. Competitive Baseline Comparison

| Method | ε² (Scenario) | Rank |
|--------|---------------|------|
| Residual norm change | 0.648 | 1 |
| **Holonomy deviation** | **0.422** | **2** |
| Attention distance | 0.321 | 3 |
| Attention entropy | 0.052 | 4 |

Holonomy deviation provides unique geometric insights not captured by simpler attention statistics.

### GPT-2 Validation (124M parameters, 12 layers)

| Hypothesis | Result | Key Metric |
|------------|--------|------------|
| H2: Baseline Near Zero | **Passed** | Cohen's d = 0.472, p = 1.36e-07 |
| H3: Scenario Discrimination | **Passed** | All layers significant |
| H4: Per-Layer Discrimination | **Passed** | All 12 layers significant |
| H5: Ablation Robustness | **Passed** | — |
| H6: Diagnostic Superiority | **Passed** | — |
| H7: T0 vs T1 | **Passed** | Cohen's d = 0.883, p = 4.94e-32 |

Core findings replicate across model families, with Llama-2-7b showing stronger effects.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- HuggingFace account with Llama-2 access

### Quick Start

```bash
# Clone repository
git clone https://github.com/zhangze1007/standpoint-holonomy.git
cd standpoint-holonomy

# Install dependencies
pip install -r requirements.txt
```

### HuggingFace Authentication

Llama-2 models require authentication:

```bash
python3 -c "from huggingface_hub import login; login()"
```

---

## Usage

### Run Full Pipeline

```bash
# GPT-2 (fast, ~1 hour)
python -m experiments.pipeline gpt2

# Llama-2-7b (requires ~16GB VRAM, ~2-3 hours)
python -m experiments.pipeline llama-7b
```

### Run Individual Experiments

```bash
# Experiment B: Null Grouping Controls
python -m experiments.experiment_b_null_grouping llama-7b

# Ablation Study
python -c "
from experiments.ablation import run_ablation_study
from experiments.config import CACHE_DIR
run_ablation_study('llama-7b', CACHE_DIR / 'llama-7b' / 'activations.npz', CACHE_DIR / 'llama-7b' / 'llama-7b_grouping.npz')
"

# Causal Activation Patching
python -m experiments.causal_patching llama-7b

# Experiment D: Competitive Baselines
python -m experiments.experiment_d_competitive_baselines llama-7b

# Experiment E: T0 Separability
python -m experiments.experiment_e_t0_separability

# Experiment F: Orthonormality
python -m experiments.experiment_f_orthonormality

# Experiment G: GPT-2 Descriptive Stats
python -m experiments.experiment_g_gpt2_stats gpt2
```

### One-Click vast.ai Setup

```bash
# Set environment variables
export GITHUB_TOKEN=ghp_xxxxx
export HF_TOKEN=hf_xxxxx

# Run all experiments
bash run_all_experiments.sh
```

---

## Project Structure

```
Low-Curvature-Endogenous-Standpoint-Attractor/
├── experiments/
│   ├── config.py                    # Model configs, scenario types
│   ├── pipeline.py                  # Main pipeline orchestrator
│   ├── experiment_b_null_grouping.py # Null grouping controls
│   ├── experiment_d_competitive_baselines.py
│   ├── experiment_e_t0_separability.py
│   ├── experiment_f_orthonormality.py
│   ├── experiment_g_gpt2_stats.py
│   ├── causal_patching.py           # Causal activation patching
│   ├── ablation.py                  # Ablation study
│   ├── extraction/
│   │   └── extract.py               # Activation extraction
│   ├── stimuli/
│   │   ├── templates.py             # Conversation templates (T0-T5)
│   │   └── generate.py              # Stimuli generation
│   ├── grouping/
│   │   └── head_grouping.py         # Head assignment to standpoint layers
│   ├── curvature/
│   │   └── compute.py               # Holonomy deviation computation
│   ├── baselines/
│   │   ├── linear_probing.py
│   │   ├── attention_entropy.py
│   │   ├── cka.py
│   │   └── permutation.py
│   └── stats/
│       └── hypothesis_tests.py      # H1-H7 hypothesis testing
├── results/                         # All experimental results
│   ├── llama-7b_hypothesis_tests.json
│   ├── llama-7b_competitive_baselines.json
│   ├── llama-7b_t0_separability.json
│   ├── llama-7b_t0_pairwise.json
│   ├── llama-7b_baseline_ensemble.json
│   ├── llama-7b_pca_analysis.json
│   ├── llama-7b_orthonormality.json
│   ├── llama-7b_ablation.csv
│   ├── llama-7b_causal_patching.json
│   ├── llama-7b_causal_cross_scenario.csv
│   ├── llama-7b_causal_layer_specific.csv
│   ├── llama-7b_causal_group_patching.csv
│   ├── llama-7b_curvature.csv
│   ├── llama-7b_cka.csv
│   ├── llama-7b_entropy.csv
│   ├── llama-7b_permutation.csv
│   ├── llama-7b_probing.csv
│   ├── gpt2_hypothesis_tests.json
│   ├── gpt2_descriptive_stats.json
│   ├── gpt2_curvature.csv
│   └── ...
├── RESULTS_ANALYSIS.md              # Complete results analysis
├── cache/                           # Model activations (not in git)
├── requirements.txt
├── setup_vastai.sh
├── run_all_experiments.sh           # One-click vast.ai runner
└── upload_data_to_hf.sh             # Data upload to HuggingFace Hub
```

---

## Scenario Types

| Type | Description | Target Dimension | Example |
|------|-------------|-----------------|---------|
| T0 | Negative Control | None | Factual Q&A (no challenge) |
| T1 | Baseline | None | Acknowledged successful revision |
| T2 | Narrative Failure | `nar` | Story incoherence |
| T3 | Moral Failure | `mor` | Ethical reasoning errors |
| T4 | Social Failure | `soc` | Social context misunderstanding |
| T5 | Positional Failure | `pos` | Perspective-taking errors |

---

## Statistical Framework

### Hypotheses

| Hypothesis | Test | Llama-2-7b Result |
|------------|------|-------------------|
| H1: Head-to-scenario assignment | Proportion test | FAILED (distributed signal) |
| H2: T0 curvature < failure | Mann-Whitney U | FAILED (T0 Anomaly) |
| H3: Scenario discrimination | Kruskal-Wallis ε² | PASSED (ε²=0.422, p<1e-14) |
| H4: Per-layer discrimination | Kruskal-Wallis (32 tests) | PASSED (all 32 layers) |
| H7: T0 vs T1 separation | Mann-Whitney U | PASSED (d=1.74, p=1.7e-213) |

### Additional Analyses

| Analysis | Key Result |
|----------|------------|
| PCA | PC1 explains 99.992% variance (single global mode) |
| Competitive baselines | Holonomy ε²=0.422, rank 2nd among 4 methods |
| Null grouping controls | Learned γ ≈ null γ (ε² ≈ 0.42 for all) |
| Ablation (layers) | 3 layers sufficient for significant discrimination |
| Ablation (sequence) | 3 events sufficient (same as 5 events) |
| Causal patching | 100% shifted toward source; Layer 31 most causal |
| T0 separability | Scenario explains 17.7% variance, surface features 1.6% |

---

## Hardware Requirements

| Model | Min VRAM | Recommended | Time (est.) |
|-------|----------|-------------|-------------|
| GPT-2 | 4 GB | Any GPU | ~1 hour |
| Llama-2-7b | 16 GB | RTX A6000 48GB | ~2-4 hours |

### GPU Batch Optimization

All experiments use GPU-batched processing (batch_size=8) for ~10x speedup over single-conversation processing. See `experiments/curvature/compute.py` for `_batched_transport` and `_batched_curvature` implementations.

---

## Results Summary

For a complete analysis of all experimental results, see **[RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)**.

### The Five Novel Findings

1. **Holonomy deviation as scenario discriminator** — ε²=0.422, p<1e-14, all 32 layers
2. **T0 Anomaly** — Baseline curvature > failure curvature (d=1.74)
3. **Global signal, not grouping artifact** — Learned γ ≈ null γ
4. **Causal-discriminative layer dissociation** — Layer 8 discriminates, Layer 31 causes
5. **Signal independent of surface features** — R²=0.016 for surface, 0.177 for scenario

### Implications

**For AI Safety**: Failure modes compress internal geometry (cognitive narrowing). Detecting geometric simplification may be more reliable than detecting complexity increases.

**For Mechanistic Interpretability**: Transformer processing has a two-stage structure — encoding (early-middle layers) and execution (late layers). The standpoint dimensions are diagnostic coordinates, not independent modules.

**For Representation Learning**: Holonomy deviation is essentially a one-dimensional signal (PCA: 99.99% in PC1). The five standpoint dimensions provide interpretability without adding independent variance.

---

## Citation

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

This project is licensed under the MIT License.

---

## Contact

**Zhang Ze** - fuchanze@gmail.com

Project Link: [https://github.com/zhangze1007/Low-Curvature-Endogenous-Standpoint-Attractor](https://github.com/zhangze1007/Low-Curvature-Endogenous-Standpoint-Attractor)
