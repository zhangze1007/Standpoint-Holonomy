#!/usr/bin/env bash
# =============================================================================
# LCESA: One-Click Experiment Runner for vast.ai
# =============================================================================
# Usage:
#   1. Rent an A6000 48GB or A100 40GB instance on vast.ai
#   2. Clone repo: git clone https://github.com/<user>/Low-Curvature-Endogenous-Standpoint-Attractor.git
#   3. cd Low-Curvature-Endogenous-Standpoint-Attractor
#   4. Set HF_TOKEN if using gated models: export HF_TOKEN=hf_xxxxx
#   5. bash run_all_experiments.sh
#
# Expected runtime: ~2-4 hours on A6000 48GB
# Expected cost: ~$1-3 total
# =============================================================================

set -euo pipefail

MODEL="llama-7b"
LOG_DIR="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "LCESA Experiment Suite"
echo "============================================"
echo "Model: $MODEL"
echo "Log dir: $LOG_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "============================================"

# -------------------------------------------
# Step 0: Environment setup
# -------------------------------------------
echo ""
echo "[Step 0] Setting up environment ..."

pip install -q torch transformers accelerate bitsandbytes \
    numpy scipy scikit-learn pandas matplotlib seaborn \
    statsmodels huggingface_hub 2>&1 | tail -3

# HuggingFace authentication (needed for gated models like Llama-2)
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN found, logging in ..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null
else
    echo "  WARNING: HF_TOKEN not set. Experiment F (orthonormality) will fail for gated models."
    echo "  Set with: export HF_TOKEN=hf_xxxxx"
fi

# Verify GPU
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected. GPU experiments will be slow or fail.')
"

# Verify data files
echo ""
echo "[Step 0] Checking data files ..."
python3 -c "
from pathlib import Path
cache = Path('cache/$MODEL')
needed = ['activations.npz', '${MODEL}_grouping.npz']
for f in needed:
    p = cache / f
    if p.exists():
        print(f'  OK: {p} ({p.stat().st_size / 1e6:.1f} MB)')
    else:
        print(f'  MISSING: {p}')

# Check split files (optional, for memory optimization)
attn = cache / 'attention_only.npz'
vdir = cache / 'value_layers'
if attn.exists():
    print(f'  OK: {attn} ({attn.stat().st_size / 1e6:.1f} MB)')
if vdir.exists():
    n = len(list(vdir.glob('layer_*.npy')))
    print(f'  OK: {vdir}/ ({n} layer files)')
"

# -------------------------------------------
# Step 1: Experiment B — Null Grouping Controls
# -------------------------------------------
echo ""
echo "[Step 1/6] Experiment B: Null Grouping Controls"
echo "  ETA: ~30-60 min"
python3 -m experiments.experiment_b_null_grouping "$MODEL" \
    2>&1 | tee "$LOG_DIR/experiment_b.log"
echo "  Done: Experiment B"

# -------------------------------------------
# Step 2: Ablation Study
# -------------------------------------------
echo ""
echo "[Step 2/6] Ablation Study (layer drop + sequence length)"
echo "  ETA: ~20-40 min"
python3 -c "
from experiments.ablation import run_ablation_study
from experiments.config import CACHE_DIR
run_ablation_study(
    '$MODEL',
    CACHE_DIR / '$MODEL' / 'activations.npz',
    CACHE_DIR / '$MODEL' / '${MODEL}_grouping.npz',
)
" 2>&1 | tee "$LOG_DIR/ablation.log"
echo "  Done: Ablation"

# -------------------------------------------
# Step 3: Permutation Test
# -------------------------------------------
echo ""
echo "[Step 3/6] Permutation Test (1000 head permutations)"
echo "  ETA: ~10-20 min"
python3 -m experiments.baselines.permutation "$MODEL" \
    2>&1 | tee "$LOG_DIR/permutation.log"
echo "  Done: Permutation"

# -------------------------------------------
# Step 4: Experiment E — T0 Separability
# -------------------------------------------
echo ""
echo "[Step 4/6] Experiment E: T0 Separability (OLS regression)"
echo "  ETA: ~1 min"
python3 -m experiments.experiment_e_t0_separability \
    2>&1 | tee "$LOG_DIR/experiment_e.log"
echo "  Done: Experiment E"

# -------------------------------------------
# Step 5: Experiment D — Competitive Baselines
# -------------------------------------------
echo ""
echo "[Step 5/6] Experiment D: Competitive Baselines"
echo "  ETA: ~5-10 min"
python3 -m experiments.experiment_d_competitive_baselines "$MODEL" \
    2>&1 | tee "$LOG_DIR/experiment_d.log"
echo "  Done: Experiment D"

# -------------------------------------------
# Step 6: Experiment F — Orthonormality
# (Requires model weights — needs HF_TOKEN for gated models)
# -------------------------------------------
echo ""
echo "[Step 6/6] Experiment F: Orthonormality"
echo "  ETA: ~5-10 min"
if [ -n "${HF_TOKEN:-}" ] || python3 -c "
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
    python3 -m experiments.experiment_f_orthonormality \
        2>&1 | tee "$LOG_DIR/experiment_f.log"
    echo "  Done: Experiment F"
else
    echo "  SKIPPED: No HF_TOKEN, cannot load gated model"
    echo "  Run manually with: export HF_TOKEN=hf_xxxxx && python3 -m experiments.experiment_f_orthonormality"
fi

# -------------------------------------------
# Step 7: Causal Activation Patching
# -------------------------------------------
echo ""
echo "[Step 7/7] Causal Activation Patching"
echo "  ETA: ~30-60 min"
python3 -m experiments.causal_patching "$MODEL" \
    2>&1 | tee "$LOG_DIR/causal_patching.log"
echo "  Done: Causal Patching"

# -------------------------------------------
# Summary
# -------------------------------------------
echo ""
echo "============================================"
echo "All 7 experiments complete!"
echo "============================================"
echo "End: $(date)"
echo ""
echo "Results:"
ls -lh results/*.json results/*.csv 2>/dev/null || echo "  (check results/ directory)"
echo ""
echo "Logs: $LOG_DIR/"
echo ""

# Print key results if available
python3 -c "
import json
from pathlib import Path

results_dir = Path('results')
for f in sorted(results_dir.glob('*${MODEL}*')):
    if f.suffix == '.json':
        try:
            data = json.loads(f.read_text())
            # Print top-level keys only
            keys = list(data.keys())[:5]
            print(f'{f.name}: {keys}')
        except:
            pass
" 2>/dev/null

echo ""
echo "To download results:"
echo "  tar czf lcesa_results.tar.gz results/ $LOG_DIR/"
echo "  # Then: scp or upload to your storage"
