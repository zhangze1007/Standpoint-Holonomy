#!/usr/bin/env bash
# =============================================================================
# LCESA: One-Click Experiment Runner for vast.ai (Private Repo)
# =============================================================================
# Usage:
#   1. Rent an A6000 48GB or A100 40GB instance on vast.ai
#   2. SSH into the instance
#   3. Set environment variables:
#        export GITHUB_TOKEN=ghp_xxxxx       # GitHub Personal Access Token (需要 repo 权限)
#        export HF_TOKEN=hf_xxxxx            # HuggingFace Token (Llama-2 是 gated 模型)
#        export HF_DATA_REPO=user/lcesa-data # HF dataset repo (数据文件)
#   4. Run this script:
#        bash run_all_experiments.sh
#
# Expected runtime: ~2-4 hours on A6000 48GB
# Expected cost: ~$1-3 total
#
# 数据文件 (cache/) 不在 git 里，会自动从 HuggingFace Hub 下载。
# 先在本地运行 upload_data_to_hf.sh 上传数据。
# =============================================================================

set -euo pipefail

MODEL="llama-7b"
REPO_URL="https://github.com/zhangze1007/Low-Curvature-Endogenous-Standpoint-Attractor.git"
REPO_DIR="Low-Curvature-Endogenous-Standpoint-Attractor"
LOG_DIR="$REPO_DIR/logs_$(date +%Y%m%d_%H%M%S)"

# -------------------------------------------
# Step 0: Clone repo (private, needs GITHUB_TOKEN)
# -------------------------------------------
echo "============================================"
echo "LCESA Experiment Suite"
echo "============================================"

if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "ERROR: GITHUB_TOKEN not set."
    echo "  Create a Personal Access Token at: https://github.com/settings/tokens"
    echo "  Needs 'repo' scope for private repos."
    echo "  Then: export GITHUB_TOKEN=ghp_xxxxx"
    exit 1
fi

if [ ! -d "$REPO_DIR" ]; then
    echo "[Setup] Cloning private repo ..."
    git clone "https://${GITHUB_TOKEN}@github.com/zhangze1007/Low-Curvature-Endogenous-Standpoint-Attractor.git"
else
    echo "[Setup] Repo already exists, pulling latest ..."
    cd "$REPO_DIR" && git pull && cd ..
fi

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

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
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
else
    echo "  WARNING: HF_TOKEN not set. Experiment F will fail for gated models."
    echo "  Set with: export HF_TOKEN=hf_xxxxx"
fi

# -------------------------------------------
# Step 0b: Check / download data files
# -------------------------------------------
echo ""
echo "[Step 0b] Checking data files ..."

CACHE_DIR="$REPO_DIR/cache/$MODEL"
DATA_MISSING=0

for f in "activations.npz" "${MODEL}_grouping.npz"; do
    if [ ! -f "$CACHE_DIR/$f" ]; then
        echo "  MISSING: $CACHE_DIR/$f"
        DATA_MISSING=1
    else
        echo "  OK: $CACHE_DIR/$f ($(du -h "$CACHE_DIR/$f" | cut -f1))"
    fi
done

if [ "$DATA_MISSING" -eq 1 ]; then
    echo ""
    echo "  Data files missing. Attempting download from HuggingFace Hub ..."

    if [ -z "${HF_DATA_REPO:-}" ]; then
        HF_DATA_REPO="zhangze1007/lcesa-activations"
        echo "  HF_DATA_REPO not set, using default: $HF_DATA_REPO"
    fi

    echo "  Downloading from $HF_DATA_REPO ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_DATA_REPO', repo_type='dataset', local_dir='.')
"

    # Verify download
    if [ ! -f "$CACHE_DIR/activations.npz" ]; then
        echo "  ERROR: Download failed or data not found in repo."
        echo "  Check that $HF_DATA_REPO contains cache/$MODEL/"
        exit 1
    fi
    echo "  Download complete."
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

# -------------------------------------------
# Step 1: Main Pipeline (generates curvature CSV for C/G/H)
# -------------------------------------------
echo ""
echo "[Step 1/11] Main Pipeline (curvature, probing, CKA, hypothesis tests)"
echo "  ETA: ~2-3 hours"
python3 -m experiments.pipeline "$MODEL" \
    2>&1 | tee "$LOG_DIR/pipeline.log"
echo "  Done: Pipeline"

# -------------------------------------------
# Step 2: Experiment A — Baseline Ensemble
# -------------------------------------------
echo ""
echo "[Step 2/11] Experiment A: Baseline Ensemble (5 random U_exp subsets)"
echo "  ETA: ~30 min"
python3 -m experiments.experiment_a_baseline_ensemble "$MODEL" \
    2>&1 | tee "$LOG_DIR/experiment_a.log"
echo "  Done: Experiment A"

# -------------------------------------------
# Step 3: Experiment B — Null Grouping Controls
# -------------------------------------------
echo ""
echo "[Step 3/11] Experiment B: Null Grouping Controls"
echo "  ETA: ~30-60 min"
python3 -m experiments.experiment_b_null_grouping "$MODEL" \
    2>&1 | tee "$LOG_DIR/experiment_b.log"
echo "  Done: Experiment B"

# -------------------------------------------
# Step 4: Ablation Study
# -------------------------------------------
echo ""
echo "[Step 4/11] Ablation Study (layer drop + sequence length)"
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
# Step 5: Permutation Test
# -------------------------------------------
echo ""
echo "[Step 5/11] Permutation Test (1000 head permutations)"
echo "  ETA: ~10-20 min"
python3 -m experiments.baselines.permutation "$MODEL" \
    2>&1 | tee "$LOG_DIR/permutation.log"
echo "  Done: Permutation"

# -------------------------------------------
# Step 6: Experiment E — T0 Separability
# -------------------------------------------
echo ""
echo "[Step 6/11] Experiment E: T0 Separability (OLS regression)"
echo "  ETA: ~1 min"
python3 -m experiments.experiment_e_t0_separability \
    2>&1 | tee "$LOG_DIR/experiment_e.log"
echo "  Done: Experiment E"

# -------------------------------------------
# Step 7: Experiment D — Competitive Baselines
# -------------------------------------------
echo ""
echo "[Step 7/11] Experiment D: Competitive Baselines"
echo "  ETA: ~5-10 min"
python3 -m experiments.experiment_d_competitive_baselines "$MODEL" \
    2>&1 | tee "$LOG_DIR/experiment_d.log"
echo "  Done: Experiment D"

# -------------------------------------------
# Step 8: Experiment F — Orthonormality
# (Requires model weights — needs HF_TOKEN for gated models)
# -------------------------------------------
echo ""
echo "[Step 8/11] Experiment F: Orthonormality"
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
# Step 9: Causal Activation Patching
# -------------------------------------------
echo ""
echo "[Step 9/11] Causal Activation Patching"
echo "  ETA: ~30-60 min"
python3 -m experiments.causal_patching "$MODEL" \
    2>&1 | tee "$LOG_DIR/causal_patching.log"
echo "  Done: Causal Patching"

# -------------------------------------------
# Step 10: Experiment C — PCA Decomposition
# (Needs curvature CSV from pipeline)
# -------------------------------------------
echo ""
echo "[Step 10/11] Experiment C: PCA Decomposition of Block Holonomy"
echo "  ETA: ~5 min"
python3 -m experiments.experiment_c_pca "$MODEL" \
    2>&1 | tee "$LOG_DIR/experiment_c.log"
echo "  Done: Experiment C"

# -------------------------------------------
# Step 11: Experiment G — GPT-2 Descriptive Stats & H1a
# -------------------------------------------
echo ""
echo "[Step 11a/11] Experiment G: Descriptive Statistics & H1a"
echo "  ETA: ~5 min"
python3 -m experiments.experiment_g_gpt2_stats "$MODEL" \
    2>&1 | tee "$LOG_DIR/experiment_g.log"
echo "  Done: Experiment G"

# -------------------------------------------
# Step 11b: Experiment H — T0 Pairwise Comparisons
# -------------------------------------------
echo ""
echo "[Step 11b/11] Experiment H: T0 Pairwise Comparisons"
echo "  ETA: ~5 min"
python3 -m experiments.experiment_h_t0_pairwise "$MODEL" \
    2>&1 | tee "$LOG_DIR/experiment_h.log"
echo "  Done: Experiment H"

# -------------------------------------------
# Summary
# -------------------------------------------
echo ""
echo "============================================"
echo "All 11 experiments complete!"
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
