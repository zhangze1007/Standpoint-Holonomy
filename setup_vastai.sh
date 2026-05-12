#!/bin/bash
# LCESA Pipeline Setup for vast.ai / RunPod
# ============================================
# Prerequisites:
#   1. GPU instance with >=16GB VRAM (A10G 24GB recommended)
#   2. CUDA drivers pre-installed (vast.ai images include them)
#
# Usage:
#   git clone <this-repo-url> && cd Low-Curvature-Endogenous-Standpoint-Attractor
#   bash setup_vastai.sh

set -e

echo "=== LCESA Pipeline Setup ==="

# 1. Install Python dependencies
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

# 2. HuggingFace login (Llama-2 is a gated model)
echo "[2/4] HuggingFace authentication..."
echo "You need a HuggingFace token with access to meta-llama/Llama-2-7b-chat-hf"
echo "Get one at: https://huggingface.co/settings/tokens"
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli login
else
    pip install huggingface_hub
    huggingface-cli login
fi

# 3. Verify GPU
echo "[3/4] Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {gpu} ({mem:.1f} GB)')
    if mem < 14:
        print('  WARNING: <14GB VRAM may OOM with Llama-2-7b FP16')
else:
    print('  ERROR: No GPU detected!')
    exit(1)
"

# 4. Run the pipeline
echo "[4/4] Starting Llama-2-7b pipeline..."
echo "  This will take ~20-30 minutes on A10G."
echo ""
python -m experiments.pipeline llama-7b
