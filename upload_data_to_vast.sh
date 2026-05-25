#!/usr/bin/env bash
# =============================================================================
# Upload LCESA data files to vast.ai instance
# =============================================================================
# Run this script on your LOCAL machine (where the data is).
#
# Usage:
#   export VAST_HOST=<your-vast-ip>
#   export VAST_PORT=<your-vast-ssh-port>
#   bash upload_data_to_vast.sh
#
# Or manually:
#   scp -P <PORT> -r cache/ root@<HOST>:/root/Low-Curvature-Endogenous-Standpoint-Attractor/
# =============================================================================

set -euo pipefail

MODEL="${MODEL:-llama-7b}"
REPO_DIR="Low-Curvature-Endogenous-Standpoint-Attractor"

if [ -z "${VAST_HOST:-}" ] || [ -z "${VAST_PORT:-}" ]; then
    echo "Usage:"
    echo "  export VAST_HOST=<your-vast-ip>"
    echo "  export VAST_PORT=<your-vast-ssh-port>"
    echo "  bash upload_data_to_vast.sh"
    echo ""
    echo "Find these in vast.ai instance details (Connect tab)."
    exit 1
fi

echo "Uploading data to vast.ai ..."
echo "  Host: $VAST_HOST:$VAST_PORT"
echo ""

# Create cache directory on remote
ssh -p "$VAST_PORT" root@"$VAST_HOST" \
    "mkdir -p ~/$REPO_DIR/cache/$MODEL/value_layers"

# Upload data files
echo "[1/3] Uploading activations.npz ..."
scp -P "$VAST_PORT" \
    cache/$MODEL/activations.npz \
    root@"$VAST_HOST":~/$REPO_DIR/cache/$MODEL/

echo "[2/3] Uploading grouping.npz ..."
scp -P "$VAST_PORT" \
    cache/$MODEL/${MODEL}_grouping.npz \
    root@"$VAST_HOST":~/$REPO_DIR/cache/$MODEL/

echo "[3/3] Uploading value_layers/ (32 files) ..."
scp -P "$VAST_PORT" \
    -r cache/$MODEL/value_layers/ \
    root@"$VAST_HOST":~/$REPO_DIR/cache/$MODEL/value_layers/

echo ""
echo "Done! Data uploaded to vast.ai instance."
echo ""
echo "Next steps on vast.ai:"
echo "  ssh -p $VAST_PORT root@$VAST_HOST"
echo "  cd $REPO_DIR"
echo "  export GITHUB_TOKEN=ghp_xxxxx"
echo "  export HF_TOKEN=hf_xxxxx"
echo "  bash run_all_experiments.sh"
