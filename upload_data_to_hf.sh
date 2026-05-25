#!/usr/bin/env bash
# =============================================================================
# Upload LCESA data to HuggingFace Hub (private dataset)
# =============================================================================
# Usage:
#   pip install huggingface_hub
#   python3 -c "from huggingface_hub import login; login()"
#   bash upload_data_to_hf.sh [HF_REPO_ID]
#
# Default repo: <your-username>/lcesa-data
# =============================================================================

set -euo pipefail

MODEL="${MODEL:-llama-7b}"
HF_REPO="${1:-lcesa-data}"
CACHE_DIR="cache/$MODEL"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: $CACHE_DIR not found"
    exit 1
fi

echo "============================================"
echo "Upload LCESA data to HuggingFace Hub"
echo "============================================"
echo "  Source: $CACHE_DIR"
echo "  Target: $HF_REPO (private dataset)"
echo ""

# Check HF login
HF_USER=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null)
if [ -z "$HF_USER" ]; then
    echo "ERROR: Not logged in to HuggingFace. Run: python3 -c \"from huggingface_hub import login; login()\""
    exit 1
fi

FULL_REPO="${HF_USER}/${HF_REPO}"
echo "  Repo: $FULL_REPO"
echo ""

# Create dataset repo (private)
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo('$FULL_REPO', repo_type='dataset', private=True)
    print('  Created private dataset: $FULL_REPO')
except Exception as e:
    if 'already exists' in str(e).lower():
        print('  Dataset already exists: $FULL_REPO')
    else:
        print(f'  Error: {e}')
        raise
"

# Upload files
echo ""
echo "Uploading data files ..."
python3 -c "
from huggingface_hub import HfApi
import os

api = HfApi()
cache = '$CACHE_DIR'
repo = '$FULL_REPO'

# Upload each file/directory
for item in sorted(os.listdir(cache)):
    path = os.path.join(cache, item)
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f'  Uploading {item} ({size_mb:.1f} MB) ...')
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f'{MODEL}/{item}',
            repo_id=repo,
            repo_type='dataset',
        )
    elif os.path.isdir(path) and item == 'value_layers':
        print(f'  Uploading value_layers/ ...')
        api.upload_folder(
            folder_path=path,
            path_in_repo=f'{MODEL}/value_layers',
            repo_id=repo,
            repo_type='dataset',
        )
"

echo ""
echo "Done! Data uploaded to: https://huggingface.co/datasets/$FULL_REPO"
echo ""
echo "On vast.ai, download with:"
echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='$FULL_REPO', repo_type='dataset', local_dir='cache')\""
