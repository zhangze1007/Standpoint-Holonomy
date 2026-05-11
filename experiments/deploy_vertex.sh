#!/bin/bash
# Deploy LCESA experiment to Vertex AI
# Usage: bash experiments/deploy_vertex.sh [model_name]
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Vertex AI API enabled
#   - Sufficient quota for T4/A100 GPUs
#
# Budget: ~$15 for LLaMA-7B (T4, ~4hrs), ~$30 for LLaMA-13B (A100, ~3hrs)

set -e

MODEL_NAME="${1:-llama-7b}"
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="${GCP_REGION:-us-central1}"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project configured. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "=== LCESA Vertex AI Deployment ==="
echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Model:    $MODEL_NAME"
echo ""

# Machine/accelerator config per model
case "$MODEL_NAME" in
    llama-7b)
        MACHINE_TYPE="n1-standard-8"
        ACCELERATOR="type=nvidia-tesla-t4,count=1"
        ;;
    llama-13b)
        MACHINE_TYPE="n1-standard-16"
        ACCELERATOR="type=nvidia-tesla-a100,count=1"
        ;;
    gpt2)
        echo "GPT-2 can run on Colab free tier. No Vertex AI deployment needed."
        echo "Run locally: python -m experiments.pipeline gpt2"
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        echo "Available: llama-7b, llama-13b, gpt2"
        exit 1
        ;;
esac

BUCKET_NAME="gs://${PROJECT_ID}-lcesa-experiments"
IMAGE="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"

# Create bucket
echo "Creating storage bucket..."
gsutil mb -l "$REGION" "$BUCKET_NAME" 2>/dev/null || echo "  Bucket already exists."

# Upload experiment code
echo "Uploading experiment code..."
gsutil -m cp -r experiments/ "$BUCKET_NAME/code/"
gsutil cp data/stimuli.json "$BUCKET_NAME/code/data/" 2>/dev/null || true

# Submit custom job
echo "Submitting Vertex AI job..."
gcloud ai custom-jobs create \
    --region="$REGION" \
    --display-name="lcesa-${MODEL_NAME}" \
    --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator=${ACCELERATOR},replica-count=1,container-image-uri=${IMAGE}" \
    --args="pip install transformer_lens transformers accelerate bitsandbytes numpy scipy scikit-learn pandas matplotlib seaborn sentence-transformers && cd /gcs/${PROJECT_ID}-lcesa-experiments/code && python -m experiments.pipeline ${MODEL_NAME}"

echo ""
echo "Job submitted. Monitor at:"
echo "https://console.cloud.google.com/vertex-ai/training/custom-jobs"
echo ""
echo "Results will be saved to: ${BUCKET_NAME}/code/results/"
echo "Download with: gsutil -m cp -r ${BUCKET_NAME}/code/results/ ./results/"
