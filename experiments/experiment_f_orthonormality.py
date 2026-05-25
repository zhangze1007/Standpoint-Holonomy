#!/usr/bin/env python3
"""
Experiment F: Value Matrix Orthonormality Measurement

Computes ε_h = ||V_h^T V_h - Id_{d_v}||_F for each attention head.
This measures how close value matrices are to having orthonormal columns.
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def get_transformer_layers(model):
    """Get transformer layers from model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2
    raise ValueError(f"Unknown model architecture: {type(model)}")


def get_attn_module(layer):
    """Get attention module from layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn  # Llama
    if hasattr(layer, "attn"):
        return layer.attn  # GPT-2
    raise ValueError(f"Unknown layer type: {type(layer)}")


def extract_value_matrices(model, model_name):
    """Extract value matrices for all heads."""
    layers = get_transformer_layers(model)

    if model_name == "gpt2":
        n_layers = 12
        n_heads = 12
        d_model = 768
        d_head = 64
    elif model_name == "llama-7b":
        n_layers = 32
        n_heads = 32
        d_model = 4096
        d_head = 128
    else:
        raise ValueError(f"Unknown model: {model_name}")

    value_matrices = np.zeros((n_layers, n_heads, d_model, d_head), dtype=np.float32)

    for layer_idx in range(n_layers):
        attn = get_attn_module(layers[layer_idx])

        # Try different attribute names
        w_v = None
        if hasattr(attn, "v_proj"):
            w_v = attn.v_proj.weight.detach().cpu().float().numpy()
        elif hasattr(attn, "V"):
            w_v = attn.V.weight.detach().cpu().float().numpy()
        elif hasattr(attn, "c_attn"):
            # GPT-2: combined QKV
            c_attn_weight = attn.c_attn.weight.detach().cpu().float().numpy()
            split = d_model
            w_v = c_attn_weight[:, 2*split:3*split]

        if w_v is not None:
            # Reshape to (n_heads, d_model, d_head)
            if w_v.shape == (d_model, d_model):
                w_v = w_v.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
            elif w_v.shape == (n_heads * d_head, d_model):
                w_v = w_v.reshape(n_heads, d_head, d_model).transpose(0, 2, 1)
            elif w_v.shape == (d_model, n_heads * d_head):
                w_v = w_v.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
            value_matrices[layer_idx] = w_v

    return value_matrices


def compute_orthonormality(value_matrices):
    """
    Compute ε_h = ||V_h^T V_h - Id_{d_v}||_F for each head.

    Returns:
        epsilons: dict mapping (layer, head) -> epsilon value
    """
    n_layers, n_heads, d_model, d_head = value_matrices.shape
    epsilons = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            V_h = value_matrices[layer, head]  # (d_model, d_head)
            VtV = V_h.T @ V_h  # (d_head, d_head)
            identity = np.eye(d_head, dtype=np.float32)
            epsilon = float(np.linalg.norm(VtV - identity, 'fro'))
            epsilons[(layer, head)] = epsilon

    return epsilons


def main():
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"

    print("=" * 60)
    print(f"Value Matrix Orthonormality: {model_name}")
    print("=" * 60)

    # Load model
    print(f"\n[1/3] Loading {model_name}...")
    if model_name == "gpt2":
        hf_name = "gpt2"
    elif model_name == "llama-7b":
        hf_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Extract value matrices
    print("\n[2/3] Extracting value matrices...")
    value_matrices = extract_value_matrices(model, model_name)
    print(f"  Shape: {value_matrices.shape}")

    # Compute orthonormality
    print("\n[3/3] Computing orthonormality deviations...")
    epsilons = compute_orthonormality(value_matrices)

    # Statistics
    eps_values = list(epsilons.values())
    print(f"\n  Total heads: {len(eps_values)}")
    print(f"  Mean ε_h: {np.mean(eps_values):.6f}")
    print(f"  Std ε_h: {np.std(eps_values):.6f}")
    print(f"  Min ε_h: {np.min(eps_values):.6f}")
    print(f"  Max ε_h: {np.max(eps_values):.6f}")
    print(f"  Median ε_h: {np.median(eps_values):.6f}")

    # Per-layer statistics
    n_layers = value_matrices.shape[0]
    n_heads = value_matrices.shape[1]
    print(f"\n  Per-layer mean ε_h:")
    for layer in range(n_layers):
        layer_eps = [epsilons[(layer, h)] for h in range(n_heads)]
        print(f"    Layer {layer:2d}: {np.mean(layer_eps):.6f}")

    # Save results
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_model": value_matrices.shape[2],
        "d_head": value_matrices.shape[3],
        "mean_epsilon": float(np.mean(eps_values)),
        "std_epsilon": float(np.std(eps_values)),
        "min_epsilon": float(np.min(eps_values)),
        "max_epsilon": float(np.max(eps_values)),
        "median_epsilon": float(np.median(eps_values)),
        "per_layer_mean": {
            str(layer): float(np.mean([epsilons[(layer, h)] for h in range(n_heads)]))
            for layer in range(n_layers)
        },
        "per_head": {
            f"L{layer}_H{head}": float(epsilons[(layer, head)])
            for layer in range(n_layers)
            for head in range(n_heads)
        }
    }

    output_path = RESULTS_DIR / f"{model_name}_orthonormality.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
