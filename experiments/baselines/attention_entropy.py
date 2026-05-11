"""Attention entropy baseline: entropy of attention distributions."""
import numpy as np
import pandas as pd
from pathlib import Path

from experiments.config import CACHE_DIR, RESULTS_DIR


def compute_attention_entropy(attention: np.ndarray) -> np.ndarray:
    """Compute entropy of attention from event 0 to event 2.

    Args:
        attention: (n_layers, n_heads, n_events, n_events)

    Returns:
        entropy: (n_layers, n_heads)
    """
    # Attention from event 0 to event 2
    attn = attention[:, :, 2, 0]  # (n_layers, n_heads)
    # Clip to avoid log(0)
    attn = np.clip(attn, 1e-10, 1.0)
    entropy = -np.sum(attn * np.log(attn), axis=-1)
    return entropy


def run_attention_entropy(
    model_name: str,
    activations_path: Path,
    output_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Compute attention entropy for all test conversations."""
    data = np.load(activations_path, allow_pickle=True)

    results = []
    for key in data.files:
        parts = key.rsplit("/", 1)
        conv_id = parts[0]
        field = parts[1]

        if field != "attention" or "/test/" not in conv_id:
            continue

        attention = data[key]
        scenario = conv_id.split("/")[0]
        entropy = compute_attention_entropy(attention)

        for layer in range(entropy.shape[0]):
            results.append({
                "conversation_id": conv_id,
                "scenario": scenario,
                "layer": layer,
                "entropy_mean": entropy[layer].mean(),
                "entropy_std": entropy[layer].std(),
                "baseline": "attention_entropy",
            })

    df = pd.DataFrame(results)
    output_path = output_dir / f"{model_name}_entropy.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Entropy results: {len(df)} rows -> {output_path}")
    return df


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    run_attention_entropy(model_name, CACHE_DIR / model_name / "activations.npz")
