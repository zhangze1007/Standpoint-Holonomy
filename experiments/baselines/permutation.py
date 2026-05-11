"""Scrambled head grouping permutation test."""
import numpy as np
import pandas as pd
from pathlib import Path

from experiments.config import LAYER_NAMES, MODELS, CACHE_DIR, RESULTS_DIR


def run_permutation_test(
    model_name: str,
    activations_path: Path,
    gamma_path: Path,
    n_permutations: int = 1000,
    output_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Test block-specificity under random head assignments."""
    config = MODELS[model_name]
    data = np.load(activations_path, allow_pickle=True)
    grouping = np.load(gamma_path)
    gamma_real = grouping["gamma"]

    # Get a sample T2 test conversation
    test_key = None
    for key in data.files:
        if "T2/test/" in key and key.endswith("/attention"):
            test_key = key
            break

    if test_key is None:
        print("No T2 test conversation found")
        return pd.DataFrame()

    attn = data[test_key]

    results = []
    np.random.seed(42)

    for b in range(n_permutations):
        # Random permutation preserving layer sizes
        gamma_perm = np.random.permutation(gamma_real)

        # Compute mean attention per layer with permuted gamma
        for k_idx, k_name in enumerate(LAYER_NAMES):
            head_mask = gamma_perm == k_idx
            if not np.any(head_mask):
                continue
            mean_attn = attn[:, head_mask, 2, 0].mean()
            results.append({
                "permutation": b,
                "layer": k_name,
                "mean_attention": mean_attn,
            })

    df = pd.DataFrame(results)
    output_path = output_dir / f"{model_name}_permutation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Permutation test: {n_permutations} permutations -> {output_path}")
    return df


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    run_permutation_test(
        model_name,
        CACHE_DIR / model_name / "activations.npz",
        CACHE_DIR / model_name / f"{model_name}_grouping.npz",
    )
