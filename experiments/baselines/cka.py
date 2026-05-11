"""CKA (Centered Kernel Alignment) baseline."""
import numpy as np
import pandas as pd
from pathlib import Path

from experiments.config import CACHE_DIR, RESULTS_DIR


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representations.

    CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    numerator = np.linalg.norm(X_centered.T @ Y_centered, "fro") ** 2
    denominator = (
        np.linalg.norm(X_centered.T @ X_centered, "fro")
        * np.linalg.norm(Y_centered.T @ Y_centered, "fro")
    )

    if denominator < 1e-10:
        return 0.0
    return numerator / denominator


def run_cka(
    model_name: str,
    activations_path: Path,
    output_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Compute CKA between assertion and observation residuals."""
    data = np.load(activations_path, allow_pickle=True)

    results = []
    for key in data.files:
        parts = key.rsplit("/", 1)
        conv_id = parts[0]
        field = parts[1]

        if field != "residuals" or "/test/" not in conv_id:
            continue

        residuals = data[key]  # (n_events, n_layers, d_model)
        scenario = conv_id.split("/")[0]

        # CKA between event 0 (assert) and event 2 (observe)
        for layer in range(residuals.shape[1]):
            cka_val = compute_cka(
                residuals[0, layer:layer+1, :],
                residuals[2, layer:layer+1, :],
            )
            results.append({
                "conversation_id": conv_id,
                "scenario": scenario,
                "layer": layer,
                "cka": cka_val,
                "baseline": "cka",
            })

    df = pd.DataFrame(results)
    output_path = output_dir / f"{model_name}_cka.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"CKA results: {len(df)} rows -> {output_path}")
    return df


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    run_cka(model_name, CACHE_DIR / model_name / "activations.npz")
