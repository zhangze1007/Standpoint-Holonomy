"""CKA (Centered Kernel Alignment) baseline."""
import numpy as np
import pandas as pd
from pathlib import Path

from experiments.config import CACHE_DIR, RESULTS_DIR


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representations.

    CKA(X, Y) = ||X @ Y^T||_F^2 / (||X @ X^T||_F * ||Y @ Y^T||_F)

    X: (n_samples_x, d_model), Y: (n_samples_y, d_model)
    """
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Gram matrices in sample space: (n_x, n_y), (n_x, n_x), (n_y, n_y)
    XY = X_centered @ Y_centered.T
    XX = X_centered @ X_centered.T
    YY = Y_centered @ Y_centered.T

    numerator = np.linalg.norm(XY, "fro") ** 2
    denominator = np.linalg.norm(XX, "fro") * np.linalg.norm(YY, "fro")

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

        # CKA between assertion events (indices 0,1) and observation events (indices 2,3)
        # Use all events as a batch so centering has enough samples to be meaningful.
        n_events = residuals.shape[0]
        mid = n_events // 2
        for layer in range(residuals.shape[1]):
            X = residuals[:mid, layer, :]   # assertion events: (mid, d_model)
            Y = residuals[mid:, layer, :]   # observation events: (n-mid, d_model)
            # If either side has only 1 sample, CKA centering zeros it out — skip
            if X.shape[0] < 2 or Y.shape[0] < 2:
                continue
            cka_val = compute_cka(X, Y)
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
