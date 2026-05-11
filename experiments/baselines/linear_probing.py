"""Linear probing baseline: train classifier on residuals to detect failure modes."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from experiments.config import LAYER_NAMES, SCENARIO_TYPES, CACHE_DIR, RESULTS_DIR, MODELS


def run_linear_probing(
    model_name: str,
    activations_path: Path,
    output_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Train linear probes per layer to detect failure vs. control."""
    config = MODELS[model_name]
    data = np.load(activations_path, allow_pickle=True)

    # Organize: extract residuals and labels
    X_by_layer = {l: [] for l in range(config.n_layers)}
    y_all = []

    for key in data.files:
        parts = key.rsplit("/", 1)
        conv_id = parts[0]
        field = parts[1]

        if field != "residuals" or "/test/" not in conv_id:
            continue

        residuals = data[key]  # (n_events, n_layers, d_model)
        scenario = conv_id.split("/")[0]

        # Use final event's residual
        final_residual = residuals[-1]  # (n_layers, d_model)

        for layer in range(config.n_layers):
            X_by_layer[layer].append(final_residual[layer])

        # Label: 0 for T1 (control), 1 for failure modes
        y_all.append(0 if scenario == "T1" else 1)

    y = np.array(y_all)

    # Train probes
    results = []
    for layer in range(config.n_layers):
        if not X_by_layer[layer]:
            continue
        X = np.array(X_by_layer[layer])
        X = StandardScaler().fit_transform(X)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, y, cv=5, scoring="f1")

        results.append({
            "model": model_name,
            "layer": layer,
            "f1_mean": scores.mean(),
            "f1_std": scores.std(),
            "baseline": "linear_probing",
        })

    df = pd.DataFrame(results)
    output_path = output_dir / f"{model_name}_probing.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Probing results: {len(df)} layers -> {output_path}")
    return df


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    run_linear_probing(model_name, CACHE_DIR / model_name / "activations.npz")
