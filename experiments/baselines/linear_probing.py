"""Linear probing baseline: train classifier on residuals to detect failure modes."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.config import LAYER_NAMES, SCENARIO_TYPES, CACHE_DIR, RESULTS_DIR, MODELS


def run_linear_probing(
    model_name: str,
    activations_path: Path,
    output_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Train linear probes per layer: binary (T1 vs rest) AND 5-way multi-class."""
    config = MODELS[model_name]
    data = np.load(activations_path, allow_pickle=True)

    # Organize: extract residuals and labels
    X_by_layer = {l: [] for l in range(config.n_layers)}
    y_binary = []
    y_multi = []
    scenario_to_int = {s: i for i, s in enumerate(SCENARIO_TYPES)}

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

        y_binary.append(0 if scenario == "T1" else 1)
        y_multi.append(scenario_to_int[scenario])

    y_bin = np.array(y_binary)
    y_mul = np.array(y_multi)
    n_classes = len(np.unique(y_mul))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Train probes
    results = []
    for layer in range(config.n_layers):
        if not X_by_layer[layer]:
            continue
        X = np.array(X_by_layer[layer])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])

        # Binary probe: T1 vs rest
        bin_scores = cross_val_score(pipe, X, y_bin, cv=cv, scoring="f1")
        results.append({
            "model": model_name,
            "layer": layer,
            "probe_type": "binary",
            "f1_mean": bin_scores.mean(),
            "f1_std": bin_scores.std(),
            "baseline": "linear_probing",
        })

        # Multi-class probe: 5-way scenario classification (macro F1)
        multi_scores = cross_val_score(pipe, X, y_mul, cv=cv, scoring="f1_macro")
        results.append({
            "model": model_name,
            "layer": layer,
            "probe_type": "multiclass",
            "f1_mean": multi_scores.mean(),
            "f1_std": multi_scores.std(),
            "baseline": "linear_probing",
        })

        # PCA-reduced probe: 10 components — checks if separability is genuine
        # or just a dimensionality artifact (768 dims >> 150 samples)
        pipe_pca = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=10, random_state=42)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        pca_scores = cross_val_score(pipe_pca, X, y_mul, cv=cv, scoring="f1_macro")
        results.append({
            "model": model_name,
            "layer": layer,
            "probe_type": "multiclass_pca10",
            "f1_mean": pca_scores.mean(),
            "f1_std": pca_scores.std(),
            "baseline": "linear_probing",
        })

    df = pd.DataFrame(results)
    output_path = output_dir / f"{model_name}_probing.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Probing results: {len(df)} rows -> {output_path}")
    return df


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    run_linear_probing(model_name, CACHE_DIR / model_name / "activations.npz")
