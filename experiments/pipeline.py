"""Full LCESA experiment pipeline."""
import sys
from pathlib import Path

from experiments.config import MODELS, DATA_DIR, CACHE_DIR, RESULTS_DIR
from experiments.stimuli.generate import generate_all_stimuli
from experiments.extraction.extract import extract_all
from experiments.grouping.head_grouping import run_head_grouping
from experiments.curvature.compute import run_curvature_computation
from experiments.baselines.linear_probing import run_linear_probing
from experiments.baselines.attention_entropy import run_attention_entropy
from experiments.baselines.cka import run_cka
from experiments.baselines.permutation import run_permutation_test
from experiments.stats.hypothesis_tests import run_all_tests
from experiments.visualization.plots import plot_curvature_heatmap, plot_curvature_boxplots


def run_pipeline(model_name: str, skip_extraction: bool = False):
    """Run full pipeline for one model."""
    config = MODELS[model_name]
    stimuli_path = DATA_DIR / "stimuli.json"
    activations_path = CACHE_DIR / model_name / "activations.npz"
    gamma_path = DATA_DIR / f"{model_name}_grouping.npz"

    print(f"\n{'='*60}")
    print(f"LCESA Pipeline: {model_name}")
    print(f"{'='*60}")

    # Step 1: Generate stimuli (shared across models)
    if not stimuli_path.exists():
        print("\n[1/8] Generating stimuli...")
        generate_all_stimuli()
    else:
        print("\n[1/8] Stimuli already exist, skipping.")

    # Step 2: Extract activations
    if not skip_extraction and not activations_path.exists():
        print("\n[2/8] Extracting activations...")
        extract_all(stimuli_path, model_name)
    else:
        print("\n[2/8] Activations already exist, skipping.")

    # Step 3: Head grouping
    if not gamma_path.exists():
        print("\n[3/8] Running head grouping...")
        run_head_grouping(model_name, activations_path)
    else:
        print("\n[3/8] Grouping already exists, skipping.")

    # Step 4: Curvature computation
    curvature_path = RESULTS_DIR / f"{model_name}_curvature.csv"
    if not curvature_path.exists():
        print("\n[4/8] Computing curvature...")
        run_curvature_computation(model_name, activations_path, gamma_path)
    else:
        print("\n[4/8] Curvature already computed, skipping.")

    # Step 5: Baselines
    print("\n[5/8] Running baselines...")
    run_linear_probing(model_name, activations_path)
    run_attention_entropy(model_name, activations_path)
    run_cka(model_name, activations_path)
    run_permutation_test(model_name, activations_path, gamma_path)

    # Step 5b: Ablation study
    ablation_path = RESULTS_DIR / f"{model_name}_ablation.csv"
    if not ablation_path.exists():
        print("\n[5b/8] Running ablation study...")
        from experiments.ablation import run_ablation_study
        run_ablation_study(model_name, activations_path, gamma_path)
    else:
        print("\n[5b/8] Ablation results already exist, skipping.")

    # Step 6: Hypothesis tests
    print("\n[6/8] Running hypothesis tests...")
    run_all_tests(model_name)

    # Step 7: Visualizations
    print("\n[7/8] Generating visualizations...")
    import pandas as pd
    df = pd.read_csv(curvature_path)
    plot_curvature_heatmap(df, model_name)
    plot_curvature_boxplots(df, model_name)

    print(f"\n{'='*60}")
    print(f"Pipeline complete for {model_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    run_pipeline(model)
