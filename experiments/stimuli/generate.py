"""
LCESA Experiment Stimulus Generator
=====================================
Expands conversation templates into grouped/test stimulus sets and
persists them as JSON for downstream pipeline stages.
"""

import copy
import json
import random
from pathlib import Path

from experiments.config import DATA_DIR, N_GROUPING, N_TEST, SCENARIO_TYPES
from experiments.stimuli.templates import ALL_TEMPLATES


def expand_template(template: dict, variant_id: str) -> dict:
    """Deep-copy *template* and attach a unique *variant_id*."""
    variant = copy.deepcopy(template)
    variant["variant_id"] = variant_id
    return variant


def generate_scenario_stimuli(
    scenario: str,
    templates: list[dict],
    n_grouping: int = N_GROUPING,
    n_test: int = N_TEST,
) -> dict[str, list[dict]]:
    """Generate grouping and test variants for a single scenario type.

    Variants are distributed evenly across the available templates, shuffled
    with a fixed seed for reproducibility, and split into grouping / test
    partitions.

    Returns
    -------
    dict with keys "grouping" and "test", each a list of variant dicts.
    """
    n_total = n_grouping + n_test
    n_templates = len(templates)

    # Build an equal distribution of variants across templates
    variants: list[dict] = []
    for i in range(n_total):
        tpl_idx = i % n_templates
        variant_id = f"{scenario}_v{i:03d}"
        variants.append(expand_template(templates[tpl_idx], variant_id))

    # Reproducible shuffle
    random.seed(42)
    random.shuffle(variants)

    return {
        "grouping": variants[:n_grouping],
        "test": variants[n_grouping:],
    }


def generate_all_stimuli(output_dir: Path = DATA_DIR) -> None:
    """Generate stimuli for every scenario and write to *output_dir*/stimuli.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stimuli: dict[str, dict[str, list[dict]]] = {}
    total_conversations = 0

    for scenario in SCENARIO_TYPES:
        templates = ALL_TEMPLATES[scenario]
        stimuli = generate_scenario_stimuli(scenario, templates)
        all_stimuli[scenario] = stimuli
        n = len(stimuli["grouping"]) + len(stimuli["test"])
        total_conversations += n
        print(f"  {scenario}: {n} conversations "
              f"({len(stimuli['grouping'])} grouping + {len(stimuli['test'])} test)")

    out_path = output_dir / "stimuli.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_stimuli, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {total_conversations} conversations")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    generate_all_stimuli()
