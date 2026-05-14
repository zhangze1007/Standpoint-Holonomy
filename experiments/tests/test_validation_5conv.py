"""
5-Conversation Validation Test
==============================
Verifies extraction speed, storage size, and array shapes after
Phase 1 vectorization optimization.

Runs GPT-2 on CPU (no GPU required) with 5 conversations (1 per scenario).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.config import MODELS
from experiments.extraction.extract import (
    extract_attention_for_conversation,
    extract_value_matrices,
    load_model,
)


def main():
    # ------------------------------------------------------------------
    # 1. Load model (GPT-2, CPU)
    # ------------------------------------------------------------------
    model_config = MODELS["gpt2"]
    model_config.device = "cpu"
    model, tokenizer = load_model(model_config)

    # ------------------------------------------------------------------
    # 2. Load stimuli, pick 1 conversation per scenario
    # ------------------------------------------------------------------
    stimuli_path = ROOT / "data" / "stimuli.json"
    with open(stimuli_path) as f:
        stimuli = json.load(f)

    test_convs = {}
    for scenario in sorted(stimuli.keys()):
        convs = stimuli[scenario]["grouping"]
        if convs:
            test_convs[scenario] = convs[0]

    print(f"\n{'='*60}")
    print(f"Validation: {len(test_convs)} conversations, GPT-2 (CPU)")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 3. Extract value matrices once
    # ------------------------------------------------------------------
    values_path = ROOT / "cache" / "gpt2" / "value_matrices_test.npz"
    t0 = time.time()
    vm = extract_value_matrices(model, model_config, values_path)
    vm_time = time.time() - t0
    vm_size_mb = values_path.stat().st_size / 1e6
    print(f"  Value matrices shape: {vm.shape}")
    print(f"  Value matrices time: {vm_time:.1f}s")
    print(f"  Value matrices size: {vm_size_mb:.1f} MB\n")

    # ------------------------------------------------------------------
    # 4. Extract activations for each test conversation
    # ------------------------------------------------------------------
    results = []
    for scenario, conv in test_convs.items():
        print(f"--- {scenario} ---")
        t0 = time.time()
        result = extract_attention_for_conversation(
            model, tokenizer, conv, model_config, extract_values=False,
        )
        elapsed = time.time() - t0

        attn = result["attention"]
        resid = result["residuals"]
        ranges = result["event_token_ranges"]

        # Compute per-conversation storage
        attn_bytes = attn.nbytes
        resid_bytes = resid.nbytes
        ranges_bytes = ranges.nbytes
        total_bytes = attn_bytes + resid_bytes + ranges_bytes
        total_mb = total_bytes / 1e6

        # Save to temp file to measure compressed size
        tmp_path = ROOT / "cache" / "gpt2" / f"_val_{scenario}.npz"
        np.savez_compressed(tmp_path, attention=attn, residuals=resid, event_token_ranges=ranges)
        compressed_mb = tmp_path.stat().st_size / 1e6

        results.append({
            "scenario": scenario,
            "time_s": elapsed,
            "attn_shape": attn.shape,
            "resid_shape": resid.shape,
            "ranges_shape": ranges.shape,
            "raw_mb": total_mb,
            "compressed_mb": compressed_mb,
        })

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Attention shape: {attn.shape} ({attn_bytes/1e3:.0f} KB)")
        print(f"  Residuals shape: {resid.shape} ({resid_bytes/1e3:.0f} KB)")
        print(f"  Ranges shape:    {ranges.shape}")
        print(f"  Raw size: {total_mb:.2f} MB, Compressed: {compressed_mb:.2f} MB")

        # Clean up temp file
        tmp_path.unlink()

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    times = [r["time_s"] for r in results]
    raw_sizes = [r["raw_mb"] for r in results]
    comp_sizes = [r["compressed_mb"] for r in results]

    print(f"  Conversations tested: {len(results)}")
    print(f"  Avg time per conv:    {np.mean(times):.2f}s (std: {np.std(times):.2f}s)")
    print(f"  Avg raw size/conv:    {np.mean(raw_sizes):.2f} MB")
    print(f"  Avg compressed/conv:  {np.mean(comp_sizes):.2f} MB")
    print(f"  Value matrices:       {vm_size_mb:.1f} MB (shared, one-time)")

    # Project to 225 conversations
    n_total = 225  # 5 scenarios x (15 grouping + 30 test)
    projected_conv_mb = np.mean(comp_sizes) * n_total
    projected_total_mb = projected_conv_mb + vm_size_mb
    print(f"\n  Projected for {n_total} convos:")
    print(f"    Per-conversation data: {projected_conv_mb:.0f} MB")
    print(f"    Value matrices:        {vm_size_mb:.0f} MB")
    print(f"    Total:                 {projected_total_mb:.0f} MB ({projected_total_mb/1024:.1f} GB)")

    # Check against estimates from optimization analysis
    print(f"\n  vs. Optimization Analysis Estimates:")
    print(f"    Estimated per conv: 2.7 MB, Actual: {np.mean(comp_sizes):.2f} MB")
    print(f"    Estimated total:    2.6 GB, Projected: {projected_total_mb/1024:.1f} GB")

    # Speed projection (CPU is ~10-20x slower than GPU)
    avg_time = np.mean(times)
    print(f"\n  Speed (CPU, GPT-2):")
    print(f"    Per conversation: {avg_time:.2f}s")
    print(f"    Projected {n_total} convos (CPU): {avg_time * n_total / 60:.1f} min")
    print(f"    Projected {n_total} convos (GPU, ~5x faster): {avg_time * n_total / 60 / 5:.1f} min")

    # ------------------------------------------------------------------
    # 6. Verify array correctness
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CORRECTNESS CHECKS")
    print(f"{'='*60}")

    all_pass = True
    for r in results:
        s = r["scenario"]
        attn = r["attn_shape"]
        resid = r["resid_shape"]

        # Attention should be (n_layers, n_heads, n_events, n_events)
        if attn == (12, 12, 5, 5):
            print(f"  [{s}] Attention shape OK: {attn}")
        else:
            print(f"  [{s}] Attention shape UNEXPECTED: {attn} (expected (12, 12, 5, 5))")
            all_pass = False

        # Residuals should be (n_events, n_layers, d_model)
        if resid == (5, 12, 768):
            print(f"  [{s}] Residuals shape OK: {resid}")
        else:
            print(f"  [{s}] Residuals shape UNEXPECTED: {resid} (expected (5, 12, 768))")
            all_pass = False

    if all_pass:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  SOME CHECKS FAILED")
        sys.exit(1)

    # Cleanup
    values_path.unlink(missing_ok=True)
    print(f"\nDone. Cleaned up test files.")


if __name__ == "__main__":
    main()
