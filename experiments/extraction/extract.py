"""
LCESA Activation Extraction
============================
Extract attention patterns, residual streams, and value matrices from
transformer models using TransformerLens.

Each stimulus conversation (5 events) is formatted as a multi-turn prompt,
tokenized, and run through the model with caching enabled. The resulting
activations are aggregated into per-conversation tensors and saved to disk.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer

from experiments.config import CACHE_DIR, MODELS, ModelConfig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config: ModelConfig) -> HookedTransformer:
    """Load a pretrained HookedTransformer from the given model configuration.

    Parameters
    ----------
    config : ModelConfig
        Model configuration including HuggingFace name, device, and dtype.

    Returns
    -------
    HookedTransformer
        The loaded model in eval mode.
    """
    dtype = getattr(torch, config.dtype)
    print(f"Loading model {config.name} ({config.hf_name}) on {config.device} ...")
    model = HookedTransformer.from_pretrained(
        config.hf_name,
        device=config.device,
        dtype=dtype,
    )
    model.eval()
    print(f"  Model loaded: {model.cfg.n_layers} layers, "
          f"{model.cfg.n_heads} heads, d_model={model.cfg.d_model}")
    return model


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _format_event(event: dict) -> str:
    """Format a single conversation event into a string suitable for tokenization.

    User events are wrapped with ``[INST]`` / ``[/INST]`` markers (matching the
    Llama-2-chat convention used by most chat models supported here).  Assistant
    events are emitted verbatim since they represent model output.
    """
    if event["role"] == "user":
        return f"[INST] {event['content']} [/INST] "
    else:
        return event["content"]


def _tokenize_events(
    model: HookedTransformer,
    events: List[dict],
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Tokenize all events in a conversation and track per-event token ranges.

    Parameters
    ----------
    model : HookedTransformer
        The model whose tokenizer to use.
    events : list of dict
        The conversation events (each with ``role`` and ``content`` keys).

    Returns
    -------
    tokens : torch.Tensor
        Shape ``(1, n_tokens)`` — the full token sequence for the conversation.
    event_token_ranges : list of (int, int)
        ``(start, end)`` token indices for each event.  ``end`` is exclusive.
    """
    all_token_ids: List[int] = []
    event_token_ranges: List[Tuple[int, int]] = []

    for event in events:
        text = _format_event(event)
        # prepend_bos=False because we handle BOS once for the whole conversation
        event_tokens = model.to_tokens(text, prepend_bos=False)
        # event_tokens shape: (1, n_event_tokens)
        event_token_ids = event_tokens[0].tolist()

        start = len(all_token_ids)
        all_token_ids.extend(event_token_ids)
        end = len(all_token_ids)
        event_token_ranges.append((start, end))

    tokens = torch.tensor([all_token_ids], device=model.cfg.device)
    return tokens, event_token_ranges


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_attention_for_conversation(
    model: HookedTransformer,
    conversation: dict,
    model_config: ModelConfig,
) -> Dict[str, np.ndarray]:
    """Extract activations for a single 5-event conversation.

    Parameters
    ----------
    model : HookedTransformer
        The loaded model.
    conversation : dict
        Must contain an ``events`` list of 5 dicts with ``role`` and ``content``.
    model_config : ModelConfig
        Model dimensions (used for sizing output arrays).

    Returns
    -------
    dict with keys:
        ``attention``       — (n_layers, n_heads, n_events, n_events) float32
        ``residuals``       — (n_events, n_layers, d_model) float32
        ``value_matrices``  — (n_layers, n_heads, d_model, d_head) float32
        ``event_token_ranges`` — list of (start, end) tuples
    """
    events = conversation["events"]
    n_events = len(events)
    n_layers = model_config.n_layers
    n_heads = model_config.n_heads
    d_model = model_config.d_model
    d_head = model_config.d_head

    # -- tokenize ----------------------------------------------------------
    tokens, event_token_ranges = _tokenize_events(model, events)

    # -- build names_filter for the cache ----------------------------------
    def names_filter(name: str) -> bool:
        return "hook_resid_post" in name or "hook_pattern" in name

    # -- forward pass with cache -------------------------------------------
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names_filter)

    # -- extract attention patterns ----------------------------------------
    # shape target: (n_layers, n_heads, n_events, n_events)
    attention = np.zeros((n_layers, n_heads, n_events, n_events), dtype=np.float32)

    for layer in range(n_layers):
        # cache key for attention pattern
        pattern_key = f"blocks.{layer}.attn.hook_pattern"
        if pattern_key not in cache:
            # try alternative naming conventions
            alt_keys = [
                f"blocks.{layer}.attn.hook_attn",
                f"blocks.{layer}.attn.attn",
            ]
            found = False
            for alt in alt_keys:
                if alt in cache:
                    pattern_key = alt
                    found = True
                    break
            if not found:
                raise KeyError(
                    f"Could not find attention pattern in cache for layer {layer}. "
                    f"Available keys: {[k for k in cache.keys() if 'attn' in k]}"
                )

        # pattern shape: (1, n_heads, seq_len_q, seq_len_k)
        pattern = cache[pattern_key][0].cpu().numpy()  # (n_heads, seq_q, seq_k)

        for i in range(n_events):
            q_start, q_end = event_token_ranges[i]
            for j in range(n_events):
                k_start, k_end = event_token_ranges[j]
                # mean attention from tokens of event i to tokens of event j
                attn_block = pattern[:, q_start:q_end, k_start:k_end]
                if attn_block.size > 0:
                    attention[:, :, i, j] = attn_block.mean(axis=(1, 2))

    # -- extract residual streams ------------------------------------------
    # shape target: (n_events, n_layers, d_model)
    residuals = np.zeros((n_events, n_layers, d_model), dtype=np.float32)

    for layer in range(n_layers):
        resid_key = f"blocks.{layer}.hook_resid_post"
        if resid_key not in cache:
            raise KeyError(
                f"Could not find residual stream in cache for layer {layer}. "
                f"Available keys: {[k for k in cache.keys() if 'resid' in k]}"
            )
        # resid shape: (1, seq_len, d_model)
        resid = cache[resid_key][0].cpu().numpy()

        for i in range(n_events):
            # take the residual at the last token of each event
            _, event_end = event_token_ranges[i]
            last_token_idx = event_end - 1
            residuals[i, layer, :] = resid[last_token_idx, :]

    # -- extract value matrices from model weights -------------------------
    # shape: (n_layers, n_heads, d_model, d_head)
    value_matrices = np.zeros((n_layers, n_heads, d_model, d_head), dtype=np.float32)
    for layer in range(n_layers):
        w_v = model.blocks[layer].attn.W_V.detach().cpu().numpy()
        value_matrices[layer, :, :, :] = w_v

    return {
        "attention": attention,
        "residuals": residuals,
        "value_matrices": value_matrices,
        "event_token_ranges": event_token_ranges,
    }


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_all(
    stimuli_path: Path,
    model_name: str,
    output_dir: Path = CACHE_DIR,
) -> Path:
    """Extract activations for every conversation in the stimuli file.

    Parameters
    ----------
    stimuli_path : Path
        Path to ``data/stimuli.json``.
    model_name : str
        Key into ``MODELS`` dict (e.g. ``"gpt2"``, ``"llama-7b"``).
    output_dir : Path, optional
        Directory under which to save the output ``.npz`` file.
        Defaults to ``CACHE_DIR`` from config.

    Returns
    -------
    Path
        Path to the saved ``activations.npz`` file.
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
        )

    model_config = MODELS[model_name]
    model = load_model(model_config)

    print(f"Loading stimuli from {stimuli_path} ...")
    with open(stimuli_path, "r") as f:
        stimuli = json.load(f)

    # Collect all arrays into a flat dict keyed by "{scenario}/{split}/{idx}/{field}"
    arrays: Dict[str, np.ndarray] = {}
    total_conversations = 0

    for scenario in sorted(stimuli.keys()):
        scenario_data = stimuli[scenario]
        for split in ("grouping", "test"):
            conversations = scenario_data.get(split, [])
            if not conversations:
                continue
            print(f"  Extracting {scenario}/{split}: {len(conversations)} conversations ...")
            for idx, conv in enumerate(conversations):
                result = extract_attention_for_conversation(model, conv, model_config)
                prefix = f"{scenario}/{split}/{idx}"
                for field_name in ("attention", "residuals", "value_matrices"):
                    arrays[f"{prefix}/{field_name}"] = result[field_name]
                # store event_token_ranges as a structured array of (start, end) pairs
                ranges = result["event_token_ranges"]
                arrays[f"{prefix}/event_token_ranges"] = np.array(
                    ranges, dtype=np.int64
                )
                total_conversations += 1
                if total_conversations % 10 == 0:
                    print(f"    ... processed {total_conversations} conversations")

    # -- save ---------------------------------------------------------------
    save_dir = output_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "activations.npz"

    print(f"Saving {len(arrays)} arrays ({total_conversations} conversations) "
          f"to {save_path} ...")
    np.savez_compressed(save_path, **arrays)
    print(f"Done. Output saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    stimuli_path = Path("data/stimuli.json")
    extract_all(stimuli_path, model_name)
