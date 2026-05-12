"""
LCESA Activation Extraction
============================
Extract attention patterns, residual streams, and value matrices from
transformer models using HuggingFace Transformers with device_map="auto"
for CPU offloading (supports Llama-7b on T4 15GB VRAM).

Each stimulus conversation (5 events) is formatted as a multi-turn prompt,
tokenized, and run through the model with output_attentions=True. Residual
streams are captured via forward hooks. Results are aggregated into per-
conversation tensors and saved to disk.
"""

import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from experiments.config import CACHE_DIR, MODELS, ModelConfig


# ---------------------------------------------------------------------------
# Model loading (HuggingFace with device_map for CPU offloading)
# ---------------------------------------------------------------------------

def load_model(config: ModelConfig):
    """Load a pretrained model with device_map='auto' for CPU offloading.

    Returns (model, tokenizer) tuple.
    """
    print(f"Loading model {config.name} ({config.hf_name}) ...")

    # Auto-force 4-bit for large models to prevent OOM on T4
    use_4bit = config.load_in_4bit or ("llama" in config.name.lower())
    if use_4bit and not config.load_in_4bit:
        print("  Auto-enabling 4-bit quantization for large model")

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "eager",  # sdpa doesn't support output_attentions
    }

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config
        print("  Using 4-bit quantization with CPU offloading")
    else:
        dtype = getattr(torch, config.dtype)
        model_kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(config.hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.hf_name, **model_kwargs)
    model.eval()

    # Report device map
    if hasattr(model, "hf_device_map"):
        devices = set(model.hf_device_map.values())
        print(f"  Model loaded across devices: {devices}")
    n_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", "?"))
    n_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", "?"))
    d_model = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", "?"))
    print(f"  Architecture: {n_layers} layers, {n_heads} heads, d_model={d_model}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Model architecture helpers
# ---------------------------------------------------------------------------

def _get_transformer_layers(model):
    """Get the list of transformer decoder layers, handling different architectures."""
    # Llama / Mistral / most modern models
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # GPT-Neo / GPT-J
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return model.transformer.layers
    # OPT
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise ValueError(f"Cannot find transformer layers in model type {type(model).__name__}")


def _get_attn_module(layer):
    """Get the attention module from a transformer layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn  # Llama, Mistral
    if hasattr(layer, "attn"):
        return layer.attn  # GPT-2
    if hasattr(layer, "attention"):
        return layer.attention  # OPT
    raise ValueError(f"Cannot find attention module in layer type {type(layer).__name__}")


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _format_event(event: dict) -> str:
    """Format a single conversation event for tokenization."""
    if event["role"] == "user":
        return f"[INST] {event['content']} [/INST] "
    else:
        return event["content"]


def _tokenize_events(tokenizer, events: List[dict], device=None):
    """Tokenize all events and track per-event token ranges."""
    all_token_ids: List[int] = []
    event_token_ranges: List[Tuple[int, int]] = []

    for event in events:
        text = _format_event(event)
        event_tokens = tokenizer.encode(text, add_special_tokens=False)
        start = len(all_token_ids)
        all_token_ids.extend(event_tokens)
        end = len(all_token_ids)
        event_token_ranges.append((start, end))

    tokens = torch.tensor([all_token_ids], device=device if device else "cpu")
    return tokens, event_token_ranges


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_attention_for_conversation(
    model,
    tokenizer,
    conversation: dict,
    model_config: ModelConfig,
) -> Dict[str, np.ndarray]:
    """Extract activations for a single 5-event conversation."""
    events = conversation["events"]
    n_events = len(events)
    n_layers = model_config.n_layers
    n_heads = model_config.n_heads
    d_model = model_config.d_model
    d_head = model_config.d_head

    # -- tokenize -----------------------------------------------------------
    # Place tokens on same device as the embedding layer (first model parameter)
    embed_device = next(model.parameters()).device
    tokens, event_token_ranges = _tokenize_events(tokenizer, events, device=embed_device)

    # -- pre-allocate output arrays (CPU only) -----------------------------
    attention = np.zeros((n_layers, n_heads, n_events, n_events), dtype=np.float32)
    residuals = np.zeros((n_events, n_layers, d_model), dtype=np.float32)

    # -- register hooks for residual streams -------------------------------
    layers = _get_transformer_layers(model)
    hook_handles = []

    def _make_resid_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # output is the layer output tensor: (batch, seq_len, d_model)
            # or a tuple where first element is the output tensor
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Extract last-token per event immediately to CPU
            for i in range(n_events):
                _, event_end = event_token_ranges[i]
                residuals[i, layer_idx, :] = hidden[0, event_end - 1, :].detach().cpu().float().numpy()
        return hook_fn

    for layer_idx in range(min(n_layers, len(layers))):
        handle = layers[layer_idx].register_forward_hook(_make_resid_hook(layer_idx))
        hook_handles.append(handle)

    # -- forward pass with output_attentions=True --------------------------
    try:
        with torch.no_grad():
            outputs = model(
                tokens,
                output_attentions=True,
                use_cache=False,
            )

        # -- extract attention patterns from model output ------------------
        # outputs.attentions is a tuple of (batch, heads, seq, seq) per layer
        if outputs.attentions is not None:
            for layer_idx, attn_weights in enumerate(outputs.attentions):
                if layer_idx >= n_layers:
                    break
                # attn_weights shape: (batch=1, n_heads, seq_q, seq_k)
                pat = attn_weights[0]  # (n_heads, seq_q, seq_k)
                for i in range(n_events):
                    q_s, q_e = event_token_ranges[i]
                    for j in range(n_events):
                        k_s, k_e = event_token_ranges[j]
                        block = pat[:, q_s:q_e, k_s:k_e]
                        if block.numel() > 0:
                            attention[layer_idx, :, i, j] = block.mean(dim=(1, 2)).cpu().float().numpy()
                del attn_weights
            del outputs.attentions

    finally:
        # Remove all hooks
        for handle in hook_handles:
            handle.remove()
        del hook_handles

    # -- free tokens and GPU memory ----------------------------------------
    del tokens, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- extract value matrices from model weights -------------------------
    value_matrices = np.zeros((n_layers, n_heads, d_model, d_head), dtype=np.float32)
    for layer_idx in range(min(n_layers, len(layers))):
        attn = _get_attn_module(layers[layer_idx])
        # Get V projection weight
        if hasattr(attn, "v_proj"):
            w_v = attn.v_proj.weight.detach().cpu().float().numpy()  # (d_model, d_model) or (d_model, n_heads*d_head)
        elif hasattr(attn, "V"):
            w_v = attn.V.weight.detach().cpu().float().numpy()
        else:
            # Try common patterns
            for attr_name in ["v_proj", "V", "value"]:
                if hasattr(attn, attr_name):
                    proj = getattr(attn, attr_name)
                    if hasattr(proj, "weight"):
                        w_v = proj.weight.detach().cpu().float().numpy()
                        break
            else:
                # Fallback: skip value matrices for this layer
                continue

        # Reshape to (n_heads, d_model, d_head) if needed
        if w_v.shape == (d_model, d_model):
            w_v = w_v.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        elif w_v.shape == (n_heads * d_head, d_model):
            w_v = w_v.reshape(n_heads, d_head, d_model).transpose(0, 2, 1)
        value_matrices[layer_idx, :, :, :] = w_v

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
    """Extract activations for every conversation in the stimuli file."""
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
        )

    model_config = MODELS[model_name]
    model, tokenizer = load_model(model_config)

    print(f"Loading stimuli from {stimuli_path} ...")
    with open(stimuli_path, "r") as f:
        stimuli = json.load(f)

    save_dir = output_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "activations.npz"

    all_arrays: Dict[str, np.ndarray] = {}
    total_conversations = 0
    batch_count = 0

    for scenario in sorted(stimuli.keys()):
        scenario_data = stimuli[scenario]
        for split in ("grouping", "test"):
            conversations = scenario_data.get(split, [])
            if not conversations:
                continue
            batch_arrays: Dict[str, np.ndarray] = {}
            print(f"  Extracting {scenario}/{split}: {len(conversations)} conversations ...")
            for idx, conv in enumerate(conversations):
                result = extract_attention_for_conversation(model, tokenizer, conv, model_config)
                prefix = f"{scenario}/{split}/{idx}"
                for field_name in ("attention", "residuals", "value_matrices"):
                    batch_arrays[f"{prefix}/{field_name}"] = result[field_name]
                ranges = result["event_token_ranges"]
                batch_arrays[f"{prefix}/event_token_ranges"] = np.array(
                    ranges, dtype=np.int64
                )
                del result
                total_conversations += 1
                if total_conversations % 10 == 0:
                    print(f"    ... processed {total_conversations} conversations")
                # Per-conversation GPU cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save this batch to a temp file
            batch_path = save_dir / f"_batch_{batch_count}.npz"
            np.savez_compressed(batch_path, **batch_arrays)
            batch_count += 1
            del batch_arrays
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Merge all batch files into one final file
    print(f"Merging {batch_count} batch files ...")
    for i in range(batch_count):
        batch_path = save_dir / f"_batch_{i}.npz"
        with np.load(batch_path) as data:
            for key in data:
                all_arrays[key] = data[key]
        batch_path.unlink()

    print(f"Saving {len(all_arrays)} arrays ({total_conversations} conversations) "
          f"to {save_path} ...")
    np.savez_compressed(save_path, **all_arrays)
    print(f"Done. Output saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    stimuli_path = Path("data/stimuli.json")
    extract_all(stimuli_path, model_name)
