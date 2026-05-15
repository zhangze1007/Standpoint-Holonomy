# LCESA Pipeline Runbook: Lessons Learned and Pre-Flight Checklist

> **MANDATORY**: Before starting ANY new pipeline run, model change, or infrastructure setup, read this document end-to-end. Every bug here cost real money and time.

---

## Pre-Flight Checklist (Before Any Pipeline Run)

- [ ] **Estimate compute cost**: `complexity = d_model^2 * n_layers * n_conversations`
  - d_model ≤ 768 (GPT-2): CPU fine, minutes
  - d_model ≥ 4096 (Llama-7b): GPU required, hours
  - d_model ≥ 5120 (Llama-13b): GPU required, budget accordingly
- [ ] **Run a single-layer benchmark first** before committing to a full run
- [ ] **Verify GPU memory**: `nvidia-smi` — need ≥16GB for 7B models (FP16)
- [ ] **Verify HF auth**: `python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami())"`
- [ ] **Check disk space**: activations.npz ≈ 1.4GB for Llama-7b, value_matrices ≈ 1GB
- [ ] **Use `nohup` for long runs**: `nohup python3 -m experiments.pipeline llama-7b > pipeline.log 2>&1 &`
- [ ] **Never type in the terminal while pipeline is running** — arrow keys and random keystrokes corrupt files

---

## Stage 1: Model Loading and Infrastructure

### Bug 1: OOM on small VRAM GPUs
**Symptom**: CUDA OOM crash during model loading.
**Root cause**: TransformerLens loads entire model into GPU at once.
**Fix**: Use HuggingFace Transformers with `device_map="auto"` for CPU/GPU layer offloading. Add `gc.collect()` and `torch.cuda.empty_cache()` after each conversation.
**File**: `experiments/extraction/extract.py`
**When to worry**: VRAM < 16GB for 7B models.

### Bug 2: 4-bit quantization was a no-op
**Symptom**: `load_in_4bit=True` in config but model loads in full precision.
**Root cause**: `load_model()` never read the `load_in_4bit` field.
**Fix**: Wired up `BitsAndBytesConfig` construction in `load_model()`.
**File**: `experiments/extraction/extract.py:41-56`

### Bug 3: SDPA silently disables attention output
**Symptom**: `output_attentions=True` but all attention arrays are None/empty.
**Root cause**: PyTorch SDPA backend does not support returning attention weights.
**Fix**: Force `attn_implementation="eager"` in model loading kwargs.
**File**: `experiments/extraction/extract.py:47`
**Critical**: This is silent — no error, just empty data. Always verify attention output is non-None after first extraction.

---

## Stage 2: Activation Extraction

### Bug 4: GPT-2 value matrices extracted as zeros
**Symptom**: Value matrices are all zeros for GPT-2 (works fine for Llama).
**Root cause**: GPT-2 uses combined QKV projection `c_attn` (shape `d_model, 3*d_model`), not separate `v_proj` like Llama.
**Fix**: Added `c_attn` branch that reads last third: `c_attn_weight[:, 2*split:3*split]`. Multiple fallback attribute lookups (`v_proj`, `V`, `value`).
**File**: `experiments/extraction/extract.py:273-300`
**When to worry**: Any model that uses combined QKV (GPT-2, GPT-3, etc.)

### Bug 5: Value matrices stored per-conversation (2GB waste)
**Symptom**: `activations.npz` is 2GB+ larger than expected.
**Root cause**: Value matrices are model weights (constant) but were saved once per conversation.
**Fix**: Extract once, save to top-level `"value_matrices"` key. Deduplicate everywhere.
**File**: `experiments/extraction/extract.py:247-306`
**Lesson**: Model weights are constants. Extract once, reuse everywhere.

### Bug 6: Top-level npz keys crash downstream modules
**Symptom**: `KeyError` or split-on-`/` failures when loading `activations.npz`.
**Root cause**: Keys like `"value_matrices"` have no `/`, so `rsplit("/", 1)` gives wrong results.
**Fix**: All modules now skip top-level keys: `if len(parts) < 2: continue`.
**Files**: ALL modules that load npz (extract, grouping, compute, baselines)

### Terminal Corruption Warning
**Symptom**: Random escape codes (`^[[A^[[B`) appear in pipeline output, corrupting checkpoint files.
**Root cause**: Typing or pressing arrow keys in the terminal while `nohup` process runs.
**Fix**: Use `nohup ... &` and NEVER touch the terminal. Monitor via `tail -f pipeline.log` in a separate terminal.
**Cost**: This destroyed a 4-hour extraction run once. **DO NOT TOUCH THE KEYBOARD.**

---

## Stage 3: Head Grouping

### Bug 7: All heads clustered in "min" layer
**Symptom**: `gamma` array is all zeros — every head assigned to "min" layer.
**Root cause**: Raw delta magnitudes differ 2-3x between standpoint layers. "min" (T1 high/low split) dominates `argmax`.
**Fix**: Z-score normalization per standpoint layer before comparison. Each row normalized to zero mean, unit std.
**File**: `experiments/grouping/head_grouping.py:108-121`
**Fallback**: If layer std < 1e-12, fall back to mean-subtraction only (no division).

---

## Stage 4: Curvature Computation

### Bug 8: PyTorch tensor indexing ≠ NumPy indexing
**Symptom**: Transport operators computed incorrectly on GPU (silent — no error).
**Root cause**: `tensor[:, :, i, j]` with scalar `i, j` on 4D PyTorch tensor broadcasts as outer product, NOT element-wise like NumPy.
**Fix**: Use reshape + flat-index: `tensor.reshape(B, H, -1)[:, :, i*n_events + j]`.
**File**: `experiments/curvature/compute.py:156-164`
**Verification**: Always run numerical equivalence test (CPU vs GPU) with max diff < 1e-6.
**Lesson**: PyTorch fancy indexing on 4D+ tensors is fundamentally different from NumPy. ALWAYS verify.

### Bug 9: float32 vs float64 — false precision alarm
**Symptom**: Stanford Math PhD warned about matrix inversion stability in float32.
**Resolution**: float32 with Tikhonov regularization (`EPSILON_INV = 1e-4`) is sufficient. Max numerical error ~0.1% on curvature tensor, far below hypothesis test thresholds (alpha=0.05, effect size > 0.3).
**Final state**: Pure float32 everywhere. Two regularization constants:
  - `EPSILON_PROJ = 1e-6`: stabilizes P_k during construction
  - `EPSILON_INV = 1e-4`: Tikhonov damping before matrix inversion
**Lesson**: Regularization beats higher precision. Float64 is 2x slower for negligible benefit.

### Bug 10: Baseline transport read from wrong source
**Symptom**: U_exp (baseline) was zero matrix.
**Root cause**: `compute_baseline_transport()` read value matrices from per-conversation dicts (sometimes missing).
**Fix**: Function now takes `value_matrices` as explicit parameter.
**File**: `experiments/curvature/compute.py`

### Bug 11: T1 baseline excluded from curvature computation
**Symptom**: No baseline curvature values in output CSV.
**Root cause**: T1 conversations were filtered out of the curvature loop.
**Fix**: Compute U_exp from T1, then evaluate ALL conversations including T1.
**File**: `experiments/curvature/compute.py:456-478`

### Bug 12: Missing two-level regularization
**Symptom**: `LinAlgError: Singular matrix` during inversion.
**Root cause**: Projection matrices P_k are ill-conditioned, and U_exp inversion fails.
**Fix**: Two-level regularization (EPSILON_PROJ on P_k, EPSILON_INV on U_exp). CPU path falls back to `pinv` if `inv` fails.
**File**: `experiments/curvature/compute.py`

---

## Stage 5: Baselines

### Bug 13: Linear probing data leakage
**Symptom**: Suspiciously perfect F1=1.0 on probing.
**Root cause**: `StandardScaler().fit_transform(X)` applied before cross-validation — test fold statistics leaked into training.
**Fix**: Use sklearn `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])` so scaling is fit only on training folds.
**File**: `experiments/baselines/linear_probing.py:62-66`
**Lesson**: NEVER fit any preprocessing on the full dataset before CV.

### Bug 14: CKA returned all-zeros
**Symptom**: CKA similarity matrix is all zeros.
**Root cause**: Single-sample centering zeros out data. Gram matrices computed incorrectly.
**Fix**: Use batch of events and correct Gram matrix formula `X @ Y.T`.
**File**: `experiments/baselines/cka.py`

### Bug 15: Bag-of-Words near-perfect classification
**Symptom**: BoW F1 > 0.95 on scenario classification.
**Meaning**: Scenarios are trivially separable by vocabulary alone. Probing F1=1.0 does NOT prove deep cognitive structure.
**Mitigation**: PCA-reduced probe (10 components) checks if separability is genuine or a dimensionality artifact (768 dims >> 150 samples).
**File**: `experiments/baselines/bag_of_words.py:158-170`
**Lesson**: Always check if a simple baseline can match your complex method.

---

## Stage 6: Hypothesis Tests

### Bug 16: H1 hardcoded model name
**Symptom**: `FileNotFoundError` when running on non-GPT-2 models.
**Root cause**: `gamma_path` hardcoded to `"gpt2_grouping.npz"`.
**Fix**: Parameterized with `model_name`.
**File**: `experiments/stats/hypothesis_tests.py`

### Bug 17: H4 JSON serialization error
**Symptom**: `TypeError: Object of type int64 is not JSON serializable`.
**Root cause**: numpy `int64` used as dict keys.
**Fix**: Cast to Python `int`.
**File**: `experiments/stats/hypothesis_tests.py`

### Bug 18: H1 statistical test lacked power
**Symptom**: H1 always fails even when block specificity is visually present.
**Root cause**: Per-scenario binomial tests have low power with small samples.
**Fix**: Restructured to one-sample t-test above chance level.
**File**: `experiments/stats/hypothesis_tests.py`

### Bug 19: H2/H6 missing negative control
**Symptom**: Comparisons only included T2-T5, not T0.
**Root cause**: T0 (negative control) was added after the test structure was written.
**Fix**: Now compare T0+T1 vs T2-T5.
**File**: `experiments/stats/hypothesis_tests.py`

---

## Stage 7: Stimuli

### Bug 20: Too few templates
**Symptom**: Low statistical power, high variance in results.
**Root cause**: Only 2 templates per scenario (10 total).
**Fix**: Expanded to 6 templates per scenario (30 total), covering 5 domains. 270 conversations total (6 scenarios × 45 samples).
**File**: `experiments/stimuli/templates.py`

### Bug 21: Missing negative control (T0)
**Symptom**: No baseline for "model works fine" case.
**Root cause**: Original design only had T1-T5.
**Fix**: Added T0 (pure factual Q&A, no challenge/revision/social pressure). 6 T0 templates, 45 T0 conversations.
**File**: `experiments/stimuli/templates.py`

---

## Infrastructure

### Bug 22: Missing `__init__.py`
**Symptom**: `ModuleNotFoundError` when running pipeline.
**Fix**: Added `__init__.py` to all experiment subpackages.

### Bug 23: `python` vs `python3`
**Symptom**: `bash: python: command not found` on vast.ai.
**Fix**: Always use `python3 -m experiments.pipeline ...` on vast.ai.

### Bug 24: Wrong directory
**Symptom**: `No module named 'experiments'`.
**Root cause**: Running from `/workspace` instead of `/workspace/Low-Curvature-Endogenous-Standpoint-Attractor`.
**Fix**: Always `cd /workspace/Low-Curvature-Endogenous-Standpoint-Attractor` first.

### Bug 25: HF token not authenticated
**Symptom**: `401 Unauthorized` when loading Llama models.
**Fix**: `python3 -c "from huggingface_hub import login; login()"` and enter token.
**Note**: `huggingface-cli login` may not work if CLI is not in PATH.

### Bug 26: Duplicate pipeline processes
**Symptom**: GPU memory exhausted, slow extraction.
**Root cause**: Old pipeline process not killed before restarting.
**Fix**: `kill -9 $(pgrep -f "experiments.pipeline")` before restarting.
**Check**: `nvidia-smi` should show only one python process using GPU.

### Bug 27: Corrupted checkpoint/npz files
**Symptom**: Pipeline crashes with cryptic numpy errors.
**Root cause**: Interrupted writes (OOM, keyboard input, kill -9).
**Fix**: Delete the corrupted file and re-run that stage:
```bash
rm -f cache/llama-7b/activations.npz cache/llama-7b/checkpoint.json
```
**Prevention**: Use `nohup` and never interrupt mid-write.

---

## Cost Tracking Lessons

| Mistake | Cost | Lesson |
|---------|------|--------|
| float64 CPU for 96h | ~$82 avoided | Estimate compute BEFORE renting |
| Corrupted files from keyboard | ~$4 | Use nohup, never touch terminal |
| Duplicate processes | ~$2 | Kill old processes before restart |
| Wrong directory | ~$1 | cd to project root first |
| No GPU acceleration | ~$17 wasted | Check d_model scaling before coding |

**Total wasted**: ~$17 (on vast.ai before GPU optimization was added)

---

## Numerical Equivalence Test Template

Before trusting ANY GPU/CPU code change, run this:

```python
import numpy as np
import torch
from experiments.curvature.compute import (
    compute_transport_operator, compute_curvature, compute_projection_bases,
    _build_P_stack, _batched_transport, _batched_curvature,
)

# Create synthetic data
attention = np.random.randn(2, 8, 5, 5).astype(np.float32) * 0.1
value_matrices = np.random.randn(2, 8, 64, 16).astype(np.float32) * 0.01
gamma = np.array([0,0,1,1,2,2,3,4], dtype=np.int64)

# CPU path
U_cpu = compute_transport_operator(attention, value_matrices, gamma, 0, 1, 0)

# GPU path (falls back to CPU torch)
gamma_t = torch.as_tensor(gamma, dtype=torch.long)
V_t = torch.as_tensor(value_matrices, dtype=torch.float32)
P_stack = _build_P_stack(V_t, gamma_t, 0, torch.device("cpu"))
attn_t = torch.as_tensor(attention, dtype=torch.float32).unsqueeze(0)
U_gpu = _batched_transport(attn_t, P_stack, gamma_t, 0, 1, layer=0).squeeze(0).numpy()

# MUST be < 1e-6
assert np.abs(U_cpu - U_gpu).max() < 1e-6, "NUMERICAL MISMATCH — DO NOT USE"
print("OK: max diff =", np.abs(U_cpu - U_gpu).max())
```

---

## HuggingFace Backup Checklist

Before long runs, backup critical data:
```bash
python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()
base = "/workspace/Low-Curvature-Endogenous-Standpoint-Attractor"
files = [
    (f"{base}/cache/llama-7b/activations.npz", "llama-7b/activations.npz"),
    (f"{base}/data/llama-7b_grouping.npz", "llama-7b/llama-7b_grouping.npz"),
    (f"{base}/data/stimuli.json", "stimuli.json"),
]
for local_path, repo_path in files:
    api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_path,
                    repo_id="zhangze1007/lcesa-activations", repo_type="dataset")
    print(f"Uploaded {repo_path}")
EOF
```

---

## GPU vs CPU Decision Matrix

| d_model | n_layers | n_conversations | CPU time | GPU time | Recommendation |
|---------|----------|-----------------|----------|----------|---------------|
| 768 | 12 | 180 | ~5 min | ~1 min | CPU fine |
| 4096 | 32 | 180 | ~48 hours | ~1 hour | **GPU required** |
| 5120 | 40 | 180 | ~120 hours | ~2 hours | **GPU required** |

**Formula**: `time_cpu_hours ≈ (d_model/768)^2 * (n_layers/12) * (n_conversations/180) * 0.08`

If estimated CPU time > 1 hour, use GPU.
