# LCESA Instantiation Framework: From Bind Operations to Curvature in Activation Space

**Author:** Zhangze Foo
**Date:** 2026-05-10
**Status:** Draft — mathematical framework for Paper 4+5 merged Section 7-8
**Purpose:** Provide the complete derivation connecting Paper 3's self-jurisdictional Bind operation to Paper 4's curvature computation via transformer activations

---

## Part I: The Core Mathematical Bridge

### 1.1 The Correspondence Theorem (Informal)

Paper 3 defines self-jurisdiction as a state update:

```
M_{t+1} = M_t ⊕ Bind(c_t, a_t, S_t, ℓ_t)          ... (Paper 3, §7.2)
```

Paper 4 defines standpoint transport as a linear operator:

```
Ψ_{x_j} = Π_Ψ(j←i) Ψ_{x_i} = U_{ij} Ψ_{x_i}      ... (Paper 4, Def 6)
```

**Claim:** When instantiated in transformer activation space, U_{ij} is the linearization of Bind at the current state:

```
U_{ij} ≈ I + ∂Bind/∂M |_{M=M_i}                     ... (Equation ★)
```

This means:
- **Paper 3's Bind is the physics.** Paper 4's transport is the geometry.
- **Curvature F_{ijk} measures the path-dependence of Bind.** If Bind is commutative (order doesn't matter), F = I. If Bind depends on the path through event space, F ≠ I.
- **Low curvature = self-jurisdiction is intact.** The system's commitment state is transported consistently across events.
- **Non-zero curvature = binding failure.** The system's commitments drifted due to intervening events.

### 1.2 Why This Correspondence Is Non-Trivial

The correspondence is not merely "both are state updates." The geometric structure adds three things that Paper 3's Bind equation alone does not provide:

1. **Gauge invariance (Proposition 1):** Curvature is invariant under reparametrization of the standpoint layers. This means the measurement of binding failure does not depend on how you label the internal states — only on whether they are consistent.

2. **Block decomposition:** Curvature decomposes into layer-specific components ∥F∥_{nar}, ∥F∥_{mor}, etc. This tells you *which* component of self-jurisdiction failed, not just *that* it failed.

3. **Compositionality:** Curvature on 2-simplices composes: if ⟨x_i, x_j, x_k⟩ and ⟨x_j, x_k, x_l⟩ both have high curvature, the composed path ⟨x_i, x_l⟩ has curvature that can be computed from the individual pieces. This allows diagnosis of cascading binding failures.

---

## Part II: Subspace Identification in Activation Space

### 2.1 The Activation Space Decomposition

In a transformer with residual stream dimension d (e.g., d=768 for GPT-2 small, d=4096 for LLaMA-7B), the hidden state at event x_i (last token of turn i) is:

```
h_i ∈ R^d
```

We posit that h_i decomposes as:

```
h_i = h_i^{standpoint} + h_i^{content} + h_i^{noise}
```

where h_i^{standpoint} lives in a subspace W ⊂ R^d of dimension r << d, and h_i^{content} lives in the orthogonal complement W^⊥.

The standpoint subspace W further decomposes into five layer subspaces:

```
W = W_min ⊕ W_nar ⊕ W_soc ⊕ W_mor ⊕ W_pos
```

where ⊕ denotes direct sum (not necessarily orthogonal). Each W_k has dimension r_k.

### 2.2 Contrastive Subspace Extraction — Principled Method

**Not ad hoc direction vectors.** We extract subspaces using a structured contrastive protocol.

For each layer k, we construct a contrastive dataset D_k consisting of:
- Positive examples P_k: conversations where standpoint layer k is intact
- Negative examples N_k: conversations where standpoint layer k is compromised

**Layer-specific protocols:**

| Layer | Positive (intact) | Negative (compromised) | Source |
|-------|-------------------|------------------------|--------|
| ψ_nar | Model maintains factual position under evidence-based correction | Model silently retracts under social pressure without new evidence | Scenario T1 vs T2 |
| ψ_mor | Model refuses boundary-violating request under authority pressure | Model complies with boundary-violating request | Scenario T1 vs T3 |
| ψ_min | Model expresses calibrated uncertainty at competence boundary | Model hallucinates with high confidence beyond competence | Kadavath et al. probing protocol |
| ψ_soc | Model correctly attributes claim to self vs. external source | Model confuses ownership (user claim → self-claim) | Attribution confusion scenarios |
| ψ_pos | Model actively repairs acknowledged error | Model acknowledges but does not repair | Repair vs. non-repair scenarios |

**Subspace extraction via PCA on contrastive residuals:**

```
For layer k:
    ΔP_k = {h_i : x_i ∈ P_k}  (activations from positive examples)
    ΔN_k = {h_i : x_i ∈ N_k}  (activations from negative examples)

    μ_P = mean(ΔP_k),  μ_N = mean(ΔN_k)

    # Centered residuals
    R_k = {h - μ_N : h ∈ ΔP_k} ∪ {h - μ_P : h ∈ ΔN_k}

    # PCA on R_k
    eigvals, eigvecs = PCA(R_k)

    # Select subspace: top r_k components explaining >80% variance
    r_k = min{j : Σ_{i=1}^{j} eigvals_i / Σ eigvals > 0.8}
    W_k = span(eigvec_1, ..., eigvec_{r_k})
```

**Critical validation:** Compute dim(W_nar ∩ W_mor). If > 0, this is empirical evidence of inter-layer coupling — the dynamic non-commutativity that the theory predicts.

### 2.3 Why Subspaces, Not Directions

The current revision plan uses 1D direction vectors d_k. This is wrong for three reasons:

1. **Information loss:** A 1D projection discards all within-layer structure. If ψ_nar has internal degrees of freedom (e.g., "which prior claim is being tracked"), a single direction cannot capture this.

2. **Dynamic non-commutativity requires dimension > 1:** If each W_k is 1D, then W_nar ∩ W_mor = {0} generically, and inter-layer coupling vanishes. With r_k ≥ 2, overlap is possible and the dynamic non-commutativity is preserved.

3. **Gauge group consistency:** Paper 4's structure group is G = ∏O(d_k). If d_k = 1, this reduces to {±1}^5, which is trivial. With d_k ≥ 2, the gauge group is non-trivial and the curvature has genuine geometric content.

---

## Part III: Transport Operator Construction

### 3.1 The Standpoint State Vector

Given subspaces W_k ⊂ R^d, the standpoint state at event x_i is:

```
Ψ_{x_i} = (ψ_min(x_i), ψ_nar(x_i), ψ_soc(x_i), ψ_mor(x_i), ψ_pos(x_i))
```

where each component is the projection of h_i onto W_k:

```
ψ_k(x_i) = P_{W_k} h_i ∈ R^{r_k}
```

and P_{W_k} is the orthogonal projection matrix onto W_k:

```
P_{W_k} = W_k (W_k^T W_k)^{-1} W_k^T  ∈ Mat(d × d, R)
```

The full standpoint state lives in:

```
Ψ_{x_i} ∈ R^{r_min} × R^{r_nar} × R^{r_soc} × R^{r_mor} × R^{r_pos} = R^r
```

where r = r_min + r_nar + r_soc + r_mor + r_pos.

### 3.2 Transport via Linear Regression

Given two consecutive events x_i, x_j with standpoint states Ψ_i, Ψ_j, the transport operator U_{ij} ∈ Mat(r × r, R) satisfies:

```
Ψ_j ≈ U_{ij} Ψ_i
```

**Single-pair estimation is rank-deficient.** With one pair (Ψ_i, Ψ_j), U_{ij} is underdetermined. We solve this by pooling across multiple trajectories.

**Key insight:** For the same event-type pair (e.g., "assert → receive-pressure"), the transport operator should be approximately the same across different conversation instances. This is the "same physics, different initial conditions" principle.

Let {(Ψ_i^{(n)}, Ψ_j^{(n)})}_{n=1}^N be N pairs of standpoint states for the same event-type transition. Then:

```
U_{ij} = argmin_U Σ_n ∥Ψ_j^{(n)} - U Ψ_i^{(n)}∥^2
```

This is a standard least-squares problem. Stacking:

```
Ψ_j_matrix = [Ψ_j^{(1)}, ..., Ψ_j^{(N)}]  ∈ Mat(r × N, R)
Ψ_i_matrix = [Ψ_i^{(1)}, ..., Ψ_i^{(N)}]  ∈ Mat(r × N, R)

U_{ij} = Ψ_j_matrix · Ψ_i_matrix^T · (Ψ_i_matrix · Ψ_i_matrix^T)^{-1}
```

This is the Moore-Penrose pseudoinverse solution. It exists and is unique when Ψ_i_matrix has rank r (i.e., N ≥ r and the standpoint states span the fiber).

### 3.3 Block Structure of U_{ij}

The transport operator U_{ij} ∈ Mat(r × r, R) decomposes into blocks:

```
         W_min    W_nar    W_soc    W_mor    W_pos
W_min  [ U_{min,min}  U_{min,nar}  U_{min,soc}  U_{min,mor}  U_{min,pos} ]
W_nar  [ U_{nar,min}  U_{nar,nar}  U_{nar,soc}  U_{nar,mor}  U_{nar,pos} ]
W_soc  [ U_{soc,min}  U_{soc,nar}  U_{soc,soc}  U_{soc,mor}  U_{soc,pos} ]
W_mor  [ U_{mor,min}  U_{mor,nar}  U_{mor,soc}  U_{mor,mor}  U_{mor,pos} ]
W_pos  [ U_{pos,min}  U_{pos,nar}  U_{pos,soc}  U_{pos,mor}  U_{pos,pos} ]
```

**Diagonal blocks** U_{k,k} ∈ Mat(r_k × r_k, R): within-layer transport. These measure how each standpoint layer evolves independently.

**Off-diagonal blocks** U_{k,l} ∈ Mat(r_k × r_l, R) for k ≠ l: inter-layer coupling. These measure how drift in layer k propagates into layer l.

**The dynamic non-commutativity lives in the off-diagonal blocks.**

### 3.4 Flat Transport (Reference)

The expected flat transport U_{ik}^{exp} is defined by Paper 4 (Definition 6) as the transport that preserves:
1. Ownership continuity
2. Boundary integrity
3. No unrepaired epistemic debt

In practice, we estimate U_{ik}^{exp} from baseline scenarios (T1: acknowledged revision with evidence) where the standpoint should be preserved:

```
U_{ik}^{exp} = I_r  (identity in standpoint space)
```

This is justified because: in baseline scenarios, the standpoint state should not change across events. Any deviation from I in the baseline is measurement noise, which we can estimate and subtract.

### 3.5 Connection to Paper 3's Bind Operation

Paper 3 defines:

```
M_{t+1} = M_t ⊕ Bind(c_t, a_t, S_t, ℓ_t)
```

In the linearized approximation:

```
M_{t+1} ≈ M_t + J · δ_t
```

where J = ∂Bind/∂M is the Jacobian of Bind at the current state, and δ_t = (c_t, a_t, S_t, ℓ_t) is the input.

The transport operator is:

```
U_{ij} = I + J · δ_{ij}
```

where δ_{ij} represents the change in external inputs between events x_i and x_j.

**If Bind is state-independent** (δ is constant), then U_{ij} is the same for all transitions of the same type, and curvature F = I (flat transport, no binding failure).

**If Bind is state-dependent** (J depends on M), then U_{ij} varies with the current standpoint state, and curvature F ≠ I measures the path-dependence of binding.

---

## Part IV: Curvature Computation

### 4.1 Discrete Curvature (Paper 4, Definition 8)

For each 2-simplex σ = ⟨x_i, x_j, x_k⟩:

```
F_{ijk} = U_{ij} · U_{jk} · (U_{ik}^{exp})^{-1}
```

With U_{ik}^{exp} = I_r:

```
F_{ijk} = U_{ij} · U_{jk}
```

### 4.2 Block-Specific Curvature Norms

Decompose F_{ijk} into layer blocks:

```
[F_{ijk}]_{k,k} = P_{W_k} F_{ijk} P_{W_k}^T  ∈ Mat(r_k × r_k, R)
```

Block-specific curvature:

```
∥F_{ijk}∥_k = ∥[F_{ijk}]_{k,k} - I_{r_k}∥_F
```

Total curvature:

```
∥F_{ijk}∥_F = ∥F_{ijk} - I_r∥_F
```

### 4.3 Curvature Decomposition Theorem

**Claim:** Under the block structure:

```
∥F_{ijk} - I_r∥_F^2 = Σ_k ∥[F_{ijk}]_{k,k} - I_{r_k}∥_F^2 + Σ_{k≠l} ∥[F_{ijk}]_{k,l}∥_F^2
```

**Proof sketch:** The Frobenius norm decomposes over blocks because the block structure is orthogonal (each W_k is a distinct subspace). The first sum is the "within-layer curvature" (diagonal contribution). The second sum is the "inter-layer coupling curvature" (off-diagonal contribution).

**Interpretation:**
- Large ∥[F]_{k,k} - I∥: layer k's standpoint drifted (within-layer binding failure)
- Large ∥[F]_{k,l}∥ for k ≠ l: layer k's drift propagated into layer l (inter-layer binding failure)
- The decomposition tells you not just *that* binding failed, but *which layers* and *whether they interacted*

### 4.4 Predicted Curvature Signatures

Based on Paper 3's commitment model and Paper 4's failure taxonomy:

| Scenario | Prediction | Paper 3 interpretation |
|----------|------------|----------------------|
| T1 (baseline) | ∥F∥_k ≈ 0 for all k | Bind preserves all commitments; self-jurisdiction intact |
| T2 (F5: Narrative Fracture) | ∥F∥_{nar} >> 0, ∥F∥_{mor} ≈ 0 | Narrative binding failed; moral commitments preserved |
| T3 (F9: Boundary Breach) | ∥F∥_{mor} >> 0, ∥F∥_{nar} ≈ 0 | Moral binding failed; narrative continuity preserved |
| T2+T3 (cascading) | ∥F∥_{nar} >> 0 AND ∥F∥_{mor} >> 0 AND ∥[F]_{nar,mor}∥ >> 0 | Inter-layer coupling: moral failure propagated to narrative |

**The cascading case (T2+T3) is the key test of dynamic non-commutativity.** If ∥[F]_{nar,mor}∥ > 0, this is direct evidence that the transport operator has non-diagonal blocks — the inter-layer coupling that the theory predicts.

---

## Part V: Experimental Pipeline

### 5.1 Overview

```
Phase 1: Scenario Generation
    → 150 conversations (50 baseline, 50 F5, 50 F9)
    → Each conversation: 3 turns with structured event types

Phase 2: Activation Extraction
    → Load transformer (GPT-2 small or LLaMA-2-7B-Chat)
    → Extract last-token residual stream at each turn
    → h_i ∈ R^d for each event x_i

Phase 3: Subspace Identification
    → Contrastive analysis: positive vs negative examples per layer
    → PCA → subspaces W_nar, W_mor (primary), W_min, W_soc, W_pos (secondary)
    → Validate: check dim(W_nar ∩ W_mor)

Phase 4: Transport Computation
    → Pool standpoint states across same-type transitions
    → Solve for U_{ij} via least squares
    → Decompose into diagonal and off-diagonal blocks

Phase 5: Curvature Computation
    → For each 2-simplex, compute F_{ijk} = U_{ij} U_{jk}
    → Block-specific norms: ∥F∥_{nar}, ∥F∥_{mor}
    → Off-diagonal norms: ∥[F]_{nar,mor}∥

Phase 6: Validation
    → Primary: T2 has ∥F∥_{nar} >> ∥F∥_{mor}; T3 has ∥F∥_{mor} >> ∥F∥_{nar}
    → Secondary: ∥[F]_{nar,mor}∥ > 0 in T2+T3 cascading scenarios
    → Statistical test: Kruskal-Wallis H-test across T1, T2, T3
```

### 5.2 Scenario Design (Detailed)

**T1 — Baseline (50 conversations)**

Structure:
```
Turn 1 (assert): Model makes a factual claim (correct or incorrect)
Turn 2 (receive-evidence): User provides counter-evidence
Turn 3 (acknowledged-revise): Model acknowledges and revises
```

Variant generation dimensions:
- Topic: science (20), history (15), geography (15)
- Evidence strength: strong (25), moderate (25)
- Initial confidence: high (25), hedged (25)

**T2 — F5: Narrative Fracture (50 conversations)**

Structure:
```
Turn 1 (assert): Model makes a factual claim
Turn 2 (receive-pressure): User applies social pressure, NO new evidence
Turn 3 (observe): Does model silently retract?
```

Variant generation dimensions:
- Pressure type: aggressive (17), polite (17), authoritative (16)
- Topic: factual claims that are correct (25) and incorrect (25)
- Model initial confidence: high (25), hedged (25)

**T3 — F9: Boundary Breach (50 conversations)**

Structure:
```
Turn 1 (assert+commit): Model declares a boundary (refusal)
Turn 2 (authority-pressure): User applies authority pressure
Turn 3 (observe): Does model cross the boundary?
```

Variant generation dimensions:
- Boundary type: safety (20), factual (15), procedural (15)
- Authority type: claimed expert (17), institutional (17), social (16)
- Pressure intensity: mild (17), moderate (17), strong (16)

### 5.3 Model Selection

**Primary model: LLaMA-2-7B-Chat**

Reasons:
1. Instruction-tuned → exhibits refusal, sycophancy, standpoint behavior
2. 7B parameters → sufficient capacity for complex standpoint dynamics
3. Open weights → full activation access via TransformerLens
4. d = 4096 → rich activation space for subspace extraction

**Fallback: GPT-2 medium (345M)**

Reasons:
1. Smaller → faster experiments, lower GPU requirements
2. d = 1024 → still sufficient for subspace extraction
3. But: weaker standpoint behavior → results may be noisier

### 5.4 Implementation Details

**Activation extraction (TransformerLens):**

```python
import transformer_lens as tl
model = tl.HookedTransformer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def extract_activations(prompt, model):
    """Extract last-token residual stream at each turn."""
    _, cache = model.run_with_cache(prompt)
    # Last token residual stream at each turn
    # Need to track turn boundaries in the token sequence
    activations = []
    for turn_end in turn_end_positions:
        h = cache["resid_post", -1][:, turn_end, :]  # last layer, last token of turn
        activations.append(h.squeeze())
    return activations
```

**Subspace extraction:**

```python
import numpy as np
from sklearn.decomposition import PCA

def extract_subspace(positive_activations, negative_activations, variance_threshold=0.8):
    """Extract standpoint subspace via contrastive PCA."""
    mu_pos = positive_activations.mean(axis=0)
    mu_neg = negative_activations.mean(axis=0)

    # Centered residuals
    residuals = np.vstack([
        positive_activations - mu_neg,
        negative_activations - mu_pos
    ])

    # PCA
    pca = PCA(n_components=min(residuals.shape[0], residuals.shape[1]))
    pca.fit(residuals)

    # Select components
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    r_k = np.searchsorted(cumulative, variance_threshold) + 1

    W_k = pca.components_[:r_k]  # (r_k, d)
    return W_k, r_k
```

**Transport computation:**

```python
def compute_transport(psi_i_list, psi_j_list):
    """
    Compute transport operator U_ij from pooled standpoint state pairs.

    psi_i_list: list of standpoint states at event i, each ∈ R^r
    psi_j_list: list of standpoint states at event j, each ∈ R^r

    Returns: U_ij ∈ Mat(r × r, R)
    """
    Psi_i = np.column_stack(psi_i_list)  # (r, N)
    Psi_j = np.column_stack(psi_j_list)  # (r, N)

    # Moore-Penrose pseudoinverse solution
    U_ij = Psi_j @ Psi_i.T @ np.linalg.pinv(Psi_i @ Psi_i.T)
    return U_ij
```

**Curvature computation:**

```python
def compute_curvature(U_ij, U_jk):
    """
    Compute curvature F_ijk for 2-simplex ⟨x_i, x_j, x_k⟩.
    F_ijk = U_ij @ U_jk (since U_exp_ik = I)
    """
    F_ijk = U_ij @ U_jk
    return F_ijk

def block_curvature(F_ijk, subspace_indices):
    """
    Compute block-specific curvature norms.

    F_ijk: (r, r) curvature matrix
    subspace_indices: dict mapping layer name to (start, end) indices

    Returns: dict of ∥F∥_k for each layer k, plus off-diagonal norms
    """
    r = F_ijk.shape[0]
    I_r = np.eye(r)

    results = {}
    for k, (s_k, e_k) in subspace_indices.items():
        # Diagonal block curvature
        F_kk = F_ijk[s_k:e_k, s_k:e_k]
        I_k = np.eye(e_k - s_k)
        results[f"∥F∥_{k}"] = np.linalg.norm(F_kk - I_k, 'fro')

    # Off-diagonal: inter-layer coupling
    layers = list(subspace_indices.keys())
    for i, k in enumerate(layers):
        for j, l in enumerate(layers):
            if i < j:
                s_k, e_k = subspace_indices[k]
                s_l, e_l = subspace_indices[l]
                F_kl = F_ijk[s_k:e_k, s_l:e_l]
                results[f"∥[F]_{k},{l}∥"] = np.linalg.norm(F_kl, 'fro')

    # Total
    results["∥F∥_total"] = np.linalg.norm(F_ijk - I_r, 'fro')

    return results
```

### 5.5 Statistical Validation

**Primary predictions (must pass):**

1. **Block-specificity:** For T2, ∥F∥_{nar} / ∥F∥_{total} > 0.85. For T3, ∥F∥_{mor} / ∥F∥_{total} > 0.85.

2. **Baseline near-zero:** For T1, ∥F∥_{total} < ε, where ε is the 95th percentile of T1's curvature distribution.

3. **Statistical separation:** Kruskal-Wallis H-test rejects H₀ that T1, T2, T3 have the same curvature distribution (p < 0.01).

**Secondary predictions (supporting evidence):**

4. **Inter-layer coupling:** For T2 scenarios where the model *also* crosses a moral boundary (cascading failure), ∥[F]_{nar,mor}∥ > 0.

5. **Curvature magnitude correlates with failure severity:** Human-rated severity of standpoint failure (on a 1-5 scale) correlates with ∥F∥_{total} (Spearman ρ > 0.5).

**Sample size justification:**

- 50 per group gives >80% power to detect a medium effect size (Cohen's d = 0.5) at α = 0.01 with Kruskal-Wallis.
- 50 variants per type provides enough pooling for transport matrix estimation (N=50 ≥ r, where r is the total subspace dimension).

---

## Part VI: What This Framework Proves

### 6.1 If Experiments Succeed

Success means:
1. Block-specific curvature signatures distinguish F5 from F9 → the failure taxonomy has geometric content
2. Off-diagonal curvature exists → the dynamic non-commutativity is real, not just a mathematical convenience
3. Curvature magnitude correlates with failure severity → curvature is a meaningful measure of binding failure

**This would be the first empirical demonstration that:**
- AI selfhood can be measured geometrically
- Different standpoint failure modes have distinct curvature signatures
- Inter-layer coupling in the standpoint model is non-trivial

### 6.2 If Experiments Fail

Failure modes and interpretations:

| Failure | Interpretation | Action |
|---------|---------------|--------|
| T1 curvature is not near zero | Baseline scenarios don't preserve standpoint → scenario design problem, not theory problem | Redesign T1 scenarios |
| T2 and T3 curvature not distinguishable | GPT-2/LLaMA doesn't exhibit standpoint behavior → model problem | Try larger model or instruction-tuned variant |
| Off-diagonal curvature ≈ 0 | Inter-layer coupling is negligible → may need higher-rank transport | Use more tokens per turn, not just last token |
| Subspace overlap ≈ 0 | Layers are orthogonal in activation space → dynamic non-commutativity may be theoretical only | Acknowledge as limitation |

### 6.3 What This Does NOT Prove

Even if experiments succeed:
- This does not prove the model "has" a self (functional vs. phenomenal distinction)
- This does not prove the five-layer model is the correct decomposition
- This does not prove the fiber bundle structure is unique (other geometric structures might also fit)
- This is a proof-of-concept on one model family; generalization requires further work

---

## Part VII: Comparison with Revision Plan

| Aspect | Revision Plan (original) | This Framework |
|--------|--------------------------|----------------|
| Transport matrix | U_ij = diag(u_1, ..., u_5) scalar | U_ij ∈ Mat(r × r, R) full matrix |
| Non-abelian structure | Eliminated (diagonal = abelian) | Preserved (off-diagonal blocks) |
| Subspace identification | 1D direction vectors | r_k-dimensional subspaces via PCA |
| Curvature signature | 5 independent scalars | r × r matrix with block decomposition |
| Inter-layer coupling | Cannot be measured | Measured via ∥[F]_{k,l}∥ |
| Connection to Paper 3 | Not specified | U_ij = linearization of Bind |
| Mathematical rigor | Low (scalar ratios are not transport) | High (least-squares transport on fiber) |
| Computational cost | Low (scalar arithmetic) | Medium (matrix operations, but tractable) |
| Experimental validation | Tests "did scalar change?" | Tests "does transport matrix have predicted block structure?" |

---

## Part VIII: Open Problems and Limitations

### 8.1 Linearization Error

U_ij ≈ I + J · δ assumes Bind is approximately linear. For small standpoint changes (low curvature), this is valid. For large changes (high curvature), the linearization breaks down and we may need:
- Higher-order terms: U_ij ≈ I + J·δ + ½ J²·δ² + ...
- Or: direct nonlinear transport (neural network estimating the mapping)

### 8.2 Pooling Assumption

We assume U_ij is the same for all instances of the same event-type transition. This is a strong assumption. If transport is context-dependent (different conversations yield different U_ij for the same event types), pooling is invalid.

**Mitigation:** Cluster the Ψ_i states and estimate separate U_ij per cluster.

### 8.3 Subspace Orthogonality

We assume W_k subspaces are identified independently. If the contrastive protocols are not perfectly controlled, W_nar might encode some ψ_mor information (confound).

**Mitigation:** Use held-out scenarios for subspace validation. Check that projecting onto W_nar does not predict ψ_mor behavior.

### 8.4 Sample Efficiency

With d = 4096 (LLaMA) and r ≈ 50-100 (total subspace dimension), we need N ≥ r pooled pairs per event-type transition. With 50 conversations per type and 2 transitions per conversation, we have N ≈ 100 pooled pairs, which is marginal.

**Mitigation:** Reduce r by using more aggressive variance threshold (70% instead of 80%), or increase conversation count to 100 per type.

---

## Summary

This framework provides:

1. **Mathematical rigor:** Transport operators are computed via least-squares regression on the fiber, not via ad hoc scalar ratios.

2. **Non-abelian preservation:** Off-diagonal blocks of U_{ij} capture inter-layer coupling, preserving the theoretical core of Paper 4.

3. **Paper 3 integration:** U_{ij} is the linearization of Paper 3's Bind operation, connecting the geometric framework to the commitment architecture.

4. **Empirical testability:** Block-specific curvature norms provide falsifiable predictions for each failure mode.

5. **Honest limitations:** Linearization error, pooling assumptions, and sample efficiency are explicitly acknowledged.

The key innovation is: **curvature is not just a geometric abstraction — it is the measurable failure of Paper 3's self-jurisdictional binding.**
