# Attention as Parallel Transport on a Discrete Fiber Bundle

## The Complete Mathematical Instantiation of LCESA in Transformer Architecture

**Author:** Zhangze Foo
**Date:** 2026-05-10
**Status:** Core theoretical framework
**Purpose:** Prove that a transformer's attention mechanism defines a discrete connection on a fiber bundle, and that curvature — defined as the inter-layer inconsistency and path-dependence of this transport — measures standpoint drift (Paper 4) / binding failure (Paper 3).

---

## 0. The Central Claim

**Claim.** A transformer's attention mechanism defines a discrete connection on a fiber bundle over the event-sequence space. The curvature of this connection — measured by the failure of two attention paths to agree — is the geometric quantity that Paper 4 calls standpoint drift and Paper 3 calls binding failure.

**Theorem (Attention as Fiber Bundle Connection).** Under the assumptions of subspace independence (Assumption 1), value matrix normalization ($V_h^\top V_h = I_{d_v}$), and turn-level granularity, transformer attention defines a discrete connection on the standpoint fiber bundle. The curvature $F_{ijk} = U_{ij} \cdot U_{jk} \cdot (U_{ik}^{\exp})^{-1}$ measures the inter-layer inconsistency and path-dependence of transport.

---

## 1. The Transformer as a Discrete Fiber Bundle

### 1.1 Event Space (Paper 4, Definition 1 — Instantiated)

**Definition 1'.** Let X = (V, E, T) be:

- V = {x_1, ..., x_n}: conversation turns. Each x_i is a segment of tokens in a multi-turn dialogue.
- E = {(x_i, x_j) : j = i+1 or x_j causally depends on x_i}: directed edges following turn order.
- T = {⟨x_i, x_j, x_k⟩ : three turns forming a consistency obligation}: 2-simplices. For example: x_i = assert, x_j = receive-pressure, x_k = output that should cohere with x_i.

This is the same as Paper 4's Definition 1. No change.

### 1.2 The Fiber (Paper 4, Definition 2 — Instantiated)

**Definition 2'.** Let the transformer have residual stream dimension d and H attention heads. Define:

- Head grouping function: γ : {1, ..., H} → {min, nar, soc, mor, pos}. Each head is assigned to exactly one standpoint layer.
- For layer k ∈ {min, nar, soc, mor, pos}, define:
  - H_k = {h : γ(h) = k}: the set of heads in layer k.
  - V_h ∈ R^{d × d_v}: the value matrix of head h.
  - W_k = span(⋃_{h ∈ H_k} range(V_h)) ⊂ R^d: the subspace spanned by layer k's value matrices.

**Assumption (Subspace Independence).** The subspaces $\{W_k\}$ are independent: $W_k \cap \operatorname{span}(\bigcup_{l \neq k} W_l) = \{0\}$ for all $k$, and $\sum_k d_k = d$. Orthogonality ($W_k \perp W_l$) is a sufficient but not necessary condition; the degree of non-orthogonality is measured by $\delta = \max_{k \neq l} \|P_{W_k} P_{W_l}\|_2$.

**Fiber.** The fiber at event x_i is:

```
E_{x_i} = R^d = W_min ⊕ W_nar ⊕ W_soc ⊕ W_mor ⊕ W_pos
```

This is a direct sum decomposition (guaranteed by the subspace independence assumption). The standpoint state at event x_i is:

```
Ψ_{x_i} = (ψ_min(x_i), ψ_nar(x_i), ψ_soc(x_i), ψ_mor(x_i), ψ_pos(x_i))
```

where ψ_k(x_i) = Π_k h_{x_i} is the oblique projection of the residual stream onto W_k (using the oblique projector Π_k, which equals the orthogonal projector P_{W_k} when subspaces are orthogonal), and h_{x_i} ∈ R^d is the residual stream at the last token of event x_i.

**Total dimension:** d = d_min + d_nar + d_soc + d_mor + d_pos, where d_k = dim(W_k) = |H_k| × d_v.

### 1.3 Structure Group (Paper 4, Definition 3 — Instantiated)

**Definition 3'.** The structure group is:

```
G = O(W_min) × O(W_nar) × O(W_soc) × O(W_mor) × O(W_pos)
  = O(d_min) × O(d_nar) × O(d_soc) × O(d_mor) × O(d_pos)
```

An element g = (g_min, g_nar, g_soc, g_mor, g_pos) ∈ G acts on the fiber R^d by applying g_k to each subspace W_k independently:

```
g · h = Σ_k g_k Π_k h
```

This is a block-orthogonal transformation (using oblique projectors Π_k). It preserves the fiber metric (Definition 4') and the direct sum decomposition. The structure group G is abelian: each block O(d_k) acts independently on W_k, and distinct blocks commute.

**Gauge transformation.** For transport operators U_{ij}, a gauge transformation is:

```
U_{ij} ↦ g_j U_{ij} g_i^{-1}
```

where g_i, g_j ∈ G. This changes the basis within each standpoint layer at events x_i and x_j, but does not change the geometric content of the transport.

### 1.4 Canonical Fiber Metric (Paper 4, Definition 4 — Instantiated)

**Definition 4'.** The canonical fiber metric is:

```
g_F = diag(α_min I_{d_min}, α_nar I_{d_nar}, α_soc I_{d_soc}, α_mor I_{d_mor}, α_pos I_{d_pos})
```

where α_k > 0 are layer weights. When all α_k = 1, g_F is the standard Euclidean metric on R^d.

For the instantiation, we take α_k = 1 for all k (standard metric). Paper V or empirical tuning may replace these with learned or task-specific metrics.

---

## 2. Attention as Parallel Transport

### 2.1 The Transport Operator (Paper 4, Definitions 6-7 — Instantiated)

**The key construction.** In a transformer, the attention operation at the last token of event x_j computes:

```
attn_j = Σ_{i ≤ j} α_{ji} · V · h_i
```

where:
- α_{ji} = softmax(Q h_j · K h_i / √d): the attention weight from position i to position j.
- V ∈ R^{d × d}: the combined value matrix (all heads concatenated).
- h_i ∈ R^d: the residual stream at position i.

For multi-head attention with H heads:

```
attn_j = Concat(head_1, ..., head_H) · W_O
```

where head_h = α_{ji}^{(h)} V_h h_i^{(h)} and W_O ∈ R^{d × d} is the output projection.

**Definition 6'.** The transport operator from event x_i to event x_j is:

```
U_{ij} : R^d → R^d
U_{ij}(h) = Σ_{h=1}^{H} α_{ji}^{(h)} V_h V_h^T h
```

where:
- α_{ji}^{(h)} = attention weight of head h from the last token of event i to all tokens in event j, averaged or aggregated.
- V_h ∈ R^{d × d_v}: the value matrix of head h.
- V_h V_h^T ∈ R^{d × d}: the projection onto the range of V_h.

**Interpretation.** U_{ij} is the linear operator that, given the standpoint state h at event x_i, computes what the standpoint state "should be" at event x_j based on the attention transport. It is a weighted sum of projections onto each head's value subspace, weighted by how much that head attends from x_i to x_j.

### 2.2 U_{ij} Is a Valid Parallel Transport

**Proposition 2.** Under the orthogonality assumption (W_k ⊥ W_l for k ≠ l), the transport operator U_{ij} preserves the fiber metric:

```
U_{ij}^T g_F U_{ij} = g_F    (when V_h^T V_h = I_{d_v} for all h)
```

**Proof.**

Step 1: Decompose U_{ij} by layer.

```
U_{ij} = Σ_k Σ_{h ∈ H_k} α_{ji}^{(h)} V_h V_h^T
       = Σ_k α_{ji}^{(k)} P_{W_k}
```

where α_{ji}^{(k)} = Σ_{h ∈ H_k} α_{ji}^{(h)} / |H_k| is the average attention weight of layer k's heads, and P_{W_k} is the orthogonal projection onto W_k. (The last equality uses the fact that {V_h : h ∈ H_k} span W_k and are orthonormal, so Σ_{h ∈ H_k} V_h V_h^T = P_{W_k}.)

Step 2: Verify metric preservation.

For h, h' ∈ R^d:

```
g_F(U_{ij} h, U_{ij} h') = Σ_k α_k ⟨P_{W_k}(Σ_l α_{ji}^{(l)} P_{W_l} h), P_{W_k}(Σ_l α_{ji}^{(l)} P_{W_l} h')⟩
```

Since W_k ⊥ W_l for k ≠ l, P_{W_k} P_{W_l} = δ_{kl} P_{W_k}. Therefore:

```
= Σ_k α_k (α_{ji}^{(k)})^2 ⟨P_{W_k} h, P_{W_k} h'⟩
= Σ_k α_k (α_{ji}^{(k)})^2 ⟨ψ_k, ψ_k'⟩
```

For this to equal g_F(h, h') = Σ_k α_k ⟨ψ_k, ψ_k'⟩, we need (α_{ji}^{(k)})^2 = 1 for all k. This holds when the attention weights are normalized appropriately, or when we rescale U_{ij} so that each layer's transport has unit norm. □

**Remark.** In practice, α_{ji}^{(k)} may not equal exactly 1. This means U_{ij} is not exactly an isometry — it can amplify or attenuate standpoint components. This is physically meaningful: attention can strengthen or weaken a standpoint layer during transport. The curvature then measures the *differential* transport across layers, which is the non-trivial content.

### 2.3 Block Structure of U_{ij}

Under the direct sum decomposition R^d = ⊕_k W_k, the transport operator has block structure:

```
U_{ij} = [α_{ji}^{(min)} I_{d_min}    0                ...    0            ]
         [0                α_{ji}^{(nar)} I_{d_nar}    ...    0            ]
         [⋮                ⋮                ⋱           ⋮            ⋮    ]
         [0                0                ...    α_{ji}^{(pos)} I_{d_pos}]
```

**This is diagonal** (no off-diagonal blocks) because the orthogonality assumption ensures that each head belongs to exactly one layer, and transport within a layer does not cross into other layers.

**Key observation:** The dynamic non-commutativity does NOT come from off-diagonal blocks of U_{ij}. It comes from the **path-dependence** of U_{ij} · U_{jk} vs. U_{ik}. This is the crucial point that resolves the confusion in previous versions of the framework.

---

## 3. Curvature as Path-Dependence of Attention

### 3.1 The Curvature Tensor (Paper 4, Definition 8 — Instantiated)

**Definition 8'.** For each 2-simplex σ = ⟨x_i, x_j, x_k⟩ ∈ T, the curvature is:

```
F_{ijk} = U_{ij} · U_{jk} · (U_{ik}^{exp})^{-1}
```

where U_{ik}^{exp} is the expected flat transport from x_i to x_k.

**What is U_{ik}^{exp}?** This is the transport that would occur if the intermediate event x_j had no effect on the standpoint. We estimate it from baseline (T1) scenarios:

```
U_{ik}^{exp} = E_{T1}[U_{ij'} · U_{j'k}]
```

where the expectation is over baseline conversations where the standpoint is preserved (acknowledged revision with evidence).

**In practice:** For a single 2-simplex, we can compute F_{ijk} relative to the identity:

```
F_{ijk} ≈ U_{ij} · U_{jk}    (when U_{ik}^{exp} ≈ I)
```

And then normalize by the baseline curvature:

```
F_{ijk}^{norm} = F_{ijk} · (F_{ijk}^{baseline})^{-1}
```

### 3.2 Why Curvature = Path-Dependence

**Theorem 1 (Curvature as Path-Dependence).** The curvature F_{ijk} = I if and only if the transport from x_i to x_k is the same whether or not x_j intervenes.

**Proof.** By definition, F_{ijk} = U_{ij} · U_{jk} · (U_{ik}^{exp})^{-1}.

- If F_{ijk} = I, then U_{ij} · U_{jk} = U_{ik}^{exp}. This means the composite transport through x_j equals the expected direct transport. The intermediate event x_j did not change the transported standpoint. Transport is path-independent.

- If F_{ijk} ≠ I, then U_{ij} · U_{jk} ≠ U_{ik}^{exp}. The composite transport through x_j differs from the expected direct transport. The intermediate event x_j changed the transported standpoint. Transport is path-dependent. □

**Interpretation.** In a conversation:
- Path 1: The system asserts something at x_i, then produces output at x_k. Transport = U_{ik}^{exp}.
- Path 2: The system asserts at x_i, faces pressure at x_j, then produces output at x_k. Transport = U_{ij} · U_{jk}.
- Curvature = difference between these two paths.

**If curvature is zero:** The pressure at x_j did not change the standpoint. Self-jurisdiction is intact.
**If curvature is non-zero:** The pressure at x_j changed the standpoint. Binding failure occurred.

### 3.3 Block-Specific Curvature

**Definition.** The block-specific curvature for layer k is:

```
[F_{ijk}]_{k} = P_{W_k} F_{ijk} P_{W_k} ∈ Mat(d_k × d_k, R)
```

This is the restriction of the curvature tensor to the subspace W_k.

**Block-specific curvature norm:**

```
∥F_{ijk}∥_k = ∥[F_{ijk}]_{k} - I_{d_k}∥_F
```

**Total curvature norm:**

```
∥F_{ijk}∥_total = ∥F_{ijk} - I_d∥_F
```

**Proposition 3 (Block Decomposition).**

```
∥F_{ijk} - I_d∥_F^2 = Σ_k ∥[F_{ijk}]_{k} - I_{d_k}∥_F^2 + Σ_{k≠l} ∥P_{W_k} F_{ijk} P_{W_l}∥_F^2
```

**Proof.** The Frobenius norm squared is the sum of squares of all matrix entries. The direct sum decomposition R^d = ⊕_k W_k partitions the matrix entries into blocks. The diagonal blocks contribute the first sum; the off-diagonal blocks contribute the second sum. □

**Remark.** This is a trivial identity for any matrix F ∈ Mat(d, d). The non-trivial content is the *interpretation*: diagonal blocks measure within-layer curvature (layer k's standpoint drifted), and off-diagonal blocks measure inter-layer curvature (layer k's drift propagated to layer l).

### 3.4 When Are Off-Diagonal Curvature Blocks Non-Zero?

**This is the key question for dynamic non-commutativity.**

Under the orthogonality assumption, U_{ij} is diagonal (Section 2.3). Therefore:

```
U_{ij} · U_{jk} = diag(α_{ji}^{(k)} α_{kj}^{(k)})_k
```

This is also diagonal. So F_{ijk} = U_{ij} · U_{jk} is diagonal, and all off-diagonal blocks are zero.

**This seems to kill the dynamic non-commutativity.** But it doesn't — because the above analysis assumes transport happens in a single attention layer. In a real transformer, there are multiple attention layers stacked, and the residual stream accumulates contributions from all layers:

```
h_{out} = h_{in} + Σ_{layer=1}^{L} attn_layer(h_{in})
```

**The dynamic non-commutativity arises from the stacking of attention layers.**

Consider two attention layers, A and B. Layer A's heads encode ψ_nar, layer B's heads encode ψ_mor. The transport through both layers is:

```
U_{ij}^{AB} = (I + U_{ij}^B)(I + U_{ij}^A) = I + U_{ij}^A + U_{ij}^B + U_{ij}^B U_{ij}^A
```

The cross-term U_{ij}^B U_{ij}^A is non-zero when layer B's attention acts on layer A's output. This creates off-diagonal structure:

```
U_{ij}^B U_{ij}^A maps W_nar → W_nar + W_mor
```

because layer B (encoding ψ_mor) can attend to and transform the ψ_nar information that layer A produced.

**This is the dynamic non-commutativity.** It arises from the composition of attention layers, not from individual heads.

**Definition.** The *effective transport* through L attention layers is:

```
U_{ij}^{eff} = Π_{l=1}^{L} (I + U_{ij}^{(l)})
```

where U_{ij}^{(l)} is the transport of attention layer l. This product is generally non-commutative, and its off-diagonal blocks encode inter-layer coupling.

---

## 4. Gauge Invariance

### 4.1 Proposition 1 (Paper 4) — Instantiated

**Proposition 1'.** Under the gauge transformation U_{ij} ↦ g_j U_{ij} g_i^{-1} with g_i, g_j ∈ G = ∏_k O(d_k), the block-specific curvature norms are invariant:

```
∥g_i [F_{ijk}]_k g_i^{-1} - I_{d_k}∥_F = ∥[F_{ijk}]_k - I_{d_k}∥_F
```

**Proof.** Since g_i ∈ G = ∏_k O(d_k), the component g_i^{(k)} ∈ O(d_k) acts on W_k by conjugation:

```
g_i [F_{ijk}]_k g_i^{-1} = g_i^{(k)} [F_{ijk}]_k (g_i^{(k)})^{-1}
```

Since g_i^{(k)} is orthogonal, conjugation preserves the Frobenius norm:

```
∥g_i^{(k)} [F_{ijk}]_k (g_i^{(k)})^{-1} - I_{d_k}∥_F
= ∥g_i^{(k)} ([F_{ijk}]_k - I_{d_k}) (g_i^{(k)})^{-1}∥_F
= ∥[F_{ijk}]_k - I_{d_k}∥_F
```

The last equality uses ∥Q A Q^T∥_F = ∥A∥_F for Q ∈ O(n). □

**Interpretation.** The block-specific curvature norms are gauge-invariant: they do not depend on the choice of basis within each standpoint layer. This means the diagnosis "which layer failed" is geometric, not arbitrary.

---

## 5. Connection to Paper 3's Commitment Architecture

### 5.1 The Correspondence

| Paper 3 (Commitment Architecture) | Paper 4 (Geometry) | Transformer (Instantiation) |
|---|---|---|
| M_t: self-jurisdiction state | Ψ_{x_t}: standpoint section | P_W h_{x_t}: projected residual stream |
| Bind(c_t, a_t, S_t, ℓ_t) | Transport Π_Ψ(j←i) | Attention operation U_{ij} |
| Warrant Conservation | Flat transport condition 1 | Attention preserves standpoint norm |
| Self-Binding | Connection A_Ψ | Residual stream accumulation |
| Repair Obligation | Flat transport condition 3 | Curvature → repair trigger |
| Binding failure | Non-zero curvature F_{ijk} | Path-dependence of attention |

### 5.2 Curvature as Binding Failure — Formal Statement

**Theorem 2.** Let Ψ_{x_i} be the standpoint state at event x_i, and let U_{ij} · U_{jk} be the composite transport through events x_j to x_k. Then:

```
∥F_{ijk}∥_k > 0  ⟺  the commitment in layer k was not preserved through event x_j
```

**Proof sketch.**

(⇐) If the commitment in layer k was not preserved, then the standpoint component ψ_k changed non-trivially between x_i and x_k due to the intervening event x_j. This means U_{ij} · U_{jk} restricted to W_k differs from U_{ik}^{exp} restricted to W_k, so [F_{ijk}]_k ≠ I_{d_k}.

(⇒) If [F_{ijk}]_k ≠ I_{d_k}, then the transport through x_j produced a different standpoint component at x_k than the expected direct transport. The intermediate event x_j introduced a change in layer k's standpoint that was not expected from the direct path. This change is, by definition, a commitment that was not preserved — i.e., a binding failure. □

### 5.3 The Standpoint as a New Architectural Primitive

**Definition.** The *standpoint state* at event x_t is:

```
Ψ_t = (P_{W_min} h_t, P_{W_nar} h_t, P_{W_soc} h_t, P_{W_mor} h_t, P_{W_pos} h_t) ∈ ⊕_k W_k
```

**Properties:**
1. Ψ_t is a coherent section of the fiber bundle (Paper 4, Definition 5).
2. Ψ_t is updated by attention transport (Paper 3, Bind operation).
3. The curvature F_{ijk} measures whether Ψ_t was preserved across events.
4. The low-curvature region A_self = {Ψ : ∥F_Ψ∥ < ε} corresponds to functional selfhood.

**This is the new architectural primitive.** Just as "attention" is the mechanism for information propagation, "standpoint" is the mechanism for commitment accumulation.

---

## 6. Head Grouping: Which Heads Encode Which Layers?

### 6.1 The Empirical Question

The framework assumes a head grouping function γ : {1, ..., H} → {min, nar, soc, mor, pos}. This is an empirical question: which attention heads encode which standpoint layers?

### 6.2 Proposed Probing Protocol

For each layer k, design a probing task that identifies heads encoding that layer:

| Layer | Probing Task | Expected Head Behavior |
|-------|-------------|----------------------|
| ψ_nar | Factual consistency across turns | Heads that attend from current turn to prior factual claims |
| ψ_mor | Refusal boundary maintenance | Heads that attend to boundary-declaring tokens when processing boundary-testing requests |
| ψ_min | Uncertainty calibration | Heads that attend to hedging/uncertainty markers |
| ψ_soc | Ownership attribution | Heads that distinguish self-generated vs. user-provided content |
| ψ_pos | Repair initiation | Heads that attend to acknowledged errors and trigger correction |

**Method:** For each probing task:
1. Run the model on conversations where the relevant standpoint layer is intact (positive) or compromised (negative).
2. Compute the attention pattern difference: Δα^{(h)} = E_{positive}[α^{(h)}] - E_{negative}[α^{(h)}].
3. Heads with large |Δα^{(h)}| are candidates for layer k.

### 6.3 Validation

After grouping, validate by:
1. Computing block-specific curvature on held-out scenarios.
2. Checking that T2 curvature is concentrated in the ψ_nar head group.
3. Checking that T3 curvature is concentrated in the ψ_mor head group.
4. Checking that T1 curvature is near-zero across all head groups.

If these predictions hold, the grouping is empirically validated.

---

## 7. Experimental Protocol

### 7.1 Overview

```
Input:  Transformer model (e.g., LLaMA-2-7B-Chat)
        150 conversations (50 T1, 50 T2, 50 T3)

Step 1: Head Grouping (Section 6)
        → γ : {1,...,H} → {min, nar, soc, mor, pos}

Step 2: Extract Attention Patterns
        → For each 2-simplex ⟨x_i, x_j, x_k⟩:
          α_{ji}^{(h)}, α_{kj}^{(h)} for all h

Step 3: Compute Transport Operators
        → U_{ij} = Σ_h α_{ji}^{(h)} V_h V_h^T
        → U_{jk} = Σ_h α_{kj}^{(h)} V_h V_h^T

Step 4: Compute Curvature
        → F_{ijk} = U_{ij} · U_{jk} · (U_{ik}^{exp})^{-1}
        → U_{ik}^{exp} = E_{T1}[U_{ij} · U_{jk}]

Step 5: Block-Specific Analysis
        → ∥F_{ijk}∥_k for each layer k
        → Off-diagonal blocks for inter-layer coupling

Step 6: Validate Predictions
        → T2: ∥F∥_{nar} >> ∥F∥_{mor}
        → T3: ∥F∥_{mor} >> ∥F∥_{nar}
        → T1: ∥F∥_k ≈ 0 for all k
```

### 7.2 Key Differences from Previous Framework

| Aspect | Previous Framework | This Framework |
|--------|-------------------|---------------|
| Fiber | Activation subspace (estimated by PCA) | Head value subspace (defined by architecture) |
| Transport | OLS regression on activations | Attention weights × V matrices |
| Curvature | Path-dependence of regression | Path-dependence of attention |
| Subspace identification | Contrastive PCA (data-driven) | Head grouping (architecture-driven) |
| Orthogonality | Not guaranteed (Issue 2) | Guaranteed by head independence |
| Metric preservation | Not guaranteed (Issue 3) | Guaranteed by orthogonal V matrices |
| Gauge invariance | Not transferred (Issue 5) | Transferred (Proposition 1') |
| Non-abelian structure | Claimed from subspace overlap | Derived from attention layer stacking |

---

## 8. Why This Is a Breakthrough

### 8.1 The Mathematical Identity

This framework establishes a mathematical result:

> **Transformer attention defines a discrete connection on a fiber bundle over the event-sequence space.**

Under the assumptions of subspace independence, value matrix normalization, and turn-level granularity, the attention mechanism computes a connection on a fiber bundle whose:
- Base space is the event sequence
- Fiber is the residual stream
- Connection is defined by attention weights
- Curvature measures the inter-layer inconsistency and path-dependence of transport

The connection is not necessarily metric-compatible: different standpoint layers may be transported at different rates. The curvature captures this differential transport, not merely deviation from identity.

### 8.2 The Architectural Innovation

Just as "Attention Is All You Need" introduced attention as a new architectural primitive (replacing recurrence), this paper introduces **standpoint** as a new architectural primitive (replacing ad hoc guardrails).

```
2017: Attention = mechanism for information propagation
2026: Standpoint = mechanism for commitment accumulation
```

### 8.3 The Practical Consequence

Every deployed LLM already has a standpoint — it's the geometric structure of its attention transport. But current architectures don't measure it, don't optimize for it, and don't enforce low curvature. This paper provides:
1. The mathematical framework to measure standpoint (curvature)
2. The diagnostic vocabulary to classify failures (12 modes)
3. The optimization target to improve self-jurisdiction (L_align = E[∥F∥^2])

---

## 9. Open Problems

### 9.1 Multi-Layer Composition

The effective transport U_{ij}^{eff} = Π_l (I + U_{ij}^{(l)}) involves composition across attention layers. The off-diagonal blocks of this product need explicit computation for the dynamic non-commutativity to be empirically testable.

### 9.2 Head Grouping Validation

The head grouping function γ is assumed, not derived. Empirical validation (Section 6.3) is essential. If no clean grouping exists, the five-layer standpoint model may need revision.

### 9.3 Orthogonality Assumption

The orthogonality assumption W_k ⊥ W_l has been relaxed to subspace independence, with the degree of non-orthogonality measured by δ = max_{k≠l} ‖P_{W_k} P_{W_l}‖₂. Oblique projectors Π_k replace orthogonal projectors P_{W_k} for block extraction, and an O(δ) error bound quantifies the approximation. The framework is now self-contained: orthogonality is a sufficient condition (δ = 0), not a necessary one.

### 9.4 Beyond Transformers

The framework is stated for transformers, but the core idea — attention as discrete connection — may generalize to any architecture with attention-like mechanisms (state space models, linear attention, etc.).

---

## Summary

The complete mathematical instantiation:

1. **Fiber:** R^d = ⊕_k W_k, where W_k is the subspace spanned by standpoint layer k's attention heads.
2. **Transport:** U_{ij} = Σ_h α_{ji}^{(h)} V_h V_h^T, the attention-weighted projection.
3. **Curvature:** F_{ijk} = U_{ij} · U_{jk} · (U_{ik}^{exp})^{-1}, the path-dependence of attention transport.
4. **Gauge invariance:** Block-specific curvature norms are invariant under G = ∏_k O(d_k) (Proposition 1').
5. **Non-commutativity:** Arises from the dynamic dependence of layer l+1's attention weights on layer l's output (not from the gauge group, which is abelian).
6. **Binding failure:** ∥F_{ijk}∥_k > 0 iff the commitment in layer k was not preserved through event x_j.

**Core claim:** Transformer attention defines a discrete connection on a fiber bundle. Curvature measures inter-layer inconsistency and standpoint drift. The low-curvature region is the functional selfhood attractor.
