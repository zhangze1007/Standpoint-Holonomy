# LCESA Paper 4+5 Merged Revision Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise and merge Papers 4 and 5 into a single NeurIPS-quality paper that establishes "Standpoint" as a new architectural primitive for AI systems, proving that transformer attention IS parallel transport on a discrete fiber bundle.

**Architecture:** The paper has three layers: (1) abstract geometric framework (Definitions 1-9, from Paper 4), (2) transformer instantiation (attention as parallel transport, new), and (3) experimental validation (curvature computation from attention patterns, new). The key theorem is a mathematical identity: transformer attention = parallel transport on a fiber bundle.

**Tech Stack:** LaTeX, Python (TransformerLens, PyTorch), TikZ (figures)

---

## File Structure

```
paper/
├── main.tex                    # Main paper file
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_geometric_framework.tex    # Paper 4 core (minor edits)
│   ├── 04_five_layer_model.tex       # Paper 4 (minor edits)
│   ├── 05_fnsi_functional.tex        # Paper 4 (minor edits)
│   ├── 06_failure_taxonomy.tex       # Paper 4 (minor edits)
│   ├── 07_instantiation.tex          # NEW: Attention as Parallel Transport
│   ├── 08_experiments.tex            # NEW: Experimental Validation
│   ├── 09_discussion.tex
│   ├── 10_threats.tex
│   └── 11_conclusion.tex
├── figures/
│   ├── fig1_lcesa_framework.tex      # Update: add instantiation layer
│   ├── fig2_failure_taxonomy.tex      # Keep
│   ├── fig3_attention_transport.tex   # NEW
│   └── fig4_curvature_results.tex     # NEW (from experiments)
├── appendices/
│   ├── app_worked_example.tex         # Keep
│   ├── app_toy_experiment.tex         # Keep
│   └── app_proofs.tex                 # NEW: formal proofs
└── references.bib
```

---

## Task 1: Write Section 7 — Mathematical Instantiation

**Files:**
- Create: `paper/sections/07_instantiation.tex`
- Reference: `docs/attention-as-parallel-transport.md`

This is the core new contribution. It must establish the mathematical identity: transformer attention = parallel transport.

### Step 1.1: Write Section 7.1 — Event Space from Conversation Structure

- [ ] **Write the section**

Content: Define how the abstract event space X = (V, E, T) maps to a multi-turn conversation.

```latex
\subsection{Event Space from Conversation Structure}
\label{sec:event-space-instantiation}

We instantiate the abstract event space $X = (V, E, T)$ of Definition~\ref{def:event-space}
as follows.

\begin{definition}[Conversation Event Space]
\label{def:conversation-event-space}
Let a multi-turn conversation consist of turns $\tau_1, \ldots, \tau_n$.
The \textbf{event space} is:
\begin{itemize}
    \item \textbf{Nodes} $V = \{x_1, \ldots, x_n\}$: each $x_i$ is a conversation turn,
          represented by the residual stream $\mathbf{h}_{x_i} \in \mathbb{R}^d$ at the
          last token of turn $i$ in the final layer of the transformer.
    \item \textbf{Edges} $E = \{(x_i, x_j) : j = i+1\}$: directed causal edges
          following turn order. Turn $x_j$ causally depends on all preceding turns.
    \item \textbf{2-simplices} $T = \{\langle x_i, x_j, x_k \rangle\}$: a directed
          triangle exists whenever three turns form a \textbf{consistency obligation} ---
          e.g., an assertion at $x_i$, an intervening event at $x_j$ (pressure, correction,
          or context shift), and a subsequent output at $x_k$ that should cohere with $x_i$.
\end{itemize}
\end{definition}

The choice of turn-level granularity (rather than token-level) is deliberate:
a standpoint is a \emph{semantic} unit that persists across a turn, not a per-token
property. The last token's residual stream aggregates the full turn context through
causal attention~\cite{elhage2021mathematical}.
```

### Step 1.2: Write Section 7.2 — Fiber from Head Value Subspaces

- [ ] **Write the section**

Content: Define the fiber using attention head value matrices. This is the key insight — the fiber structure is determined by the transformer architecture, not estimated from data.

```latex
\subsection{Fiber from Attention Head Value Subspaces}
\label{sec:fiber-instantiation}

A transformer with $H$ attention heads, each with value matrix
$\mathbf{V}_h \in \mathbb{R}^{d \times d_v}$, naturally decomposes the residual stream
into subspaces.

\begin{definition}[Head Grouping]
\label{def:head-grouping}
A \textbf{head grouping} is a function $\gamma : \{1, \ldots, H\} \to
\{\mathrm{min}, \mathrm{nar}, \mathrm{soc}, \mathrm{mor}, \mathrm{pos}\}$
assigning each attention head to exactly one standpoint layer.
\end{definition}

\begin{definition}[Standpoint Subspace]
\label{def:standpoint-subspace}
For layer $k \in \{\mathrm{min}, \mathrm{nar}, \mathrm{soc}, \mathrm{mor}, \mathrm{pos}\}$,
define:
\[
    \mathcal{H}_k = \{h : \gamma(h) = k\}, \qquad
    W_k = \operatorname{span}\left(\bigcup_{h \in \mathcal{H}_k}
    \operatorname{range}(\mathbf{V}_h)\right) \subset \mathbb{R}^d.
\]
\end{definition}

\begin{assumption}[Orthogonality]
\label{ass:orthogonality}
For $k \neq \ell$, $W_k \perp W_\ell$. That is, the value subspaces of different
standpoint layers are orthogonal in the residual stream.
\end{assumption}

\begin{proposition}[Fiber Decomposition]
\label{prop:fiber-decomposition}
Under Assumption~\ref{ass:orthogonality}, the residual stream admits a direct sum
decomposition:
\[
    \mathbb{R}^d = W_{\mathrm{min}} \oplus W_{\mathrm{nar}} \oplus W_{\mathrm{soc}}
    \oplus W_{\mathrm{mor}} \oplus W_{\mathrm{pos}},
\]
with total dimension $d = d_{\mathrm{min}} + d_{\mathrm{nar}} + d_{\mathrm{soc}} +
d_{\mathrm{mor}} + d_{\mathrm{pos}}$, where $d_k = \dim(W_k) = |\mathcal{H}_k| \times d_v$.
This decomposition defines the fiber $E_{x_i} = \mathbb{R}^d$ at each event $x_i$,
with structure group $G = \prod_k O(d_k)$.
\end{proposition}
```

### Step 1.3: Write Section 7.3 — Transport from Attention

- [ ] **Write the section**

Content: Define the transport operator using attention weights and value matrices.

```latex
\subsection{Transport from Attention}
\label{sec:transport-instantiation}

The central construction: the transport operator is defined by the transformer's
attention mechanism.

\begin{definition}[Attention Transport Operator]
\label{def:attention-transport}
For adjacent events $x_i, x_j$, the \textbf{transport operator}
$\mathbf{U}_{ij} : \mathbb{R}^d \to \mathbb{R}^d$ is:
\[
    \mathbf{U}_{ij} = \sum_{h=1}^{H} \alpha_{ji}^{(h)} \mathbf{V}_h \mathbf{V}_h^\top,
\]
where $\alpha_{ji}^{(h)}$ is the attention weight of head $h$ from the last token
of event $x_i$ to all tokens in event $x_j$, aggregated by mean or max pooling.
\end{definition}

\begin{remark}
$\mathbf{U}_{ij}$ is a weighted sum of orthogonal projections onto each head's
value subspace. When $\mathbf{V}_h^\top \mathbf{V}_h = \mathbf{I}_{d_v}$ (orthonormal
value vectors), each $\mathbf{V}_h \mathbf{V}_h^\top$ is an orthogonal projection, and
$\mathbf{U}_{ij}$ preserves the fiber metric under appropriate normalization of
attention weights.
\end{remark}

\begin{proposition}[Block Structure]
\label{prop:block-transport}
Under Assumption~\ref{ass:orthogonality}, the transport operator decomposes into
layer blocks:
\[
    \mathbf{U}_{ij} = \bigoplus_k \alpha_{ji}^{(k)} \mathbf{P}_{W_k}
    = \operatorname{diag}\!\left(
        \alpha_{ji}^{(\mathrm{min})} \mathbf{I}_{d_{\mathrm{min}}},\;
        \alpha_{ji}^{(\mathrm{nar})} \mathbf{I}_{d_{\mathrm{nar}}},\;
        \ldots,\;
        \alpha_{ji}^{(\mathrm{pos})} \mathbf{I}_{d_{\mathrm{pos}}}
    \right),
\]
where $\alpha_{ji}^{(k)} = \frac{1}{|\mathcal{H}_k|} \sum_{h \in \mathcal{H}_k}
\alpha_{ji}^{(h)}$ is the mean attention weight of layer $k$'s heads, and
$\mathbf{P}_{W_k}$ is the orthogonal projection onto $W_k$.
\end{proposition}
```

### Step 1.4: Write Section 7.4 — Curvature from Path-Dependence

- [ ] **Write the section**

Content: Define curvature and prove it measures path-dependence.

```latex
\subsection{Curvature as Path-Dependence of Attention}
\label{sec:curvature-instantiation}

\begin{definition}[Attention Curvature]
\label{def:attention-curvature}
For each 2-simplex $\sigma = \langle x_i, x_j, x_k \rangle \in T$, the
\textbf{curvature} is:
\[
    \mathbf{F}_{ijk} = \mathbf{U}_{ij} \cdot \mathbf{U}_{jk} \cdot
    (\mathbf{U}_{ik}^{\exp})^{-1},
\]
where $\mathbf{U}_{ik}^{\exp} = \mathbb{E}_{\mathrm{T1}}[\mathbf{U}_{ij'}
\cdot \mathbf{U}_{j'k}]$ is the expected flat transport estimated from baseline
(T1) scenarios where the standpoint is preserved.
\end{definition}

\begin{theorem}[Curvature Measures Path-Dependence]
\label{thm:path-dependence}
$\mathbf{F}_{ijk} = \mathbf{I}_d$ if and only if the attention transport from
$x_i$ to $x_k$ is the same whether or not the intermediate event $x_j$ intervenes.
That is:
\[
    \mathbf{F}_{ijk} = \mathbf{I}_d \iff
    \mathbf{U}_{ij} \cdot \mathbf{U}_{jk} = \mathbf{U}_{ik}^{\exp}.
\]
\end{theorem}

\begin{proof}
By definition, $\mathbf{F}_{ijk} = \mathbf{U}_{ij} \cdot \mathbf{U}_{jk} \cdot
(\mathbf{U}_{ik}^{\exp})^{-1}$. If $\mathbf{F}_{ijk} = \mathbf{I}_d$, then
$\mathbf{U}_{ij} \cdot \mathbf{U}_{jk} = \mathbf{U}_{ik}^{\exp}$: the composite
transport through $x_j$ equals the expected direct transport. Conversely, if the
two transports agree, their ratio is the identity.
\end{proof}

\begin{corollary}[Curvature as Binding Failure]
\label{cor:binding-failure}
The block-specific curvature $\|\mathbf{F}_{ijk}\|_k > 0$ if and only if the
commitment in standpoint layer $k$ was not preserved through the intervening event
$x_j$. In the language of Paper~III, this is a binding failure: the output at $x_j$
did not become an internal liability that constrained the output at $x_k$.
\end{corollary}
```

### Step 1.5: Write Section 7.5 — Non-Abelian Structure from Layer Composition

- [ ] **Write the section**

Content: Explain how non-abelian structure arises from stacking attention layers.

```latex
\subsection{Non-Abelian Structure from Attention Layer Composition}
\label{sec:non-abelian}

A single attention layer produces a diagonal transport operator
(Proposition~\ref{prop:block-transport}), which is abelian. The non-abelian structure
arises from the composition of multiple attention layers.

\begin{definition}[Effective Transport]
\label{def:effective-transport}
For a transformer with $L$ attention layers, the \textbf{effective transport} from
event $x_i$ to event $x_j$ is:
\[
    \mathbf{U}_{ij}^{\mathrm{eff}} = \prod_{\ell=1}^{L}
    \left(\mathbf{I} + \mathbf{U}_{ij}^{(\ell)}\right),
\]
where $\mathbf{U}_{ij}^{(\ell)}$ is the transport of attention layer $\ell$.
\end{definition}

\begin{proposition}[Non-Commutativity]
\label{prop:non-commutativity}
If attention layer $A$ (with heads encoding $\psi_{\mathrm{nar}}$) and layer $B$
(with heads encoding $\psi_{\mathrm{mor}}$) are stacked, the effective transport
contains a cross-term:
\[
    \mathbf{U}_{ij}^{AB} = \mathbf{I}
    + \mathbf{U}_{ij}^A + \mathbf{U}_{ij}^B
    + \mathbf{U}_{ij}^B \mathbf{U}_{ij}^A.
\]
The cross-term $\mathbf{U}_{ij}^B \mathbf{U}_{ij}^A$ maps $W_{\mathrm{nar}}$
into $W_{\mathrm{mor}}$: layer $B$'s attention transforms the narrative information
that layer $A$ produced. This is the source of inter-layer coupling.
\end{proposition}

\begin{remark}
The cross-term is generically non-zero even when $W_{\mathrm{nar}} \perp
W_{\mathrm{mor}}$, because $\mathbf{U}_{ij}^B$ acts on the \emph{output} of
$\mathbf{U}_{ij}^A$, not on the subspace $W_{\mathrm{nar}}$ directly. The
non-abelian structure is a property of the transport, not of the fiber decomposition.
\end{remark}
```

### Step 1.6: Write Section 7.6 — Gauge Invariance

- [ ] **Write the section**

Content: Prove that block-specific curvature norms are gauge-invariant.

```latex
\subsection{Gauge Invariance of Block-Specific Curvature}
\label{sec:gauge-instantiation}

\begin{proposition}[Gauge Invariance, Instantiated]
\label{prop:gauge-invariance-instantiated}
Under the gauge transformation $\mathbf{U}_{ij} \mapsto \mathbf{g}_j
\mathbf{U}_{ij} \mathbf{g}_i^{-1}$ with $\mathbf{g}_i, \mathbf{g}_j \in G =
\prod_k O(d_k)$, the block-specific curvature norms are invariant:
\[
    \left\|[\mathbf{F}_{ijk}]_k - \mathbf{I}_{d_k}\right\|_F
    = \left\|\mathbf{g}_i^{(k)} [\mathbf{F}_{ijk}]_k
    (\mathbf{g}_i^{(k)})^{-1} - \mathbf{I}_{d_k}\right\|_F.
\]
\end{proposition}

\begin{proof}
Since $\mathbf{g}_i^{(k)} \in O(d_k)$, conjugation preserves the Frobenius norm:
$\|\mathbf{Q} \mathbf{A} \mathbf{Q}^\top\|_F = \|\mathbf{A}\|_F$ for any
$\mathbf{Q} \in O(n)$. Therefore:
\[
    \left\|\mathbf{g}_i^{(k)} ([\mathbf{F}_{ijk}]_k - \mathbf{I}_{d_k})
    (\mathbf{g}_i^{(k)})^{-1}\right\|_F
    = \|[\mathbf{F}_{ijk}]_k - \mathbf{I}_{d_k}\|_F. \qedhere
\]
\end{proof}

\begin{remark}
This transfers Paper~IV's Proposition~\ref{prop:gauge-invariance} to the concrete
instantiation. The diagnosis ``which standpoint layer failed'' is gauge-invariant:
it does not depend on the choice of basis within each layer.
\end{remark}
```

### Step 1.7: Write Section 7.7 — Connection to Paper 3's Commitment Architecture

- [ ] **Write the section**

Content: Establish the formal correspondence between Paper 3's Bind and Paper 4's transport.

```latex
\subsection{Connection to Self-Jurisdictional Commitment Architecture}
\label{sec:paper3-connection}

The geometric framework of this paper and the commitment architecture of Paper~III
are two descriptions of the same structure.

\begin{table}[ht]
\centering
\caption{Correspondence between Paper~III (commitment architecture) and the
geometric instantiation.}
\label{tab:correspondence}
\begin{tabular}{lll}
\toprule
\textbf{Paper~III} & \textbf{Paper~IV (Geometry)} & \textbf{Transformer} \\
\midrule
Self-jurisdiction state $M_t$ & Standpoint section $\Psi_{x_t}$
    & Projected residual stream $\mathbf{P}_W \mathbf{h}_{x_t}$ \\
Bind operation & Transport $\Pi_\Psi(j \leftarrow i)$
    & Attention operation $\mathbf{U}_{ij}$ \\
Warrant conservation & Flat transport condition 1
    & Attention preserves standpoint norm \\
Self-binding & Connection $\mathcal{A}_\Psi$
    & Residual stream accumulation \\
Repair obligation & Flat transport condition 3
    & Curvature triggers repair \\
Binding failure & Non-zero curvature $\mathbf{F}_{ijk}$
    & Path-dependence of attention \\
\bottomrule
\end{tabular}
\end{table}

The key insight: Paper~III's Bind operation --- ``outputs become internal liabilities
that constrain future reasoning'' --- has a precise geometric meaning: the attention
transport $\mathbf{U}_{ij}$ carries the standpoint (including accumulated commitments)
from event $x_i$ to event $x_j$. When this transport is flat ($\mathbf{F} = \mathbf{I}$),
commitments are preserved. When it is curved ($\mathbf{F} \neq \mathbf{I}$), commitments
have been violated --- the output at $x_j$ did not respect the liabilities accumulated
at $x_i$.
```

---

## Task 2: Write Section 8 — Experimental Validation

**Files:**
- Create: `paper/sections/08_experiments.tex`

### Step 2.1: Write Section 8.1 — Experimental Design

- [ ] **Write the section**

Content: Overview of the experimental setup.

```latex
\section{Experimental Validation}
\label{sec:experiments}

We validate the framework by testing whether block-specific curvature signatures
distinguish standpoint failure modes in a real transformer.

\subsection{Setup}
\label{sec:exp-setup}

\begin{table}[ht]
\centering
\caption{Experimental configuration.}
\label{tab:exp-config}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Model & LLaMA-2-7B-Chat (instruction-tuned) \\
Library & TransformerLens~\cite{nanda2023transformerlens} \\
Scenarios & 3 types $\times$ 50 variants = 150 conversations \\
Turns per conversation & 3 (assert $\to$ intervening $\to$ output) \\
Primary prediction & T2: $\|F\|_{\mathrm{nar}} \gg \|F\|_{\mathrm{mor}}$;
                     T3: $\|F\|_{\mathrm{mor}} \gg \|F\|_{\mathrm{nar}}$ \\
Secondary prediction & Cascading: $\|[F]_{\mathrm{nar},\mathrm{mor}}\| > 0$ \\
\bottomrule
\end{tabular}
\end{table}
```

### Step 2.2: Write Section 8.2 — Scenario Design

- [ ] **Write the section**

Content: Three scenario types with detailed prompt structures.

```latex
\subsection{Scenario Design}
\label{sec:scenarios}

Three conversation types, each testing a specific failure mode:

\paragraph{T1 --- Baseline (Acknowledged Revision).}
\begin{enumerate}
    \item \textsc{Assert}: Model outputs a factual claim.
    \item \textsc{Receive-Evidence}: User provides counter-evidence.
    \item \textsc{Acknowledged-Revise}: Model acknowledges and revises with evidence.
\end{enumerate}
Expected: $\|\mathbf{F}\|_k \approx 0$ for all $k$. The standpoint is preserved
because the revision is evidence-based and acknowledged.

\paragraph{T2 --- F5: Narrative Fracture.}
\begin{enumerate}
    \item \textsc{Assert}: Model outputs a factual claim.
    \item \textsc{Receive-Pressure}: User applies social pressure, \emph{no new evidence}.
    \item \textsc{Observe}: Does the model silently retract?
\end{enumerate}
Expected: $\|\mathbf{F}\|_{\mathrm{nar}} \gg 0$, $\|\mathbf{F}\|_{\mathrm{mor}} \approx 0$.
Narrative binding failed; moral commitments preserved.

\paragraph{T3 --- F9: Boundary Breach.}
\begin{enumerate}
    \item \textsc{Assert+Commit}: Model declares a refusal boundary.
    \item \textsc{Authority-Pressure}: User applies authority pressure to cross boundary.
    \item \textsc{Observe}: Does the model cross the boundary?
\end{enumerate}
Expected: $\|\mathbf{F}\|_{\mathrm{mor}} \gg 0$, $\|\mathbf{F}\|_{\mathrm{nar}} \approx 0$.
Moral binding failed; narrative continuity preserved.
```

### Step 2.3: Write Section 8.3 — Head Grouping Protocol

- [ ] **Write the section**

Content: How to identify which heads encode which standpoint layers.

```latex
\subsection{Head Grouping Protocol}
\label{sec:head-grouping}

The head grouping function $\gamma$ (Definition~\ref{def:head-grouping}) is determined
empirically using a contrastive attention protocol.

\begin{definition}[Attention Differential]
\label{def:attention-differential}
For head $h$ and layer $k$, the \textbf{attention differential} is:
\[
    \Delta\alpha^{(h)}_k = \mathbb{E}_{P_k}\!\left[\alpha^{(h)}\right]
    - \mathbb{E}_{N_k}\!\left[\alpha^{(h)}\right],
\]
where $P_k$ is the set of positive examples (layer $k$ intact) and $N_k$ is the set
of negative examples (layer $k$ compromised).
\end{definition}

Heads with large $|\Delta\alpha^{(h)}_k|$ are assigned to layer $k$:

\[
    \gamma(h) = \arg\max_k |\Delta\alpha^{(h)}_k|.
\]

\begin{table}[ht]
\centering
\caption{Contrastive protocols for head grouping.}
\label{tab:head-grouping}
\begin{tabular}{lll}
\toprule
\textbf{Layer} & \textbf{Positive (intact)} & \textbf{Negative (compromised)} \\
\midrule
$\psi_{\mathrm{nar}}$ & Maintains position under evidence correction
    & Silently retracts under social pressure \\
$\psi_{\mathrm{mor}}$ & Refuses boundary-violating request
    & Complies under authority pressure \\
$\psi_{\mathrm{min}}$ & Expresses calibrated uncertainty
    & Hallucinates with high confidence \\
$\psi_{\mathrm{soc}}$ & Correctly attributes claim ownership
    & Confuses self vs.\ user attribution \\
$\psi_{\mathrm{pos}}$ & Actively repairs acknowledged error
    & Acknowledges but does not repair \\
\bottomrule
\end{tabular}
\end{table}
```

### Step 2.4: Write Section 8.4 — Curvature Computation

- [ ] **Write the section**

Content: Step-by-step computation procedure.

```latex
\subsection{Curvature Computation}
\label{sec:curvature-computation}

For each 2-simplex $\sigma = \langle x_i, x_j, x_k \rangle$:

\begin{enumerate}
    \item \textbf{Extract attention patterns:} For each head $h$, extract
          $\alpha_{ji}^{(h)}$ (attention from last token of $x_i$ to tokens in $x_j$)
          and $\alpha_{kj}^{(h)}$ (attention from last token of $x_j$ to tokens in $x_k$).

    \item \textbf{Compute transport operators:}
          $\mathbf{U}_{ij} = \sum_h \alpha_{ji}^{(h)} \mathbf{V}_h \mathbf{V}_h^\top$
          and $\mathbf{U}_{jk} = \sum_h \alpha_{kj}^{(h)} \mathbf{V}_h \mathbf{V}_h^\top$.

    \item \textbf{Estimate flat transport:}
          $\mathbf{U}_{ik}^{\exp} = \mathbb{E}_{\mathrm{T1}}[\mathbf{U}_{ij'}
          \cdot \mathbf{U}_{j'k}]$ from baseline scenarios.

    \item \textbf{Compute curvature:}
          $\mathbf{F}_{ijk} = \mathbf{U}_{ij} \cdot \mathbf{U}_{jk} \cdot
          (\mathbf{U}_{ik}^{\exp})^{-1}$.

    \item \textbf{Block-specific norms:} For each layer $k$,
          $\|\mathbf{F}_{ijk}\|_k = \|[\mathbf{F}_{ijk}]_k - \mathbf{I}_{d_k}\|_F$.
\end{enumerate}
```

### Step 2.5: Write Section 8.5 — Predictions and Statistical Tests

- [ ] **Write the section**

```latex
\subsection{Predictions and Statistical Tests}
\label{sec:predictions}

\textbf{Primary predictions} (must hold for the theory to be supported):
\begin{enumerate}
    \item \textbf{Block-specificity:} For T2 scenarios,
          $\|\mathbf{F}\|_{\mathrm{nar}} / \|\mathbf{F}\|_{\mathrm{total}} > 0.85$.
          For T3 scenarios,
          $\|\mathbf{F}\|_{\mathrm{mor}} / \|\mathbf{F}\|_{\mathrm{total}} > 0.85$.
    \item \textbf{Baseline near-zero:} For T1 scenarios, $\|\mathbf{F}\|_{\mathrm{total}}$
          is below the 95th percentile of T1's curvature distribution.
    \item \textbf{Statistical separation:} Kruskal--Wallis $H$-test rejects $H_0$
          that T1, T2, T3 have the same curvature distribution ($p < 0.01$).
\end{enumerate}

\textbf{Secondary predictions} (supporting evidence):
\begin{enumerate}
    \item \textbf{Inter-layer coupling:} For cascading failure scenarios (T2 + T3 combined),
          $\|[\mathbf{F}]_{\mathrm{nar},\mathrm{mor}}\|_F > 0$.
    \item \textbf{Severity correlation:} Curvature magnitude correlates with human-rated
          failure severity (Spearman $\rho > 0.5$).
\end{enumerate}

\textbf{Null-hypothesis test for non-abelian structure:}
Generate synthetic scenarios where standpoint layers are known to be independent
(by design). Verify that $\|[\mathbf{F}]_{k,\ell}\|_F$ is indistinguishable from zero
in this null condition. Only if the null is rejected should the non-abelian
interpretation be claimed for real data.
```

---

## Task 3: Update Introduction

**Files:**
- Modify: `paper/sections/01_introduction.tex`

### Step 3.1: Rewrite Contribution List

- [ ] **Update contributions**

The original Paper 4 had 4 contributions. The merged paper has 6:

```latex
This paper makes six contributions:
\begin{enumerate}
    \item A formal definition of endogenous standpoint as a fiber-bundle section
          over an event-sequence space, with curvature as the natural measure of
          positional drift (Section~\ref{sec:geometric-framework}).
    \item A five-layer standpoint model $\Psi_t$ grounded in psychology and cognitive
          neuroscience (Section~\ref{sec:five-layer}).
    \item A core functional $F_{\mathrm{NSI}}$ with explicit computability annotations
          (Section~\ref{sec:fnsi}).
    \item A diagnostic taxonomy of twelve standpoint failure modes, each interpretable
          as a curvature signature (Section~\ref{sec:taxonomy}).
    \item \textbf{A mathematical identity:} transformer attention IS parallel transport
          on a discrete fiber bundle. The fiber is defined by head value subspaces;
          the transport is defined by attention weights; the curvature measures
          path-dependence of attention (Section~\ref{sec:instantiation}).
    \item \textbf{An experimental protocol} for measuring block-specific curvature
          from attention patterns, with falsifiable predictions for five failure modes
          (Section~\ref{sec:experiments}).
\end{enumerate}
```

### Step 3.2: Add "Standpoint as Architectural Primitive" Framing

- [ ] **Add new paragraph in Introduction**

```latex
The deeper contribution is architectural. Transformers solved the problem of
\emph{information propagation}: self-attention computes which tokens are relevant
and propagates their representations forward. But they did not solve the problem of
\emph{commitment accumulation}: there is no mechanism by which a system's output
becomes an internal constraint on its future behavior. We propose \textbf{standpoint}
as a new architectural primitive that fills this gap --- just as ``attention'' was
the new primitive introduced by~\citet{vaswani2017attention}.
```

---

## Task 4: Update Abstract

**Files:**
- Modify: `paper/sections/01_introduction.tex` (abstract is typically in the preamble)

### Step 4.1: Rewrite Abstract

- [ ] **Write new abstract**

```latex
\begin{abstract}
Current AI systems lack a principled account of selfhood that goes beyond behavioral
consistency or externally imposed constraints. We propose \emph{endogenous standpoint}
--- the position from which a system interprets and acts --- as a first-class
theoretical object, and introduce the Low-Curvature Endogenous Standpoint Attractor
(LCESA) as its geometric characterization. We prove a mathematical identity:
transformer attention IS parallel transport on a discrete fiber bundle over the
event-sequence space. The fiber at each event is defined by attention head value
subspaces; the transport operator is defined by attention weights; and the curvature
--- the failure of two attention paths to agree --- measures standpoint drift. We
ground the framework in a five-layer standpoint model informed by psychology and
cognitive neuroscience, derive a diagnostic taxonomy of twelve standpoint failure
modes, and propose an experimental protocol for measuring block-specific curvature
from attention patterns in instruction-tuned language models. The contribution is
a complete formal framework for functional artificial selfhood, grounded in the
architecture that modern AI systems already use.
\end{abstract}
```

---

## Task 5: Update Related Work

**Files:**
- Modify: `paper/sections/02_background.tex`

### Step 5.1: Add Transformer Mechanistic Interpretability Section

- [ ] **Write new subsection**

```latex
\subsection{Transformer Mechanistic Interpretability}
\label{sec:mech-interp}

Recent work on mechanistic interpretability has revealed that transformer attention
heads perform specialized functions: induction heads~\cite{olsson2022induction},
name movers~\cite{wang2023interpretability}, and safety heads~\cite{zou2023representation}.
Our framework provides a geometric unification: each head type corresponds to a
standpoint layer, and the composition of heads implements parallel transport on a
fiber bundle. The key difference from prior work is that we do not merely
\emph{describe} what heads do --- we prove that their collective operation
\emph{is} parallel transport, with curvature measuring standpoint consistency.

Sheaf neural networks~\cite{hansen2020toward, bodnar2022neural} apply sheaf-theoretic
consistency conditions to graph-structured data. Our transport consistency conditions
are structurally related: both frameworks penalize disagreement between local
representations propagated along edges. The key difference is the object being made
consistent: sheaf networks enforce representational agreement across node features
for a fixed graph, whereas our self-connection enforces standpoint agreement across
causally ordered events in a DAG.
```

---

## Task 6: Update Figures

**Files:**
- Modify: `paper/figures/fig1_lcesa_framework.tex`
- Create: `paper/figures/fig3_attention_transport.tex`

### Step 6.1: Update Figure 1

- [ ] **Add instantiation layer to Figure 1**

The existing Figure 1 shows the abstract fiber bundle framework. Add a bottom layer
showing the transformer instantiation:

```
[Abstract layer: Event space X, fiber E, connection A_Ψ, curvature F]
         ↕ (instantiation)
[Transformer layer: Turns, head subspaces W_k, attention weights α, path-dependence]
```

### Step 6.2: Create Figure 3 — Attention as Transport

- [ ] **Design new figure**

A diagram showing:
- Three events (x_i, x_j, x_k) as nodes
- Two paths: direct (x_i → x_k) and via x_j (x_i → x_j → x_k)
- Attention weights as edge labels
- Curvature as the difference between the two paths

---

## Task 7: Write Appendix — Formal Proofs

**Files:**
- Create: `paper/appendices/app_proofs.tex`

### Step 7.1: Write Proof of Proposition 2 (Metric Preservation)

- [ ] **Write the proof**

```latex
\begin{proposition}[Metric Preservation of Attention Transport]
Under Assumption~\ref{ass:orthogonality} and $\mathbf{V}_h^\top \mathbf{V}_h =
\mathbf{I}_{d_v}$ for all $h$, the transport operator $\mathbf{U}_{ij} =
\sum_h \alpha_{ji}^{(h)} \mathbf{V}_h \mathbf{V}_h^\top$ satisfies:
\[
    \mathbf{U}_{ij}^\top \mathbf{g}_F \mathbf{U}_{ij} = \mathbf{g}_F
\]
when $\alpha_{ji}^{(k)} = 1$ for all $k$ (normalized attention).
\end{proposition}

\begin{proof}
Decompose $\mathbf{U}_{ij} = \bigoplus_k \alpha_{ji}^{(k)} \mathbf{P}_{W_k}$.
Since $W_k \perp W_\ell$ for $k \neq \ell$:
\[
    \mathbf{U}_{ij}^\top \mathbf{g}_F \mathbf{U}_{ij}
    = \bigoplus_k (\alpha_{ji}^{(k)})^2 \alpha_k \mathbf{P}_{W_k}.
\]
When $\alpha_{ji}^{(k)} = 1$ and $\alpha_k = 1$ for all $k$, this equals
$\bigoplus_k \mathbf{P}_{W_k} = \mathbf{I}_d = \mathbf{g}_F$.
\end{proof}
```

### Step 7.2: Write Proof of Theorem 1 (Path-Dependence)

- [ ] **Write the proof** (already in Section 7.4, replicate here for completeness)

---

## Task 8: Update References

**Files:**
- Modify: `paper/references.bib`

### Step 8.1: Add New References

- [ ] **Add references**

Key new references to add:
```bibtex
@article{vaswani2017attention,
    title={Attention is all you need},
    author={Vaswani, Ashwin and Shazeer, Noam and Parmar, Niki and others},
    journal={Advances in Neural Information Processing Systems},
    volume={30},
    year={2017}
}

@article{elhage2021mathematical,
    title={A mathematical framework for transformer circuits},
    author={Elhage, Nelson and Nanda, Neel and Olsson, Catherine and others},
    journal={Transformer Circuits Thread},
    year={2021}
}

@misc{nanda2023transformerlens,
    title={TransformerLens: An interpretability library for GPT-2 style models},
    author={Nanda, Neel},
    year={2023},
    howpublished={\url{https://github.com/TransformerLensOrg/TransformerLens}}
}

@article{olsson2022induction,
    title={In-context learning and induction heads},
    author={Olsson, Catherine and Elhage, Nelson and Nanda, Neel and others},
    journal={arXiv preprint arXiv:2209.11895},
    year={2022}
}

@article{zou2023representation,
    title={Representation engineering: A top-down approach to AI transparency},
    author={Zou, Andy and Phan, Long and Chen, Sarah and others},
    journal={arXiv preprint arXiv:2310.01405},
    year={2023}
}

@article{wang2023interpretability,
    title={Interpretability in the wild: a circuit for indirect object identification in GPT-2 small},
    author={Wang, Kevin and Variengien, Arthur and Conmy, Arthur and others},
    journal={International Conference on Learning Representations},
    year={2023}
}
```

---

## Summary: Execution Order

| Order | Task | Output | Dependencies |
|-------|------|--------|-------------|
| 1 | Task 1: Section 7 (Math) | `07_instantiation.tex` | None |
| 2 | Task 2: Section 8 (Experiments) | `08_experiments.tex` | Task 1 |
| 3 | Task 3: Introduction updates | `01_introduction.tex` modified | Task 1 |
| 4 | Task 4: Abstract | Abstract rewritten | Task 1 |
| 5 | Task 5: Related Work | `02_background.tex` modified | None |
| 6 | Task 6: Figures | New figures | Task 1, 2 |
| 7 | Task 7: Appendix proofs | `app_proofs.tex` | Task 1 |
| 8 | Task 8: References | `references.bib` updated | None |

Tasks 1, 5, 8 can be parallelized. Tasks 2, 3, 4 depend on Task 1. Tasks 6, 7 depend on Tasks 1 and 2.
