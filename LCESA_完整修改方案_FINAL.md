# LCESA Paper IV — 完整修改方案 FINAL
**整合来源：** GPT方案一、二（去重提取）+ GPT方案三 + Claude分析  
**整合原则：** 删除重复、删除错误优先级判断、删除过度声明、保留所有有实质价值的修改建议  
**日期：** 2026年5月  

---

## 前言：四份方案的整合逻辑

在读这份方案之前，你需要知道我做了哪些取舍，以及为什么。

**从GPT方案一、二里删除的内容：**
GPT一和二高度重合，有效内容只有四点：H1拆成H1a/H1b、T0重新定位、三层命题结构、以及baseline ensemble的初步构想。其余内容要么重复，要么有一个根本性的优先级错误——把"前瞻预测实验"列为第一优先级。这是PR逻辑而不是科学逻辑：你还没确立curvature测量的是有意义的内部状态，就去做预测实验，即便预测成功，审稿人也可以说预测的只是prompt风格规律，和standpoint理论无关。因此前瞻预测实验从本方案中删除，移到vast.ai之后的Paper V范畴。

**从GPT方案三里保留的内容：**
方案三是三份GPT方案里最成熟的，第一次正面回应了数学内部问题。保留：vector bundle声明、baseline ensemble设计、三层数据分离、null grouping controls、PCA重构方向、T0控制实验。删除：T0六组控制实验被列为与baseline ensemble并列优先级——这个顺序是错的，T0控制实验必须在baseline ensemble证明T0 anomaly是baseline-robust之后才有意义做。

**从Claude分析里保留的全部内容：**
这份分析挖掘了其他三份方案完全没有触碰的数学内部问题，全部保留：holonomy术语修正、主丛/向量丛不兼容修复、值矩阵正交性假设验证、H8.19/H8.20/H8.21缺失问题处理、自引文献处理、U_exp数据循环问题的正确解法。

**贯穿全文的核心原则：**
本方案的所有修改围绕同一个逻辑主轴——**从"强五层定位理论"降阶到"全局几何复杂度信号 + 条件性诊断投影"**。这不是妥协，是把理论收敛到证据真正支持的层级。降阶之后论文不再依赖一个被自己的数据否定了的预测，反而更稳。

---

## 第一部分：数学与术语层面的修改

这部分的修改是最紧迫的，因为它们不需要任何新实验，只需要改文字，但如果不改，会在审稿阶段被直接击穿。

---

### 修改1：全文术语替换——"curvature tensor" → "discrete holonomy deviation"

**问题的性质：**
论文把 $F_{ijk} = U_{ij} \cdot U_{jk} \cdot (U^{\exp}_{ik})^{-1}$ 称为"curvature tensor"。这在数学上是不准确的。经典微分几何的曲率张量（curvature 2-form）是联络1-形式的局部无穷小量，通过 $\mathcal{F} = dA + A \wedge A$ 定义，作用在切向量对上，是逐点的微分量。论文定义的 $F_{ijk}$ 是有限离散三角形 $\langle x_i, x_j, x_k \rangle$ 上的完整路径偏差，数学上这是holonomy element，测量沿一个闭合路径的完整传输偏差，不是局部微分量。两个概念在数学层级上是不同的：曲率是holonomy的无穷小极限，但holonomy不等于曲率。对微分几何背景的审稿人，这个词的错误使用会直接触发信任危机，且无法自圆其说。

**修改范围：**
全文所有出现"curvature tensor"的地方，替换成"discrete holonomy deviation"。出现"curvature"单词但指代 $F_{ijk}$ 或 $\|F_{ijk}\|_k$ 的地方，替换成"holonomy deviation"。"block-specific curvature"替换成"block-specific holonomy norm"。标题"Curvature Geometry of Cognitive States"需要修改，建议改为"Holonomy Geometry of Cognitive States in Transformer Internal Representations"。

**需要新增的Remark（插入Section 3.4 Definition 3.7之后）：**

> *Remark 3.8 (Discrete holonomy vs. differential curvature). The quantity $F_{ijk}$ defined above is a finite holonomy element over the discrete triangle $\langle x_i, x_j, x_k \rangle \in \mathcal{T}$. It measures the failure of path-independent transport around a completed discrete loop: transporting from $x_i$ through $x_j$ to $x_k$ via the attention mechanism, and comparing against the expected direct transport $U^{\exp}_{ik}$. This differs from the differential curvature 2-form $\mathcal{F} = dA + A \wedge A$ of classical gauge theory, which is a local infinitesimal quantity defined on the tangent bundle of a smooth manifold. The present framework is a discrete analogue: the event space $\mathcal{X} = (V, E, \mathcal{T})$ is a simplicial complex, not a smooth manifold, and the transport operators are finite linear maps, not differential forms. We use "holonomy deviation" throughout to denote $F_{ijk}$ and its block-specific norms $\|F_{ijk}\|_k$, following the convention in discrete differential geometry (Crane et al., 2013). The term "curvature" appears in informal usage to denote the magnitude of holonomy deviation as a scalar diagnostic; all formal definitions use the precise holonomy terminology.*

**注意：** Abstract里的"curvature analysis"和"curvature vectors"也需要对应修改，改成"holonomy analysis"和"holonomy deviation vectors"。Results部分里"curvature discriminates"改成"holonomy deviation discriminates"。

---

### 修改2：结构群不兼容问题——主丛框架降阶为向量丛联络

**问题的性质：**
这是本论文数学部分最严重的内部不一致，GPT三份方案都没有完整处理，只有Claude分析识别了它。

论文Section 3.3声明结构群为：
$$G = \prod_{k \in K} O(d_k)$$

其中 $O(d_k)$ 是 $W_k$ 上的正交群。主丛联络的规范理论要求传输算子是结构群的群元素，即 $U_{ij}^{(k)} \in O(d_k)$，也就是说传输算子必须是正交变换（保长度的线性映射）。

但论文Proposition 7.12证明，在块对角结构下传输算子实际上是：
$$U_{ij} = \sum_{k \in K} \bar{\alpha}^{(k)}_{ji} P_{W_k}$$

其中 $\bar{\alpha}^{(k)}_{ji}$ 是标量注意力权重均值，取值范围是 $[0,1]$。这意味着在每个块 $W_k$ 上，传输算子是 $\bar{\alpha}^{(k)}_{ji} \cdot \mathrm{Id}_{d_k}$，即标量乘法。标量乘法映射 $v \mapsto \lambda v$（$\lambda \neq 1$）不是正交变换（它不保长度），所以 $\bar{\alpha}^{(k)}_{ji} \cdot \mathrm{Id}_{d_k} \notin O(d_k)$（除非 $\bar{\alpha} = 1$）。传输算子实际属于 $GL(d_k, \mathbb{R})$，不属于 $O(d_k)$。

这意味着论文把一个向量丛联络（vector bundle connection with linear transport）硬套进了主丛框架（principal bundle connection）。两者在理论上不兼容。需要说明的是：Gauge invariance的证明（Proposition 3.8、7.24）仍然是正确的——它证明的是在基底变换下块特定的holonomy norm不变，这对任何线性联络都成立，不需要主丛框架。证明本身没有问题，但框架定位需要修正。

**修改动作：**

在Section 7开头（Proposition 7.7之前，Section 7.3开始之前）插入以下段落：

> *Remark 7.2.5 (Framework clarification: vector bundle connection). The transport operators $U_{ij}$ constructed in Section 7.3 lie in $GL(d_k, \mathbb{R})$ on each standpoint block, not in $O(d_k)$. Specifically, under the block-diagonal structure (Proposition 7.12), the transport on block $W_k$ is $\bar{\alpha}^{(k)}_{ji} \cdot \mathrm{Id}_{d_k}$, which is a scalar scaling map and generally not an orthogonal transformation unless $\bar{\alpha}^{(k)}_{ji} = 1$. The present framework is therefore best understood as a vector bundle with a linear connection, rather than a principal bundle with a $G$-connection. The structure group $G = \prod_k O(d_k)$ defines the gauge symmetry of basis choices within each standpoint subspace — the freedom to rotate the orthonormal basis of $W_k$ without changing the holonomy norms — but $G$ is not the holonomy group of the connection. All results in this paper, including Gauge Invariance (Propositions 3.8, 7.24) and the Block Decomposition (Lemma 3.9, Proposition .4), hold for this more general vector-bundle setting. The gauge invariance proofs are unaffected: they show that $\|[F_{ijk}]_k - \Pi_k\|_F$ is invariant under $G$-conjugation, which is a property of orthogonal basis transformations within $W_k$, not of the full holonomy group.*

同时，在Section 3.3的Definition 3.5（Structure Group）之后，修改现有文字，把"The fiber metric $g_F$ is $G$-invariant"这句话扩展成：

> *The fiber metric $g_F$ is $G$-invariant, and the structure group $G$ defines the gauge freedom: two connection representations related by $U_{ij} \mapsto g_j U_{ij} g_i^{-1}$ (with $g_i, g_j \in G$) are gauge-equivalent. This gauge symmetry is the basis for the gauge-invariant diagnostic framework developed in Section 5. We note that the actual transport operators constructed from transformer attention (Section 7.3) are scalar scalings on each block, lying in $\mathbb{R}_{>0} \subset GL(d_k, \mathbb{R})$ rather than $O(d_k)$; the framework is accordingly a vector bundle connection with $G$-gauge symmetry, as clarified in Remark 7.2.5.*

---

### 修改3：值矩阵正交性假设——从隐含假设到明确的可验证声明

**问题的性质：**
Definition 7.11的传输算子依赖 $V_h^T V_h = \mathrm{Id}_{d_v}$（值矩阵的列向量正交归一）。这是推导 $V_h V_h^T = P_{\mathrm{range}(V_h)}$（正交投影）的关键步骤。但真实transformer的值投影矩阵没有这个约束——它们是通过梯度下降训练的，不保证正交性。论文完全没有实验验证这个假设在GPT-2或Llama-7b上的实际满足程度。

**修改动作（两处）：**

第一处，在Definition 7.11之后加一个Remark：

> *Remark 7.11.1 (Orthonormality assumption). The derivation of $V_h V_h^T = P_{\mathrm{range}(V_h)}$ requires $V_h^T V_h = \mathrm{Id}_{d_v}$, i.e., the columns of $V_h$ are orthonormal. This is not enforced by transformer training. We assess this assumption empirically by computing, for each head $h$, the deviation $\epsilon_h = \|V_h^T V_h - \mathrm{Id}_{d_v}\|_F$. The mean and distribution of $\epsilon_h$ across all heads are reported in Appendix B. When $\epsilon_h$ is small (columns are approximately orthonormal), the derived block-diagonal structure holds approximately. When $\epsilon_h$ is large, the transport operator $U_{ij}$ acquires off-diagonal cross-block contributions beyond those captured by the non-orthogonality parameter $\delta$. We find that [result to be filled after measurement]; all holonomy computations use the empirically corrected projectors described in Appendix B.*

第二处，在Appendix（新建Appendix B）里添加具体测量报告，步骤如下：
- 对GPT-2所有144个head，计算 $\epsilon_h = \|V_h^T V_h - \mathrm{Id}_{d_v}\|_F$
- 报告均值、中位数、最大值
- 对Llama-7b同样操作（1024个head）
- 如果 $\bar{\epsilon} < 0.1$，可以说"approximately satisfied"；如果 $\bar{\epsilon} > 0.3$，需要在正文讨论这对块对角近似的影响

这个测量计算量极小，在CPU上即可完成，不需要新的对话数据。

---

### 修改4：H8.19、H8.20、H8.21结果缺失——选择性报告问题处理

**问题的性质：**
Section 8.5明确陈述了9个假设（H8.16到H8.24），并在Table 5里列出了全部9个预测和统计检验。但Section 9的Table 6只报告了7个结果，H8.19（动态非交换性置换检验）、H8.20（严重性相关性）、H8.21（null置换检验）完全没有出现，没有任何解释说明这三个假设的结果在哪里、为什么没报告。

这在审稿阶段会被直接认定为选择性报告（selective reporting）。即使这三个实验真的没有跑，也必须明确说明并解释原因，不能让它们从论文里消失。

**修改动作：**

把Table 6替换成以下完整版本，覆盖所有9个假设：

| ID | 预测 | 统计检验 | GPT-2结果 | Llama-7b结果 | 状态 |
|---|---|---|---|---|---|
| H8.16 (H1) | Block specificity (dominant ratio > 0.85) | Dominant-layer ratio | × (ratio ≈ 0.203) | × (ratio ≈ 0.196) | Not confirmed |
| H8.17 (H2) | Baseline near-zero | Mann-Whitney U | ✓ (d=+0.472) | Partially ✓* | Partially confirmed |
| H8.18 (H3) | Scenario discrimination | Kruskal-Wallis ε² | ✓ (ε²=0.64–0.69) | ✓ (ε²=0.42) | Confirmed |
| H8.18/layer (H4) | Per-layer discrimination | KW per layer | ✓ (12/12 layers) | ✓ (32/32 layers) | Confirmed |
| H8.22 (H5) | Ablation sensitivity | Spearman ρ | ✓ (ρ≥0.94) | Deferred† | Partially confirmed |
| H8.23 (H6) | Diagnostic superiority | Cohen's d | ✓ (d=+0.472) | Partially ✓* | Partially confirmed |
| H8.24 (H7) | T0 anomaly | Mann-Whitney U | ✓ (d=0.883) | ✓ (d=1.743) | Confirmed |
| H8.19 | Dynamic non-commutativity | Permutation test | **Not run** | **Not run** | Deferred to Paper V |
| H8.20 | Severity correlation | Spearman ρ | **Not run** | **Not run** | Deferred to Paper V |
| H8.21 | Null grouping test | Permutation null | **Not run** | **Not run** | Deferred to Paper V |

*受T0 anomaly影响，详见Section 9.2和9.7  
†Llama-7b ablation需要额外GPU资源，deferred

在Table 6之后加一段解释：

> *Hypotheses H8.19–H8.21 require permutation-based tests over head groupings that were not completed within the computational scope of the present paper. These hypotheses are explicitly deferred to Paper V, which will also include causal validation via activation patching. The non-reporting of H8.19–H8.21 is a computational limitation, not a selective reporting decision; the hypotheses remain open and are retained in the framework for future evaluation.*

---

### 修改5：自引文献问题处理

**问题的性质：**
论文大量引用 Zhang (2025a) 和 Zhang (2025b)，两者都标注为"Manuscript in preparation"，审稿人无法核查这些文献。更严重的是，Section 7多次说"following Zhang (2025a), Definition X"、"as in Zhang (2025a), Proposition 1"，这些引用直接影响了当前论文的数学推导的可追溯性。如果编辑或审稿人要求核查，你无法提供。

**修改动作（分两层）：**

第一层，检查所有引用Zhang (2025a/b)的地方，分成两类：
- A类：当前论文里已经完整重新陈述了对应定义/命题的——改成"(following the notation of Zhang, in prep.; reproduced here as Definition X for self-containedness)"
- B类：当前论文里没有重新陈述、直接依赖外部文献的——必须把对应内容补充到当前论文的正文或附录里，然后改成A类处理

第二层，在论文首页脚注或Acknowledgments前加一句：

> *The companion papers Zhang (2025a, 2025b) formalize the abstract geometric and logical foundations of the LCESA framework and are currently under preparation for concurrent submission. All definitions and propositions cited from these works are reproduced in full in the present paper for self-contained readability; no result in the present paper depends on the companion papers for its proof.*

这句声明虽然增加了当前论文的自洽性要求（你需要确认所有引用的内容确实在当前论文里有完整陈述），但它能有效降低审稿人对循环自引的顾虑。

---

## 第二部分：理论框架的核心重构

这部分处理的是论文的理论定位，是写作层面最重要的工作。

---

### 修改6：五层standpoint从"本体模块"降阶为"诊断投影坐标系"

**为什么必须做这个修改：**
H1是论文最直接的核心预测：特定failure mode应该在对应的standpoint block产生局部化高holonomy。这个预测被完全否定——五个block的相关系数 r > 0.97，意味着它们测量的是同一个全局信号的五种线性投影。这不是一个可以用"the moral block captures general reasoning intensity"一句话轻描淡写绕过去的问题，它动摇了五层本体独立性的核心主张。如果不在理论层面正面处理，论文的整体可信度会受影响。

**修改Section 4的开头段落（现有文字）：**

现有文字把五层介绍为"functionally independent subspaces"。需要在Section 4开头插入一个重新定位段落：

> *The five-layer standpoint model introduced in this section is proposed as a theoretically motivated diagnostic coordinate system for analyzing transformer internal geometry. Each layer corresponds to a psychologically grounded dimension of selfhood (Sections 4.1–4.5), and together they define a projection basis $\{\Pi_k\}_{k \in K}$ for decomposing the total holonomy deviation. We make no claim that transformer internals are ontologically organized into five independent modules — the empirical results in Section 9.1 show that the five block-specific holonomy norms are highly correlated ($r > 0.97$), indicating a dominant global mode rather than five independent dimensions. The five-layer decomposition is therefore interpreted as an interpretable projection basis onto psychologically meaningful axes, not as a decomposition into causally independent cognitive processes. Its diagnostic value lies in providing a structured vocabulary for describing where holonomy concentrates, while the primary empirical signal is global in nature.*

**修改Section 9.1（H1 Failure）的结论段落：**

现有文字在报告H1失败后说"This suggests that the moral reasoning pathway captures a general 'reasoning intensity' signal"，然后直接进入下一个假设。这个结论需要扩展成一个理论重新定位段落：

> *The failure of H1, together with the inter-block correlation structure ($r > 0.97$ for all block pairs in both models), indicates that the five standpoint projections share a dominant global mode. The block-specific holonomy norms $\|F_{ijk}\|_k$ are not measurements of five independent cognitive processes; they are projections of a single global coherence signal onto five interpretable axes. This is consistent with the superposition hypothesis in mechanistic interpretability (Elhage et al., 2022): transformer representations encode many features in high-dimensional superposition rather than in cleanly separated subspaces, making localized block-specific measurements an approximation of a fundamentally distributed signal. The theoretical framework of Section 3 remains valid — the fiber bundle decomposition and gauge invariance results hold regardless of whether the subspaces are functionally independent. What changes is the interpretation: we reframe the five standpoint layers from ontologically independent modules to a diagnostic projection coordinate system (see Section 10.2). Under this reframing, the primary empirical contribution of the holonomy measure is as a global coherence signal, and the five-axis decomposition provides structured diagnostic vocabulary for exploring the geometry of this signal.*

**修改Section 10，新增10.2小节（Global Coherence Geometry Reframing）：**

在Discussion的Limitations之后，插入一个新的小节：

> *10.2 Theoretical Reframing: From Five-Block Decomposition to Global Coherence Geometry*
>
> *The empirical evidence warrants a principled reframing of the LCESA framework. The original theoretical vision — that transformer internals organize into five independently measurable standpoint dimensions — is not confirmed by the data. What the data do support is a different and arguably more interesting structure: a single dominant coherence manifold with interpretable projection axes.*
>
> *Concretely, a principal component analysis of the block holonomy vectors $(\|F\|_{\min}, \|F\|_{\mathrm{nar}}, \|F\|_{\mathrm{soc}}, \|F\|_{\mathrm{mor}}, \|F\|_{\mathrm{pos}})$ across all 270 conversations reveals that the first principal component accounts for the dominant variance (see Section 9.8 and Appendix C). This first component is the "global complexity mode" — a scalar summary of how much the attention transport deviates from expected paths, regardless of which standpoint layer is examined. The residual components may carry scenario-specific diagnostic signal, but this is a secondary structure superimposed on the dominant global mode.*
>
> *We interpret this structure as follows. Transformer attention does not route cognitive states through five independent channels. Instead, cognitive state differences manifest as distributed perturbations across the entire residual stream geometry, which our five-axis projection system captures from five psychologically motivated viewing angles. The moral block happens to have the highest projection weight in the global mode — not because moral processing is uniquely active across all scenarios, but because the value subspaces assigned to the moral standpoint heads happen to capture a larger fraction of the global coherence variance in both models tested.*
>
> *This reframing does not undermine the framework's value; it redirects it. The holonomy measure $\|F_{ijk}\|$ is a valid and robust global diagnostic. The five-axis projection provides interpretable vocabulary. Together they constitute a measurement system for transformer internal geometry, not a decomposition of independent cognitive processes.*

---

### 修改7：H1的降阶处理——H1a和H1b

**这个修改来自GPT方案一/二的有效建议，现在结合上面的理论重构来实施。**

把Section 8.5的Hypothesis 8.16替换成以下两个假设：

> *Hypothesis 8.16a (Block Sensitivity, weak version). For scenario T2 (Narrative Fracture), the narrative block shows a relatively higher holonomy norm than the global mean, even if it is not the absolute maximum across all blocks:*
> $$\|F_{123}\|_{\mathrm{nar}} > \frac{1}{|K|}\sum_{k \in K} \|F_{123}\|_k \quad \text{for at least 60\% of T2 conversations.}$$
> *An analogous condition holds for T3 (moral block), T4 (social block), and T5 (positional block).*
>
> *Hypothesis 8.16b (Block Localization, strong version, original H1). For scenario T2, the narrative block curvature is the strict maximum:*
> $$\frac{\|F_{123}\|_{\mathrm{nar}}}{\max_{k \in K} \|F_{123}\|_k} > 0.85 \quad \text{for at least 80\% of T2 conversations.}$$
> *This is the original prediction of Hypothesis 8.16; it is evaluated but is not required for the framework's main empirical claims.*

在Section 9.1报告时，先报告H1b（失败），再报告H1a的结果（如果成立，说明五轴投影仍有方向性信息；如果也不成立，就只保留"全局信号"这个结论）。

---

### 修改8：T0的定位修改——从"negative control"到"retrieval-complexity condition"

**修改Definition 8.3（T0的定义）最后一段：**

把现有的：
> *Expected curvature: Low, as no standpoint disruption occurs. This scenario serves as a negative control to verify that the curvature signal is specific to failure conditions.*

改成：

> *Original expected holonomy deviation: Low, as no standpoint disruption occurs. This scenario was designed as a negative control. However, as reported in Section 9.7, T0 produces significantly higher holonomy deviation than all failure scenarios in both models (GPT-2: $d = 0.883$; Llama-7b: $d = 1.743$). We therefore reinterpret T0 as a retrieval-complexity condition: a conversation type with no stance pressure but high unconstrained knowledge-retrieval demands. The T0 result is retained and analyzed as a primary finding rather than a control condition. Its theoretical implications for the relationship between internal geometric complexity and external pressure are discussed in Section 10.3.*

**修改Abstract中T0的描述：**
把"pure factual Q&A (T0)"的描述改成"unrestricted factual retrieval (T0, functioning as a retrieval-complexity condition)"，并确保在Abstract的findings里把T0 anomaly列为核心发现之一，而不是footnote。

---

### 修改9：论文整体三层命题结构的重写

**这是GPT方案一/二里最有价值的结构建议，现在结合以上所有修改实施。**

在Introduction的最后一段之前，明确插入一个"论文主张层次"段落（可以放在Section 1的结尾）：

> *This paper makes claims at three levels, with different degrees of empirical support:*
>
> ***Mathematical claims (supported by proof):*** *Transformer attention defines a vector bundle connection over a discrete event-sequence space. The holonomy deviation $F_{ijk}$ is a computable, gauge-invariant measure of path-dependent transport. Block-specific holonomy norms $\|F_{ijk}\|_k$ are gauge-invariant diagnostics (Propositions 3.8, 7.24). The five-layer projection basis provides an interpretable coordinate system for this diagnostic space.*
>
> ***Empirical claims (supported by the experiments in this paper):*** *Holonomy deviation vectors reliably discriminate between six conversation scenario types across two architecturally distinct models, at every layer (H3/H4, $\varepsilon^2 = 0.42$–$0.68$, $p < 10^{-14}$). Unrestricted factual retrieval (T0) produces higher global holonomy than adversarial failure scenarios in both models (H7, $d = 0.883$–$1.743$). Block-specific localization of holonomy is not confirmed; the dominant signal is global, consistent with distributed representation (H1: not confirmed).*
>
> ***Open claims (requiring future work):*** *Whether the five diagnostic projection axes carry independent information beyond the global mode (weak H1a, pending PCA analysis). Whether holonomy deviation has causal status as an internal state indicator, vs. being a correlate of prompt-surface features (requires activation patching, Paper V). The mechanism underlying the T0 anomaly (requires controlled factual retrieval experiments, Paper V).*

---

## 第三部分：不需要GPU的实验补充

以下四个实验全部用现有270个对话和现有pipeline完成，主要是重跑现有数据的不同子集或不同分析，CPU即可。

---

### 实验A：Baseline Ensemble——验证结论对U_exp选择的稳健性

**为什么这是最高优先级：**
U_exp（期望平坦传输）是整个holonomy计算的参照系。它从T1对话里估计。如果T1的选择影响了结论，所有结果都是参照系依赖的而非客观的。T0 anomaly尤其依赖这个问题：T0的高holonomy是相对于T1的低holonomy成立的——如果T1本身就是一个特别低holonomy的outlier baseline，T0 anomaly可能是baseline artifacts。**必须先验证这个问题，再决定T0 anomaly是否值得投入机制控制实验。**

**具体步骤：**

步骤1：把45个T1对话（已有数据）用不同随机种子（seed 0, 1, 2, 3, 4）随机分成5个不重叠的子集，每组9个。

步骤2：对每个子集 $S_m$（$m = 1, \ldots, 5$），分别估计baseline transport：
$$\hat{U}^{\exp,(m)}_{ik} = \frac{1}{9} \sum_{c \in S_m} U^{(c)}_{12} \cdot U^{(c)}_{23}$$

步骤3：对每个 $\hat{U}^{\exp,(m)}_{ik}$，重新计算所有270个对话的holonomy deviation，重跑以下检验：
- H3（Kruskal-Wallis scenario discrimination）：报告ε²
- H7（T0 vs T1 Mann-Whitney U）：报告Cohen's d和T0排名

步骤4：汇总报告，需要包括：
- H3 ε²在5个baseline下的均值和范围（如 $\bar{\varepsilon}^2 = 0.45 \pm 0.03$）
- T0排名在5个baseline下是否一致（是否每次都是最高）
- H7 Cohen's d的均值和范围

**判断标准：**
- H3 ε² > 0.30 in all 5 baselines → H3结论 baseline-robust，可以在论文里说
- T0 rank = 1 in ≥ 4/5 baselines → T0 anomaly baseline-robust，可以投入机制控制实验
- T0 rank ≠ 1 in ≥ 2/5 baselines → T0 anomaly不稳健，需要重新定位，不能作为核心发现

**在论文里的呈现：** 在Section 9新增9.9节"Baseline Robustness Analysis"，报告5组baseline的结果范围，并声明：

> *We confirm that the scenario discrimination (H3) and T0 anomaly (H7) results are robust to baseline selection: across five independent random subsamples of the T1 conversations used to estimate $U^{\exp}_{ik}$, the Kruskal-Wallis effect size ranges from [X to Y] and T0 ranks [first/consistently first] in [N/5] subsamples. All conclusions in Sections 9.3 and 9.7 are baseline-robust.*

---

### 实验B：Null Grouping Controls——验证learned grouping的统计非平凡性

**为什么必须做：**
Head grouping γ的设计有循环性：γ用T2/T3的attention differential来定义（T2提高narrative heads的敏感性，T3提高moral heads的敏感性），然后论文用同一框架来测量T2/T3的block-specific holonomy。H1仍然失败说明这个循环性没有保证结果，但审稿人仍然可以质疑"learned grouping是否优于任何grouping"。这个实验直接回应这个质疑。

**具体步骤：**

步骤1：构建三种null grouping（每种生成20个随机实例取平均）：
- **Random grouping**：把1024个head（Llama-7b）或144个head（GPT-2）随机分配到5层，保持每层head数量和learned γ相同（即 $|H_k^{\mathrm{null}}| = |H_k^{\mathrm{learned}}|$），确保分组的规模结构一致
- **Shuffled grouping**：保持learned γ的头数分布，但随机打乱哪个头属于哪层（完全随机置换）
- **Layer-uniform grouping**：按照transformer的物理层（前1/5层→ψ_min，后1/5层→ψ_pos，以此类推），不用attention differential来分配

步骤2：对每种null grouping，重新计算所有270个对话的holonomy deviation，重跑H3（Kruskal-Wallis）。

步骤3：用置换检验（1000次随机置换）建立null分布，计算learned γ的H3 ε²是否显著高于null分布的第99百分位数。

**判断标准：**
- Learned γ的ε² > 99th percentile of null distribution → learned grouping有统计上非平凡的贡献，五层分解作为诊断坐标系有实验支撑
- Learned γ的ε² ≈ null distribution均值 → learned grouping和随机grouping效果相当，五层分解的诊断价值极为有限，需要在论文里大幅弱化相关声明

**在论文里的呈现：** 在Section 8.3（Head Grouping Protocol）里新增：

> *To verify that the learned grouping $\gamma$ provides a non-trivial diagnostic decomposition, we compare it against three null groupings [...].*

结果报告在Section 9.1之后（H1 failure之后），作为H1 failure的补充说明。

---

### 实验C：PCA分解分析——量化"一个主模态"的声明

**为什么需要这个实验：**
Section 10的理论重构声明了"一个主模态 + 诊断投影轴"，但这个声明需要定量支撑。r > 0.97只说明块之间高度相关，PCA才能精确回答第一主成分解释多少方差，以及残余轴是否携带有意义的信息。

**具体步骤：**

步骤1：对所有270个对话，构建 $270 \times 5$ 的block holonomy矩阵，行是对话，列是5个block的holonomy norm（$\|F\|_{\min}, \|F\|_{\mathrm{nar}}, \|F\|_{\mathrm{soc}}, \|F\|_{\mathrm{mor}}, \|F\|_{\mathrm{pos}}$）。

步骤2：在标准化（z-score每列）之后做PCA，计算5个主成分的解释方差比例。

步骤3：分析PC1和PC2的loading向量（哪个block在哪个主成分里权重最高）。

步骤4：用PC1 score（每个对话在第一主成分上的投影）单独重跑H3（Kruskal-Wallis），报告ε²，对比用完整5维向量的H3 ε²。

步骤5：检验PC2（和更高成分）的scenario区分能力：在控制PC1的情况下，残余成分是否仍然能区分T2/T3/T4/T5（partial eta-squared或偏相关分析）。

**判断标准和对论文声明的影响：**
- PC1解释 > 95% 方差，PC2的scenario区分能力接近零 → "全局一维信号"，五轴分解仅作为参考坐标系，在论文里弱化五轴独立性声明到最低
- PC1解释 80–95% 方差，PC2有统计显著的scenario区分能力 → "主模态 + 弱二阶结构"，可以声明五轴分解提供了有限的额外诊断信息
- PC1解释 < 80% → 不寻常，需要重新检查数据处理流程

**在论文里的呈现：** 在Section 9.1之后新增Section 9.2（重新编号当前的9.2–9.8），标题为"Principal Component Structure of Block Holonomy"：

> *To quantify the global-mode hypothesis, we perform PCA on the 270×5 block holonomy matrix. The first principal component accounts for [X]% of total variance (95% CI: [...]). PC1 loadings are approximately uniform across the five standpoint blocks ($\ell_{\mathrm{min}} = [\cdot], \ell_{\mathrm{nar}} = [\cdot], \ldots$), confirming that PC1 captures a global coherence signal rather than any single standpoint dimension. The scenario discrimination power of PC1 alone is $\varepsilon^2 = [\cdot]$, [comparable to / slightly below] the full five-dimensional result ([$\varepsilon^2 = 0.42$–$0.68$]), indicating that [the bulk of diagnostic information is captured by the global mode / a small additional contribution from higher components exists]. Residual components PC2–PC5 together account for [$\cdot$]% of variance; their scenario discrimination power after controlling for PC1 is $\varepsilon^2 = [\cdot]$ ($p = [\cdot]$).*

---

### 实验D：竞争基准对比——证明holonomy框架的独特贡献

**为什么必须做：**
H3/H4是论文最强的经验结果，但它们现在可以被一个简单的替代解释攻击："不同的prompt template产生不同的attention pattern，curvature只是这个差异的旁观者统计——任何基于attention的测度都会得到同样的区分结果。"你需要证明holonomy deviation提供了比trivial baseline更多的信息，或者至少说清楚它与trivial baseline的关系。

**具体步骤：**

步骤1：对每个对话计算以下三个竞争测度：

测度1（Layer-wise attention distance）：相邻层之间注意力矩阵的Frobenius距离均值：
$$d_{\mathrm{attn}} = \frac{1}{L-1}\sum_{l=1}^{L-1} \|A^{(l+1)} - A^{(l)}\|_F$$

测度2（Mean attention entropy）：所有头的注意力分布的Shannon entropy均值：
$$d_{\mathrm{ent}} = \frac{1}{H}\sum_{h=1}^{H} H(a^{(h)})$$

测度3（Residual stream norm change）：从x1到x3的residual stream向量范数变化：
$$d_{\mathrm{res}} = \|\mathbf{h}_{x_3} - \mathbf{h}_{x_1}\|_2$$

步骤2：对每个竞争测度，运行Kruskal-Wallis H检验区分六种scenario type，报告ε²。

步骤3：与holonomy deviation的H3 ε²对比（GPT-2: 0.638–0.685；Llama-7b: 0.420–0.422）。

步骤4：计算每个竞争测度和全局holonomy scalar的Spearman相关系数，评估信息重叠程度。

**判断标准和对论文声明的影响：**
- 竞争测度的ε² << 论文的ε² → holonomy框架有独特的判别价值，fiber bundle语言是在捕捉其他测度无法捕捉的信息
- 竞争测度的ε² ≈ 论文的ε² → holonomy框架没有额外判别价值，但数学解释更丰富——论文需要承认这一点，并把贡献重新定位为"为一个可被简单测度近似的信号提供几何解释框架"
- T0 anomaly在竞争测度上是否成立：如果竞争测度也显示T0最高，说明T0 anomaly是attention统计的一般现象，不是holonomy框架特有的发现

**在论文里的呈现：** 在Section 11（Related Work）里或Section 9里新增一段竞争比较，诚实报告结果，不管结果如何都提供分析。

---

## 第四部分：需要GPU的实验（标注defer，不影响当前提交）

这部分在论文里用统一措辞标注为"deferred to Paper V"或"deferred pending computational resources"。

---

### 实验E：Activation Patching（最高优先级defer）

这是整篇论文最重要的缺失实验，但也最需要GPU。做法是：把T2场景在x1之后的特定层的residual stream activation，替换进T1场景对应位置，观察holonomy deviation是否随之改变，以及模型输出是否随之改变。如果holonomy deviation跟着activation走（而不是跟着prompt文字走），就建立了causal status。

这个实验必须在Llama-7b上做，需要显著的GPU内存（TransformerLens + Llama-7b + gradient-free patching，至少需要A100 40GB或等效）。Defer到vast.ai，预算约$30–50美元（视实验规模）。

### 实验F：Llama-7b Ablation Sensitivity（中优先级defer）

H5目前只在GPT-2上完成（ρ ≥ 0.94）。Llama-7b的ablation需要在不同层数（32→25→15→7→3层）下重跑curvature计算，评估scenario ranking的稳健性。这在GPU上可以在几小时内完成。

### 实验G：T3/T4/T5剩余batch（继续现有pipeline）

你现有pipeline已跑90/225个对话（T3/T4/T5各缺少部分）。这些需要继续跑完，补充到Paper IV的结果里。需要250GB+磁盘实例，这是vast.ai的主要用途之一。

### 实验H：T0机制控制实验（defer，依赖实验A结果）

只有在实验A证明T0 anomaly是baseline-robust之后，才值得投入以下机制控制实验：
- 控制问题难度（简单vs复杂factual questions）
- 控制答案确定性（有唯一正确答案vs有争议的事实问题）
- 控制词汇复杂度（prompt长度、词频分布匹配）

这些需要生成新的对话数据，需要GPU和时间。

---

## 第五部分：执行顺序

按照当前资源条件（无GPU，有现有270个对话数据）：

**第一阶段（第1–2周，纯写作，不需要跑任何代码）：**

1. 全文术语替换：curvature → holonomy deviation（预计2–3小时，用find & replace + 人工核查）
2. 新增Remark 3.8（holonomy vs differential curvature的说明段落）
3. 新增Section 7开头的vector bundle声明段落
4. 修改Section 4开头（五层从本体模块到诊断坐标系）
5. 修改Section 9.1结论段落（H1 failure的理论重构）
6. Section 10新增10.2小节（全局一致性几何重构）
7. 修改Definition 8.3（T0 reframing）
8. 修改Abstract（T0和五层的描述）
9. 替换Table 6（补充H8.19/H8.20/H8.21的缺失状态）
10. Section 1结尾插入三层命题结构段落
11. H1拆分成H1a和H1b（Section 8.5）
12. 自引文献处理（所有Zhang 2025a/b引用加footnote说明）

**第二阶段（第3–4周，轻量实验，CPU可跑）：**

1. 实验A：Baseline Ensemble（优先级最高，先跑）
2. 实验C：PCA分解（计算量最小，和实验A并行）
3. 实验D：竞争基准（需要额外代码，约2–3天）
4. 实验B：Null Grouping Controls（最耗时，约3–5天，放最后）
5. 值矩阵正交性测量（附录B，GPT-2部分CPU可跑）

**第三阶段（第5–6周，整合实验结果）：**

1. 根据实验A结果决定T0 anomaly的声明强度
2. 根据实验C结果决定五层分解的声明强度
3. 根据实验B结果决定五层诊断坐标系的统计支撑措辞
4. 根据实验D结果决定holonomy框架独特贡献的声明
5. 重写Section 9（新增9.2 PCA、9.9 Baseline Robustness、更新9.1 H1 discussion）
6. 重写Section 10（整合新的Discussion）
7. 更新Abstract和Introduction

**之后（vast.ai可用后）：**
- 实验G：T3/T4/T5剩余batch
- 实验F：Llama-7b ablation
- 实验E：Activation patching（Paper V范畴）

---

## 附：修改后论文的核心主张（最终版本）

**论文标题建议修改为：**  
*Holonomy Geometry of Cognitive States in Transformer Internal Representations*

**Abstract的主张顺序建议调整为：**
1. 我们提出了一个基于attention transport路径依赖性的离散holonomy deviation测度
2. 在两个模型上，这个测度能以大效应量区分六种认知情景（核心经验发现）
3. 最反直觉的发现：factual retrieval产生比adversarial failure更高的全局几何复杂度（T0 anomaly）
4. 五层诊断投影提供了这个全局信号的解释性坐标系，但block-specific localization未得到支持
5. 结果建立了holonomy geometry作为transformer内部状态分析工具的可行性

这个顺序把最强的结果放前面，把需要降阶的结果放后面，是最有利于论文整体印象的叙事结构。
