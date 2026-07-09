# 理论架构 v0.1:立场和乐理论(Standpoint Holonomy,暂定名)

> 状态:设计稿(brainstorming 阶段理论核心章节)。
> 纲领:Geometric Selfhood(几何自我性)· 第一梯级。
> 二次验证:已执行(2026-07-09)——13/13 数值检查通过,脚本 `verify_theory_v01.py`(scratchpad),
> 明细见 §11;本轮发现并修复的缺陷见 §10。
> 诚实声明:数值验证覆盖**可机检命题**(恒等式、不变性、标度律、核维数);
> 标注为 CONJ 的命题是开放猜想,通过验证 ≠ 理论无缺陷。

---

## 1. 记号与基本对象

**对话事件图**:$\mathcal{G} = (V, E)$,顶点 $V = \{x_0, \dots, x_T\}$ 为对话轮(turn),
边 $(i \to j)$,$i < j$(因果注意力,DAG)。

**状态**:$h_i^{(\ell)} \in \mathbb{R}^d$ — 事件 $i$ 末 token 在层 $\ell$ 的残差流向量
(末 token 选择由 R3 粗粒化引理辩护)。

**OV 回路**(修正后的传输载体,取代旧框架错误的 $V_hV_h^\top$):
头 $h$ 的值投影 $W_V^h \in \mathbb{R}^{d_v \times d}$、输出投影 $W_O^h \in \mathbb{R}^{d \times d_v}$,

$$W_{OV}^h := W_O^h W_V^h \in \mathbb{R}^{d\times d}, \qquad \operatorname{rank} W_{OV}^h \le d_v.$$

**传输复形(transport complex)**:$\mathcal{T} = \{U_{i\to j}^{(\ell)}\}$,

$$U_{i \to j}^{(\ell)} = \sum_{h \in \text{layer } \ell} \alpha_{ji}^{(h,\ell)}\, W_{OV}^{(h,\ell)},$$

其中 $\alpha_{ji}^{(h)}$ 为事件级注意力权重(末 token query 对事件 $i$ 的 token 聚合,见 R3)。
**路径**:$\sigma = \big((i_0 \to i_1, \ell_1), (i_1 \to i_2, \ell_2), \dots\big)$,
要求事件序单调($i_0 < i_1 < \cdots$)且**层单调**($\ell_1 \le \ell_2 \le \cdots$)——
内容先在低层搬运、后在高层搬运是模型中实际可实现的信息路径;层单调性是本框架
对"哪些复合是物理的"的显式承诺(修复记录 D-2)。

**纤维内流**:$\Phi_j^{(\ell)}(z) = z + U_{j\to j}^{(\ell)} z + \mathrm{MLP}^{(\ell)}(\mathrm{LN}(z))$
(事件节点内的层内演化,非线性;线性化 $D\Phi$ 于工作点取雅可比)。

---

## 2. 度量层:Fisher–Rao 拉回度量(所有范数的正则选择)

读出 $p(h) = \mathrm{softmax}(W_U\,\mathrm{LN}(h))$。单纯形上的 Fisher–Rao 度量拉回残差流:

$$g_h(u, v) := (J u)^\top \big(\operatorname{diag}(p) - p\,p^\top\big)(J v), \qquad
J := \frac{\partial\, (W_U \mathrm{LN}(h))}{\partial h}\Big|_{h}.$$

**性质(已数值验证,V5a–c)**:
1. 半正定(V5a:最小特征值 $-6.7\times10^{-17}$,机器零);
2. **零空间 = 功能无关方向**:LN 不变方向($\mathbf{1}_d$ 与 $h - \bar h$ 缩放方向)
   精确落入零空间(V5b:$10^{-9}$–$10^{-11}$ 量级)——度量自动忽略不改变行为的偏差;
3. **KL 二阶恒等式**:$\mathrm{KL}\big(p(h)\,\|\,p(h+\epsilon u)\big) = \tfrac{1}{2}\epsilon^2 g_h(u,u) + O(\epsilon^3)$
   (V5c:比率 0.9992→0.9998 随 $\epsilon \downarrow$)。

**约定**:$g$ 依赖工作点 $h$(修复记录 D-3)。逐对话量在该对话工作点取 $g_h$;
总体统计用平均度量 $\bar g = \mathbb{E}_{\mathcal{D}}[g_h]$。以下所有范数默认在 $g$ 下
(白化坐标 $\tilde h = g^{1/2} h$ 中写为普通 Frobenius/谱范数)。

---

## 3. 公理层

**A1(子空间分解)**:存在 $\{W_k\}_{k \in K}$,$W_k = \operatorname{span}\bigcup_{h \in H_k} \operatorname{range}(W_{OV}^h)$,
分离度 $\delta := \max_{k \neq l} \|P_k P_l\|_2 < 1$。$\delta$ 逐模型可测;所有定理误差项以 $\delta$ 表出。

**A2(参照联络,三公理)**:一致性保持系综 $\mathcal{D}_0$ 上,

$$\bar U_{i\to j} := \arg\min_U \mathbb{E}_{\mathcal{D}_0}\big\|h_j - U h_i\big\|_g^2
\;\;(\text{闭式解 } \bar U = \Sigma_{ji}\Sigma_{ii}^{-1},\ \text{分块岭回归估计,见 E1}).$$

公理:(i) 集中性——有限样本界存在(E1);(ii) **探针不变性**——满足同一判据的两系综给出
$\|\bar U - \bar U'\|_g \le \epsilon_{\mathrm{ref}}$(**可实验检验的公理**,进预测 P7);
(iii) 块内可逆——$\sigma_{\min}(\bar U|_{W_k}) \ge c > 0$。
曲率/和乐由此是**相对量**(相对于 $\mathcal{D}_0$ 的背景联络)——Cartan"模型空间+偏差"结构,
非内蕴几何;论文表述必须携带此限定(修复旧框架的内蕴性过度主张)。

**A3(读出)**:$p = \mathrm{softmax}(W_U \mathrm{LN}(h))$,即 §2 的度量来源;
读出在 $W_k$ 上的非退化度 $c_k := \sigma_{\min}\big(g^{1/2}|_{W_k}\big)$ 逐模型可测(供 B1b)。

---

## 4. 表示层

**R1(不变量对)**:相对和乐元 $F$(见 D1)的块不变量:

$$\lambda_k(F) := \big\|P_{k^\perp} F P_k\big\|_{F,g} \;(\text{泄漏}), \qquad
\tau_k(F) := \big\|P_k (F - I) P_k\big\|_{F,g} \;(\text{畸变}).$$

**命题 R1.1(gauge 不变性)**:gauge 群 $G = \prod_k O(W_k)$,作用 $F \mapsto gFg^{-1}$。
$\delta = 0$(正交分解)时 $\lambda_k, \tau_k$ **精确不变**。
*证明*:$g$ 块对角 ⟹ $P_k g = g P_k$,故 $P_{k^\perp} gFg^{-1} P_k = g\,(P_{k^\perp} F P_k)\,g^{-1}$,
Frobenius 范数在正交共轭下不变。∎(数值:V1,偏差 $4.4\times10^{-16}$。)
$\delta > 0$ 时不变性偏差为 $O(\delta \|F\|)$(数值:V1b,$\delta$=0.05/0.1/0.2 ↦ 偏差
9.1e-3/3.8e-2/8.3e-2,随 $\delta$ 单调)。**修复记录 D-1:旧表述曾主张无条件不变,已改条件式。**

**R2(head 粒度传输恒等式,取代已证伪的"exact"命题)**:
给定注意力模式,注意力块的事件间线性映射**精确**等于 $U_{i\to j} = \sum_h \alpha_{ji}^{(h)} W_{OV}^h$
(对 value 线性是恒等式,非近似;V2:相对误差 $1.5\times10^{-16}$)。
不再主张任何"标量×投影"块形式。跨块泄漏界:

$$\big\|P_l\, U_{i\to j}\, P_k\big\| \le \delta \sum_{h} \alpha^{(h)}_{ji} \big\|W_{OV}^h\big\|
\quad (l \neq k),$$

泄漏对 $\delta$ 的线性标度已验证(V6:log2 斜率 1.00/1.00/1.00)。

**R3(粗粒化引理)[CONJ-常数]**:事件级传输 = 直接项 + 经中间 token 的间接路径和;
若间接注意力质量 $\mu$(可测)有界,则 $\|U^{\mathrm{event}} - U^{\mathrm{direct}}\| \le C\mu$。
末 token 摘要的选择由此从约定升级为带误差保证的近似;$C$ 的显式形式待推导。

**R4(可辨识性)[CONJ-定理模式]**:头亲和矩阵 $a(h,h') = \|P_{\mathrm{range}(W_{OV}^h)} P_{\mathrm{range}(W_{OV}^{h'})}\|$
上谱聚类;若真分区簇内相似 $\ge a$、跨簇 $\le b$、谱隙充分,则以 $O(\delta)$ 精度恢复分区至 gauge
(Davis–Kahan 路线);$|K|$ 由特征隙给出。**结构来自权重,语义命名来自 held-out 行为对比**——
旧框架 γ 头分组的循环性在定义层消除。这是估计可行性的前置(见 E1 维度约束)。

---

## 5. 动力层

**D1(相对和乐)**:三元组 $\sigma = (i,j,k)$:

$$F_{ijk} := U_{j\to k}\, U_{i\to j}\, \bar U_{i\to k}^{-1}, \qquad
H(\sigma) := \big(\lambda_k(F), \tau_k(F)\big)_{k \in K}.$$

DAG 无回路 ⟹ "回路"经参照联络闭合(格点规范理论的固定背景场结构)。

**D2(Magnus 非对易塔)**:路径 $\sigma$ 的逐步偏差 $F_t = \exp(A_t)$,累积偏差

$$\log\big(F_m \cdots F_1\big) = \underbrace{\sum_t A_t}_{\Omega_1}
+ \underbrace{\tfrac{1}{2}\sum_{t > s} [A_t, A_s]}_{\Omega_2} + O(\|A\|^3).$$

- **动态非对易性 $\equiv \Omega_2$**:框架核心卖点首次获得显式公式与符号;
- 无量纲**非对易占比** $\rho_{\mathrm{nc}} := \|\Omega_2\| / \|\Omega_1\|$(跨模型可比,进 M3/I2);
- 三阶残差标度已验证(V3:log2 斜率 3.00/3.00/3.00);
  路径序差 $\|\log F_{AB} - \log F_{BA}\| = \|[A,B]\| \cdot (1 + O(\|A\|))$ 已验证(V3b:比率 1.000);
- **收敛域(修复记录 D-4)**:充分条件 $\sum_t \|A_t\| < \ln 2$;实测超域时改用极分解
  变体或仅报告 $F_\sigma$ 本身。域条件与 §2 度量归一互相咬合(白化把 $F$ 拉回恒等元邻域)。

**D3(解析 null)[CONJ-推导中]**:内容无关注意力($\alpha \perp h$)下
$\mathbb{E}[U] = \bar\alpha \sum_h W_{OV}^h$;矩阵 Bernstein 给 $\|U - \mathbb{E}U\|$ 尾界;
对 $\Omega_2$ 的 null 期望与集中界单独推导。显著性由此有解析基准,置换检验降为复核。

**D4(混合曲率)[CONJ-探索性,允许失败]**:$M_{i\to j} := [D\Phi_j,\, U_{i\to j}]$
(传输与纤维内加工的交换子)。认知锚点:记忆再巩固——每次提取即再加工,
一致性要求"搬运与加工可交换"。可能证明平凡;失败不伤主塔。

---

## 6. 全局层:胞腔层论

事件图上传输扭曲的胞腔层 $\mathcal{F}$:上边缘算子与能量

$$(\delta_{\mathcal{F}} \psi)_{i\to j} = U_{i\to j}\psi_i - \psi_j, \qquad
E(\psi) = \|\delta_{\mathcal{F}}\psi\|_g^2 = \langle \psi, L_{\mathcal{F}}\psi\rangle,\;
L_{\mathcal{F}} = \delta_{\mathcal{F}}^* \delta_{\mathcal{F}}.$$

**全局一致性不变量**:$\mu := \lambda_{\min}^{+}(L_{\mathcal{F}})$ 与全局截面空间维数 $\dim \ker L_{\mathcal{F}}$。

**已验证的结构事实(V4a–d;均在传输可逆假设下,修复记录 D-5)**:
- 树:$\dim\ker L = d$(根值自由传播,V4a);
- 一般环:$\dim\ker L = \dim\ker(\mathrm{Hol}(c) - I)$(V4b:双零验证);
- 平凡和乐环:$\dim\ker L = d$(V4c);
- 能量恒等式 $E(\psi) = \langle\psi, L\psi\rangle$(V4d:$2.9\times10^{-16}$)。

**命题 S1(局部—整体)[CONJ-核心新定理]**:连通图、传输可逆:全局截面存在
⟺ 每个环的和乐平凡。推论:**注意力的多路径性是全局立场不一致的拓扑必要条件**
——树状对话不可能有纯拓扑型不一致,多路径注意力才引入障碍。(加权/退化情形待推导。)

---

## 7. 桥接层(几何 ⇒ 行为)

**B1a(一致性证书,可证)**:两路径末态差 $\Delta h = (F_\sigma - I)\bar U h$;由 §2 KL 恒等式,

$$\mathrm{KL}\big(p^{A} \,\|\, p^{B}\big) = \tfrac{1}{2}\|\Delta h\|_{g}^2 + O(\|\Delta h\|^3)
\le \tfrac{1}{2}\Big(\sum_k \lambda_k^2 + \tau_k^2\Big)\,\|\bar U h\|_g^2 + O(\cdot).$$

**低和乐 ⟹ 行为一致的保证**(证书方向)——对齐场景真正需要的方向;
度量本身即行为相关性,无需独立 Lipschitz 假设(V5c 已验证核心恒等式)。

**B1b(检测方向,条件性)[CONJ]**:读出于 $W_k$ 非退化($c_k > 0$,A3 可测)时,
行为散度 $\ge c_k^2 \tau_k^2 / 2 - $ 高阶项。无条件下界不存在(零空间方向),不做虚假承诺。

---

## 8. 估计层

**E1(分块岭回归)**:全维 $\bar U \in \mathbb{R}^{d\times d}$($d^2 \sim 10^7$)不可估
(修复记录:R4 由此成为估计可行性前置);在 $\bigoplus W_k$ 内分块估计
($\sum_k r_k^2$ 参数),标准岭回归有限样本界 ⟹ 最小刺激量处方(反解 vast.ai 预算)。

**E2(逆扰动传播)**:$\|\hat F - F\| \lesssim \|U\| \, \|\bar U^{-1}\|^2 \|\hat{\bar U} - \bar U\|
\,/\,(1 - \|\bar U^{-1}\|\|\Delta\bar U\|)$;**条件数 $\kappa(\bar U)$ 强制进入一切报告**。

**E3(伪谱)**:OV 算子非正规,复合行为由伪谱控制(特征值无害 ≠ 复合无害);
报告伪谱半径,解释瞬态增长型一致性失效。

---

## 9. 可证伪预测清单(理论 → 实验的合同)

| # | 预测 | 检验 | 失败后果 |
|---|---|---|---|
| P1 | 压力场景 $H(\sigma)$ 显著高于 $\mathcal{D}_0$,多 seed 稳健 | M1,预注册判据 | 框架无经验内容 |
| P2 | $\lambda/\tau$ 按子空间归因到场景类型(估计/评测分集) | M2 | 子空间语义失效 |
| P3 | $\rho_{\mathrm{nc}}$ 显著超解析 null(D3) | M3 | 非对易卖点删除 |
| P4 | 定向扰动目标子空间传输改变下游立场行为;控制方向无效 | I1(因果) | 桥接失效 |
| P5 | $H$ 在 held-out 场景预测 persona drift,优于 probing/熵基线 | I2 | 实用价值存疑 |
| P6 | $\mu$(全局)与逐路径 $H$(局部)在多路径对话中可分离 | 新实验 | 层论层降级为诊断 |
| P7 | 参照联络探针不变性($\epsilon_{\mathrm{ref}}$ 小) | A2(ii) 直接检验 | 公理层重构 |
| P8 | 树状对话无纯拓扑不一致;多路径引入之(S1 推论) | 结构对照实验 | S1 降级 |

---

## 10. 二次验证:发现并修复的缺陷 + 开放问题

**本轮(2026-07-09)发现并修复:**
- **D-1** gauge 不变性曾被无条件主张 → 改为 $\delta=0$ 精确 + $O(\delta\|F\|)$ 偏差(V1b 实测标度);
- **D-2** 路径复合缺少层单调性簿记 → §1 传输复形定义补上"哪些复合是物理的";
- **D-3** Fisher 拉回度量的工作点依赖与退化性未声明 → §2 约定补全(逐对话工作点 + 总体平均度量);
- **D-4** BCH/Magnus 收敛域未定 → 显式充分条件 $\sum\|A_t\| < \ln 2$ + 超域降级路径;
- **D-5** 层论截面理论隐含传输可逆假设 → 显式化(V4 全部在该假设下检验);
- **D-6** 验证脚本 V6 初版实现有噪声缺陷(每 δ 重抽随机矩阵)→ 修正后标度精确为 1.00
  ——验证器本身也要被验证,此为实例。

**开放问题(CONJ 标注项,允许失败,各有降级路径):**
R3 常数、R4 完整定理、D3 的 $\Omega_2$ null 界、D4 非平凡性、S1 加权/退化情形、B1b 下界、
T4 吸引子定理(纲领冲刺位)。

**不主张**:本文档"无缺陷"。担保来自流程(§12),本轮 D-1…D-6 恰是流程产出。

---

## 11. 数值验证记录(13/13 通过)

脚本:scratchpad `verify_theory_v01.py`(numpy-only,`py -3` 运行,2026-07-09)。

| 检查 | 命题 | 结果 |
|---|---|---|
| V1 | R1.1 gauge 不变性($\delta=0$ 精确) | PASS,偏差 4.4e-16 |
| V1b | 不变性偏差随 $\delta$ 增长 | PASS,0 → 9.1e-3 → 3.8e-2 → 8.3e-2 |
| V2 | R2 OV 传输恒等式 | PASS,相对误差 1.5e-16 |
| V3 | D2 二阶 Magnus 残差 $O(s^3)$ | PASS,log2 斜率 3.00×3 |
| V3b | 路径序差 = $[A,B]$(主导阶) | PASS,比率 1.000 |
| V4a | 树层:$\dim\ker L = d$ | PASS,6 = 6 |
| V4b | 一般环:$\ker L \leftrightarrow \ker(\mathrm{Hol}-I)$ | PASS,0 = 0 |
| V4c | 平凡和乐环:$\dim\ker L = d$ | PASS,6 |
| V4d | 能量恒等式 | PASS,2.9e-16 |
| V5a | Fisher 拉回半正定 | PASS,min eig -6.7e-17 |
| V5b | LN 不变方向入零空间 | PASS,1e-9 – 1e-11 |
| V5c | KL 二阶恒等式 | PASS,比率 → 0.9998 |
| V6 | R2 泄漏界 δ-线性标度 | PASS,斜率 1.00×3 |

---

## 12. 门禁协议(每条定理进论文前)

1. **反例搜索**:数值脚本按命题前提随机生成实例,搜索违反(本文档 V 系列为首轮);
2. **交叉审**:至少一个非本家族模型独立复审推导;
3. **人脉终审**:CS 博士人脉对承重定理(R1.1、R2、S1、B1a)逐行核;
4. 三关全过才允许在论文中以定理形态出现;CONJ 项以猜想/经验观察形态出现。
