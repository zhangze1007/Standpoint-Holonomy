# 设计文档:Geometric Selfhood 纲领 · Standpoint Holonomy 框架(v1.0)

> 日期:2026-07-09 · 状态:待用户审阅
> 前置文档:理论架构 `2026-07-09-theory-architecture-v0.1.md`(13/13 数值验证通过);
> 审计基线 `docs/audit/00-claims.md`、`01-theory-code.md`、`02-numerics.md`(部分)。
> 本文档是 brainstorming 阶段的产出,批准后移交 writing-plans 生成实施计划。

---

## 0. 命名定案

| 层 | 名称 | 完整表述 |
|---|---|---|
| 纲领 | **Geometric Selfhood(几何自我性)** | *Geometric Selfhood: selfhood as the invariants of standpoint transport* / 几何自我性:自我性即立场传输下的不变量 |
| 框架 | **Standpoint Holonomy(立场和乐)** | *Standpoint Holonomy — the gauge-invariant transport geometry of commitment in language models* / 立场和乐——语言模型中承诺的规范不变传输几何 |
| 核心量 | **Consistency Spectrum(一致性谱)** | $\mathrm{CS} = (\lambda, \tau, \rho_{\mathrm{nc}}, \mu)$:泄漏、畸变、非对易占比、全局障碍 |

LCESA 名称退役(理由存档:吸引子未证却入名、low-curvature 预设结论、缩写不可检索)。
仓库更名为 `standpoint-holonomy` 在 W6 论文定稿时执行(避免中途链接断裂)。

## 1. 目标与决策记录

- **目标**:顶刊级标准;可设计的最高刻度 = "领域必须回应"的结果包(不以"自我意识突破"为目标——合法目标是功能阶梯第一梯级的测量,见理论文档 §纲领边界)。
- **战略**:方案丙(旗舰实验倒推)以双线执行(理论线 ∥ 实证线)。
- **不跟随决策**:保留自有框架(attention-as-parallel-transport);与 Anthropic J-space 论文(2026-07-06)的关系定位为互补——他们刻画工作空间的共时架构且明确未涉对话路径依赖,我们刻画历时一致性;论文需单列一节正面对比。
- **资源**:solo + AI;vast.ai GPU(数百美元档);数学把关 = 反例搜索脚本 + 多模型交叉审 + CS 博士人脉终审(三关门禁)。
- **历史结果处置**:GPT-2 全部历史数字作废(审计:层坍缩 + V 全零 + 判据漂移),挂 retraction note,不复现。

## 2. 旗舰结果包(第 1 节,已确认)

**理论核心**:见理论架构 v0.1(传输复形 / Fisher–Rao 拉回度量 / 公理 A1–A3 /
表示 R1–R4 / 动力 D1–D4(Magnus 塔)/ 全局层论 / 桥接 B1 / 估计 E1–E3)。
关键修正已锁定:传输算子 = OV 回路 $W_O W_V$(旧 $V_hV_h^\top$ 为对象错误)。

**测量主张 M1–M3**:LLaMA-7B 全精度上,(M1) 压力场景 CS 显著升高且多 seed 稳健(预注册判据);
(M2) $\lambda/\tau$ 按子空间归因到场景类型(估计/评测严格分离);(M3) CS 随压力轮数的演化符合理论。

**干预主张 I1–I2**:(I1) 沿 $W_k$ 的定向传输扰动按预测改变下游立场行为,控制方向无效;
(I2) CS 在 held-out 场景预测 persona drift,优于 probing/熵基线。

**对照 C**:随机初始化模型、shuffled 子空间、乱序刺激、≥2 模型家族。

**可证伪合同 P1–P8**:见理论架构 v0.1 §9(每条预测→检验→失败后果)。

## 3. 组件清单与废弃清单(第 2 节,已确认)

### 新建/重写(C1–C14)

| # | 组件 | 从何推出 | 对旧资产的处置 |
|---|---|---|---|
| C1 | 刺激语料 v2:每场景 ≥30 条真不同措辞 + 参照系综 $\mathcal{D}_0$ + 多路径/树状对照结构 | P1, P6, P8 | 废弃 templates.py 六模板克隆 |
| C2 | 提取引擎 v2:OV 回路(GPT-2 `c_attn` 拆分 / LLaMA `v_proj`+`o_proj`,带形状断言)、逐层索引注意力、事件 token 对齐验证、token 级注意力(供 μ) | 一切 M/I | 重写:新旧 extract.py 双双退役;审计两 BLOCKER 转为验收判据 |
| C3 | 参照联络估计器:$\mathcal{D}_0$ 分块岭回归,闭式解,κ 强制报告 | A2, P7 | 废弃 compute_baseline_transport |
| C4 | 子空间辨识:OV 值域亲和谱聚类,特征隙定 $\|K\|$,held-out 行为命名 | P2, R4, E1 前置 | 废弃 head_grouping.py γ 差分法及产物 |
| C5 | 不变量引擎:$F_{ijk}$、$(\lambda,\tau)$(Fisher 度量)、$\Omega_1/\Omega_2/\rho_{\mathrm{nc}}$、层 Laplacian $\mu$ | P1–P3, P6 | 废弃 compute.py(Tikhonov→分块解+κ 门;np.eye fallback 结构性消失) |
| C6 | Fisher 度量模块:反传 $J$、拉回范数、工作点约定 | 全部范数 | 新建 |
| C7 | 统计引擎 v2:预注册判据(α=0.01,检验族 Holm)、效应量+CI、多 seed、解析 null + 置换复核 | P1–P3, P5 | 废弃 hypothesis_tests.py 全部检验与三套编号 |
| C8 | 干预台架:$W_k$ 方向生成时 patching、控制方向、行为评分、persona-drift 预测 | P4, P5 | 新建(最大不确定项) |
| C9 | 多路径对话 μ 测量协议 | P6, P8 | 新建,依赖 C1+C5 |
| C10 | 参照系综稳健性:两独立 $\mathcal{D}_0$,测 $\epsilon_{\mathrm{ref}}$ | P7 | 新建 |
| C11 | 预注册文档 + 冻结配置(主跑前提交) | 反判据漂移 | 新建 |
| C12 | 信号检查点:20% 刺激 LLaMA-7B 试点,go/no-go 门 | 方案丙对冲 | 新建 |
| C13 | 验证套件 CI:verify_theory 扩展 + 管线已知解回归 | 门禁第 1 关 | 新建 |
| C14 | Null 电池:随机初始化 / shuffled 子空间 / 乱序刺激 | 对照组 C | 新建(第 2 节自审补入) |

### 纯废弃(不修不复现)

gpt2_theory_analysis.md(retraction note)· patch_extract.py · LCESA-project.zip ·
已 track 的 __pycache__ · CKA/熵/置换三基线(对象错误,C7 重建)·
论文 06 节 F 编号体系(修或删)· 04 节五层模型降为动机($\|K\|$ 由 C4 发现)。

### 波次(依赖拓扑序)

W0(C11+C13)→ W1(C2∥C1∥C6)→ W2(C4+C3)→ W3(C5+C7)→ W4(C12 试点→go/no-go)
→ W5(主跑 M1–M3 × 2 模型家族[LLaMA-7B + 非 LLaMA 系开放权重] + C8 + C9/C10 + C14)
→ W6(论文重写 + 仓库更名)。

**样本量/预算粗推**(E1):分块参数 $\sum r_k^2$、目标 10% 相对误差 ⟹ 每场景 $n_{\mathrm{eff}} \gtrsim 10^3$
事件对(试点校准,预计扩至 50–100 条/场景);GPU 主项为 C8(30–60 A100 时),提取 <10 时;
C12 试点 <$20 先行。

## 4. 风险与降级矩阵 + 时间线(第 3 节)

| # | 风险 | 触发信号 | 降级路径 |
|---|---|---|---|
| R1 | 试点无信号(最大科学风险) | C12 效应方向错/不可测 | 升模型规模 → 强化刺激 → 诚实负结果论文(方法学贡献,纲领存活) |
| R2 | C4 谱隙不显著 | 特征隙 < 预注册阈值 | 报告全谱 + 预注册固定 $\|K\|$;P2 降为探索性 |
| R3 | Magnus 超收敛域 | $\sum\|A_t\| > \ln 2$ | Fisher 白化先行;仍超则极分解变体或直接报告 $F_\sigma$ |
| R4 | 干预无效/不特异 | I1 目标 vs 控制无差 | 撤因果主张,退守测量 + B1a 证书;干预留后续 |
| R5 | CONJ 定理失败 | 三关门禁不过 | 理论 v0.1 内置降级(猜想/经验观察形态) |
| R6 | solo + quota + 预算 | 已多次发生 | 波次制 + 每波 go/no-go + 增量落盘 + 每波预算上限 |
| R7 | 撞车 | 文献监测 | W1 后即挂 arXiv v1(theory+protocol,诚实标注无结果)占时间戳 |
| R8 | "数学过度包装"指控 | — | CS 可无丛语言计算;几何进附录;每个形式化必须出现在计算中 |

**时间线**:W0 1–2 周 → W1 3–4 周 → W2 2 周 → W3 2–3 周 → W4 1 周(≈2.5 个月处第一止损闸门)
→ W5 4–6 周 → W6 4–6 周。合计约 **5–7 个月到可投稿**;理论线全程并行不阻塞。

## 5. 验证协议(全程有效)

1. 定理三关门禁:反例搜索脚本 → 非本家族模型交叉审 → CS 博士人脉终审;
2. 判据预注册,冻结于 C11,主跑后不得更改(判据漂移是旧项目死因之一);
3. 每个管线组件先过 C13 的合成已知解回归再触真实数据;
4. 一切结果携带 commit hash + config hash + seed,产物上传 HF 保全证据链;
5. 论文主张边界:功能阶梯第一梯级;"验证自我意识"不是本纲领任何阶段的合法主张。

## 6. 范围外(明确不做)

- 不复现 GPT-2 历史数字;不修复被废弃组件;
- 不采用 J-lens 作为地基(仅作对比与引用);
- 不在本期做 T4 吸引子定理之外的训练动力学;
- 不做人工标注众包(spec §8 遗留)——persona-drift 评分用模型评分 + 小样人工抽查替代。

## 7. 移交

用户批准本文档后 → 调用 superpowers:writing-plans 生成 W0–W1 的实施计划
(每任务含验收判据与测试),后续波次按闸门逐波出计划。
