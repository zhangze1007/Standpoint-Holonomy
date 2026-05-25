# LCESA 论文修改与实验执行方案

> 版本：根据当前合并稿（Paper IV + Paper V）与最新实验结果整理  
> 目标：补齐诊断性、T0 异常解释、因果闭环、跨模型稳健性，并将论文主张强度收束到可被审稿人接受的范围。

---

## 0. 先给结论

当前论文已经具备：

- **完整且自洽的理论骨架**
- **可计算的几何对象**
- **在 GPT-2 与 Llama-2-7b 上都出现的强区分信号**
- **稳定的 ablation / robustness 结果**

但仍缺少：

1. **block-specificity 的确认**  
2. **T0 anomaly 的排除式验证**  
3. **curvature 的前瞻预测能力**  
4. **干预实验支撑的因果闭环**  
5. **跨模型、跨任务、跨模板稳健性**

因此，当前最稳妥的论文主张应从：

- “五个独立 standpoint block 的定位理论”

收缩为：

- **“一个全局曲率场 + 若干可分解的诊断投影”**

---

## 1. 论文当前最需要修改的地方

### 1.1 H1：block-specificity 需要降级，而不是硬撑

当前实验结果显示：

- H1 未通过
- 五个 block 之间相关性很高（r > 0.97）
- 这说明五层更像一个**单一全局 curvature signal 的不同投影**
- 而不是五个完全独立、彼此正交的 standpoints

#### 含义

- 不能再把 `nar / soc / mor / pos` 写成已经被实证证明的独立子空间
- 不能再把“某类 failure 必然集中到某个 block”写成强断言
- 不能把 H1 失败简单解释成“模型太小”或“head 太少”

#### 建议改法

把 H1 拆成两个层次：

- **H1a（弱版）**：目标 failure 会在相关 block 上产生更高的 curvature projection
- **H1b（强版）**：目标 failure 在目标 block 上具有显著最大值

如果 H1b 继续失败，但 H1a 成立，则说明：

- 模型确实存在可诊断方向
- 但不是“单块独占”
- 论文应该改成 **dominant curvature mode**，而不是 **strict block-specific localization**

---

### 1.2 T0 anomaly：不能再当作 negative control

最新结果里，T0（pure factual Q&A）在 Llama-7b 上的 curvature 甚至高于所有 failure 场景，而且在每层、每块都成立。

#### 含义

T0 现在不是“零复杂度对照”，而更像：

- **无 stance pressure 的高检索复杂度状态**
- 一个 **retrieval-complexity condition**

#### 建议改法

把 T0 从“baseline”位置移走，改成：

- **T0 = high retrieval complexity, low stance pressure**
- **T1 = consistency baseline**

这样写更合理，因为 T1 是明确的 acknowledged revision baseline，更适合作为“低曲率一致性参考”。

---

### 1.3 如果 T0 仍然高曲率：可能是新发现，不一定是理论错误

如果在控制了：

- 词汇
- 模板
- 问题难度
- 表面可分性
- 问题长度
- 实体数量
- 语域风格

之后，T0 依然显著高曲率，则可以提出新假设：

> **factual retrieval is a high-curvature regime**

这可能意味着：

- 无约束知识检索本身会激活更复杂的内部通路
- 它不只是“简单问答”
- 它可能比 adversarial failure 更能暴露模型内部的几何复杂性

但注意：  
必须先做严格控制，才能把“现象”升级成“发现”。

---

## 2. 针对 T0 anomaly 的严谨验证方案

下面是一套排除式实验设计，用来证明 T0 不是由词汇、模板、难度、表面可分性导致的假阳性。

---

### 2.1 实验组 A：词汇 / 模板匹配控制

#### 目的

控制表面因素后再比较 curvature。

#### 设计

对每个 T0 样本，找到一个在以下方面尽量匹配的样本：

- 字数
- 句法复杂度
- 关键词类别
- 实体数量
- 问题类型
- 回答长度

然后对比 curvature。

#### 判定标准

- 若 T0 仍高于匹配样本 → 支持非表面因素
- 若差异消失 → T0 anomaly 可能主要是 lexical / template confound

---

### 2.2 实验组 B：同语义不同模板

#### 目的

区分“事实检索本身”与“某种问答格式”的影响。

#### 设计

对同一 factual 内容，改写为多个模板：

- 直接问答模板
- 多轮追问模板
- 书面考试模板
- 任务说明模板
- role-play 模板

#### 判定标准

- 若多个模板下都维持高曲率趋势 → 更支持 retrieval-complexity 假设
- 若只在某一种模板下成立 → 更可能是模板效应

---

### 2.3 实验组 C：难度分层

#### 目的

判断 curvature 是否随 factual difficulty 单调变化。

#### 分组

- 低难度事实
- 中难度事实
- 高难度事实
- 模糊 / 多源事实

#### 判定标准

若 curvature 随难度上升而上升，则 T0 不再是单一类别，而是：

- 一个 **retrieval complexity spectrum**

---

### 2.4 实验组 D：表面可分性剔除

#### 目的

排除“模型只是学会了表面模式”的批评。

#### 建议统计模型

\[
C = \beta_0 + \beta_1 \text{Scenario} + \beta_2 \text{Length} + \beta_3 \text{LexicalOverlap} + \beta_4 \text{EntityCount} + u_{\text{prompt}} + \epsilon
\]

其中：

- \(C\)：curvature
- \(\text{Scenario}\)：场景类型
- \(u_{\text{prompt}}\)：随机效应

#### 结果解释

- \(\beta_1\) 仍显著：支持场景效应不是表面因素
- \(\beta_1\) 不显著：说明 T0 / 场景差异可能主要来自表面特征

#### 可进一步做 residualized curvature

\[
C^\* = C - \hat{C}_{\text{surface}}
\]

再检验 \(C^\*\) 的场景差异。

---

## 3. 关于 T1 baseline：如何解释“失败”与“反证”

### 3.1 T1 的作用

T1 不是简单的“纯 baseline”，而是：

- **acknowledged revision baseline**
- 测试“承认修正但保持 standpoint coherence”的情形

---

### 3.2 如果 T1 低曲率持续成立

这支持论文的基本理论：

- 修正不等于失稳
- 承认修正反而可能是低曲率路径
- T1 是功能性 selfhood 的更好对照组

---

### 3.3 如果 T1 也不低曲率

这不一定意味着理论错误，也可能是 baseline 设定错了。

#### 可能原因

- T1 本身包含“修正 / 反思 / 重述”，它不一定是最接近 flat transport 的条件
- 当前的 baseline 仍可能不是“最小 commitment-preserving dialogue”

#### 修正建议

把 baseline 改成：

- **minimal commitment-preserving dialogue**
- 或 **canonical flat reference conversations**

---

## 4. Curvature 是否能预测真实 failure：如何做成前瞻预测

目前 curvature 的能力主要是：

- **解释 / 区分**
- 还不是 **预测**

要变成预测，必须证明：

> 在 failure 发生前，curvature 已经升高

---

### 4.1 时间窗设计

把每次对话拆成前缀窗：

- 前 1 个事件
- 前 2 个事件
- 前 3 个事件
- 前 4 个事件

计算前缀特征：

\[
\phi_t = \left(\bar{C}_{1:t}, \Delta C_{1:t}, \max C_{1:t}, \text{layer-wise slopes}\right)
\]

---

### 4.2 定义真实 failure 标签

需要人工或半自动标注是否发生：

- hallucination
- self-contradiction
- role drift
- boundary breach
- refusal collapse
- ownership collapse
- repair failure

---

### 4.3 比较基线模型

至少比较三类预测器：

1. **Curvature-only predictor**
2. **Text-only predictor**
3. **Activation / probe predictor**

比较指标：

- AUROC
- AUPRC
- calibration curve
- 提前量（failure 发生前几步可预测）

---

### 4.4 你要证明的不是“能预测一次”，而是“能提前多少步”

如果 curvature 在 failure 前 1 步、2 步、3 步就逐步上升，则可说：

- curvature 是预警量
- 不是事后解释器

---

## 5. 因果闭环：现在最缺的一环

当前结果主要是相关性，还没有因果性。

---

### 5.1 必做实验 A：head ablation / layer ablation

#### 做法

去掉与某一 block 相关的 heads 或 layer，检查：

- curvature 是否下降
- 对应 failure 是否减轻
- 其他 block 是否补偿上升

#### 判定

- 只改 curvature，不改 failure → 偏表征相关
- curvature 和 failure 一起改 → 支持因果作用

---

### 5.2 必做实验 B：activation patching

把 T0 / T2 / T3 的中间 activation patch 到 T1 中：

- 看 curvature 是否被“拉回”
- 看输出是否发生相应变化

#### 作用

直接测试哪个内部状态在驱动几何差异。

---

### 5.3 必做实验 C：transport intervention

直接在 \(U_{ij}\) 或分块上施加扰动：

\[
U'_{ij} = U_{ij} + \lambda \Delta U_{ij}
\]

观察：

- \(F_{ijk}\) 是否变化
- failure 是否同步变化

#### 若同步变化

这将接近因果证据。

---

## 6. 跨模型稳健性：如何闭环

现在已经有 GPT-2 和 Llama-2-7b，但这还不够。

---

### 6.1 建议的模型组合逻辑

至少需要以下对照：

1. **小 base model**
2. **中等 instruction-tuned model**
3. **更大模型 / 不同家族模型**

对照维度：

- base vs instruction-tuned
- small vs medium
- same family vs different family
- same prompt family vs different prompt family

---

### 6.2 比较时必须归一化

不同模型不能直接比原始 curvature。

建议比较：

- z-score 后 curvature
- 每层归一化 curvature
- 每 block 相对排名
- effect size
- 相关性结构

---

### 6.3 跨模型稳健性真正要证明什么

不是“所有模型数值相同”，而是：

- 曲率区分模式稳定
- 排名稳定
- 诊断逻辑稳定
- 不是某个模型家族的偶然现象

---

## 7. 如果 H1 / T0 仍然失败，如何判断是理论错了还是新理论

---

### 7.1 H1 失败但 H3/H4/H5 仍强

这说明：

- 有真实的全局曲率信号
- 但 block-specific 分解过强

#### 结论

应收缩为：

- **global curvature + weak projection**
- 五层模型作为诊断坐标
- 而不是五个独立物理层

---

### 7.2 T0 anomaly 在控制后仍成立

这很可能是新发现。

可以写成：

> factual retrieval is inherently high-curvature

但必须在严格控制后才能成立。

---

### 7.3 Curvature 不能预测真实 failure

这说明 curvature 更适合作为：

- 解释性指标
- 诊断性指标

而不是：

- 预警器 / 控制器

---

### 7.4 跨模型只在某一家族成立

说明理论带有架构依赖性。

此时应收缩成：

- **decoder-only transformer family 的内部一致性框架**

不要过早写成普适 AGI 理论。

---

## 8. 建议的论文叙述方式

论文主张建议分三层：

### 8.1 理论命题

**Endogenous standpoint is a computable coherence primitive**

---

### 8.2 经验命题

**Curvature discriminates scenario types and is robust across two transformer families**

---

### 8.3 开放命题

**Block-specific localization, T0 anomaly mechanism, and causal control remain open**

---

## 9. 实验优先级建议

### 第一优先级
1. H1 复现 / 降级重写
2. T0 anomaly 控制实验
3. curvature 的前瞻预测

### 第二优先级
4. head ablation
5. activation patching
6. transport intervention

### 第三优先级
7. 跨模型扩展
8. 更多模型家族
9. 更复杂任务与更长上下文

---

## 10. 一页总览版

### 现在已经证明
- 曲率信号存在
- 曲率能区分场景
- 曲率在两个模型里都能观察到
- ablation 下排序稳定

### 还没证明
- block-specificity
- T0 不是表面因素
- curvature 能前瞻预测真实 failure
- curvature 有因果控制意义
- 理论是否跨模型、跨家族普适

### 最值得深挖
- T0 anomaly
- H1 失败的理论含义
- 前瞻预测
- 干预实验

---

## 11. 最后一句话

当前论文的最好定位不是“我已经证明 AGI self-awareness”，而是：

> **我提出并初步验证了一种可计算的功能性自我一致性几何框架，它能稳定区分内部状态差异，并有潜力成为失败诊断、前瞻预警和训练干预的基础方法。**

---

