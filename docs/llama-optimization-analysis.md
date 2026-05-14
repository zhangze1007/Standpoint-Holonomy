# LCESA Llama-2-7b Extraction Optimization Analysis

Date: 2026-05-14

---

## 一、优化 ≠ 降精度

重新看了 `extract.py` 的完整代码。当前7小时跑90个对话的瓶颈**不是FP16精度问题**，而是纯Python循环：

```python
# extract.py:200-211 — 每个对话执行 32×5×5 = 800次 Python级循环
for layer_idx, attn_weights in enumerate(outputs.attentions):  # 32次
    for i in range(n_events):       # 5次
        for j in range(n_events):   # 5次
            block = pat[:, q_s:q_e, k_s:k_e]
            attention[layer_idx, :, i, j] = block.mean(dim=(1, 2))
```

**向量化方案**：把800次小tensor操作合并成1次大tensor操作，精度完全不变（都是FP32的mean计算）。预估加速 **3-5x**。

同理，residual提取（line 179-181）也是5次逐个索引，可以一次gather完成。

**这些都不碰模型权重的精度。** FP16加载 → FP32计算 → FP32存储，整条链路不变。

---

## 二、现有负性对照分析

GPT-2结果已经有这些对照：

| 对照类型 | 状态 | 说明 |
|---------|------|------|
| **Permutation test** | ✅ 已有 | `gpt2_permutation.csv` — 随机打乱head分组1000次 |
| **Ablation study** | ✅ 已有 | `gpt2_ablation.csv` — 系统性移除layers |
| **T1 baseline** | ✅ 已有 | H2测试：T1 curvature ≈ 0 vs failure > 0 |
| **Chance level** | ✅ 已有 | H1：target ratio > 1/5 chance level |
| **Random activation** | ❌ 缺失 | 用高斯噪声替代真实activation |
| **Random gamma (full pipeline)** | ⚠️ 部分 | Permutation只测了attention，没测完整curvature pipeline |
| **Cross-scenario null** | ❌ 缺失 | T1的"failure layer"应该没有集中效应 |

**对于NeurIPS/ICML**，最关键的缺失是：
1. **Random activation control** — 验证curvature度量不是对任何数据都高
2. **Full pipeline permutation** — 不只测attention，测完整的curvature → hypothesis test链路

先跑完Llama再决定，这两个对照可以在Llama数据上一起做。

---

## 三、存储方案

| 方案 | 容量 | 成本 | 可行性 |
|------|------|------|--------|
| HuggingFace免费 | 100GB | $0 | ✅ 完全够（优化后总计~2.6GB） |
| Kaggle | 20GB/dataset | $0 | ✅ 够（可以分model存） |
| HuggingFace Pro | 500GB | $9/月 | 不需要 |

优化后的存储估算：
- 每个对话：attention(0.2MB) + residuals(2.5MB) = **2.7MB**
- 225个对话：**600MB**
- Value matrices（只存一次）：**2GB**
- **总计：2.6GB** — HuggingFace免费额度绰绰有余

当前代码的问题是value_matrices每个对话都存（重复225次），优化后只提取一次。

---

## 四、具体方案

### Phase 1：提取代码优化（零精度损失）

1. **向量化attention提取** — 800次循环 → 1次tensor op
2. **批量residual提取** — 5次索引 → 1次gather
3. **Value matrices去重** — 只提取一次，存为共享文件
4. **增量checkpoint** — 每batch保存，中断可恢复
5. **减少empty_cache频率** — 每batch而非每对话

预估：**7小时 → 1-1.5小时**，存储：**15GB+ → 2.6GB**

### Phase 2：Llama完整pipeline

1. 在vast.ai重新开L40S实例（~$0.50/hr）
2. 跑优化后的extraction（~1.5小时）
3. 跑完整pipeline（grouping → curvature → baselines → hypothesis tests）
4. 上传到HuggingFace

### Phase 3：负性对照补全

1. Random activation control
2. Full pipeline permutation test
3. Cross-model comparison框架（为后续多模型准备）

### Phase 4：完整pipeline（等Llama结果后决定）

这些是后续的扩展，不急：
- 多模型对比（GPT、Llama、Mistral、Gemma）
- 大规模stimuli（真实对话数据集自动生成几万个）
- Probing classifier（在activation上训练线性probe）
- 因果干预实验（patch activation，看修改曲率是否改变模型行为）
- 跨训练阶段分析（pre-training vs RLHF后的曲率结构变化）
- 自动化eval pipeline（结果可复现，代码开源）

---

## 五、NeurIPS/ICML标准检查

当前GPT-2结果的完整性：

| 标准 | 状态 |
|------|------|
| 6个假设测试 | ✅ H1-H6（H5 informatively failed） |
| Effect size + CI | ✅ Cohen's d, bootstrap CI |
| Permutation test | ✅ 1000次 |
| Ablation study | ✅ |
| Cross-model validation | ⏳ Llama待跑 |
| Random activation control | ❌ 待加 |
| Full pipeline permutation | ❌ 待加 |
| Code + data开源 | ⏳ 准备中 |
