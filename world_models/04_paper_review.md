# World Models 论文精读

> 对照原文验证理解，提取关键洞察

**论文**: World Models (Ha & Schmidhuber, 2018)

**arXiv**: https://arxiv.org/abs/1803.10122

**本地PDF**: [papers/world_models_2018_ha_schmidhuber.pdf](papers/world_models_2018_ha_schmidhuber.pdf)

---

## 1. 论文概述

### 1.1 核心贡献

> "We explore building generative neural network models of popular reinforcement learning environments. Our world model can be trained quickly in an unsupervised manner to learn a compressed spatial and temporal representation of the environment. We show that agents can learn a policy entirely inside of a hallucinated dream environment."

**核心观点**:
- 学习环境的生成模型（世界模型）
- 在"梦境"中训练策略，减少真实交互

### 1.2 灵感来源

论文引用了认知科学的观点：

> "Our brains learn an abstract model of both spatial and temporal aspects of the environment and use this model to make predictions."

人类大脑也在做类似的事：
- 建立世界的内部模型
- 在脑内模拟和预测
- 基于模拟做决策

---

## 2. 架构对照

### 2.1 V (Vision) - VAE

| 项目 | 论文设置 | 我们的理解 |
|:---|:---|:---|
| 输入 | 64x64 RGB 图像 | ✓ 高维图像 |
| 输出 | z (32维) | ✓ 低维潜在向量 |
| 模型 | VAE | ✓ 变分自编码器 |
| 损失 | 重建 + KL | ✓ 两项平衡 |

**论文细节**:
- Encoder: 4层卷积
- Decoder: 4层反卷积
- 潜在空间: 32维高斯

### 2.2 M (Memory) - MDN-RNN

| 项目 | 论文设置 | 我们的理解 |
|:---|:---|:---|
| 模型 | LSTM | ✓ 记忆历史 |
| 隐藏层 | 256 units | - |
| 输出 | MDN (K=5) | ✓ 混合高斯 |
| 预测目标 | P(z_{t+1}) | ✓ 下一状态分布 |

**论文细节**:
- 5 个高斯混合分量
- 同时预测"结束"概率 P(done)
- 损失: 负对数似然

### 2.3 C (Controller) - 线性策略

| 项目 | 论文设置 | 我们的理解 |
|:---|:---|:---|
| 模型 | 单层线性 | ✓ 极简策略 |
| 参数量 | 867 | ✓ 防止过拟合 |
| 输入 | [z_t, h_t] | ✓ 感知 + 记忆 |
| 训练 | CMA-ES | ✓ 进化算法 |

**为什么用 CMA-ES?**
- 无需梯度，直接优化最终得分
- 对随机性鲁棒（世界模型是随机的）
- 参数少时特别高效
- 可以并行评估多个候选解

---

## 3. 训练流程对照

### 3.1 论文的训练步骤

```
步骤 1: 收集随机数据
        随机策略 -> 10,000 rollouts -> (图像, 动作) 序列

步骤 2: 训练 V (VAE)
        学会压缩图像到 z

步骤 3: 训练 M (MDN-RNN)
        在 z 序列上学习预测

步骤 4: 训练 C (Controller)
        在"梦境"中用 CMA-ES 进化
```

### 3.2 关键点

**分阶段训练**:
- V 和 M 先单独训练好
- C 在固定的 V+M 上训练
- 不是端到端联合训练

**这是简化，也是局限** (后来 Dreamer 改进了这点)

---

## 4. 实验结果

### 4.1 Car Racing

```
任务描述:
  - 赛车游戏，从上方视角
  - 连续控制: 转向, 油门, 刹车
  - 目标: 沿着赛道行驶

结果:
  +------------------+--------+
  | 方法             | 得分   |
  +------------------+--------+
  | 纯梦境训练       | ~900   |
  | 梦境 + 真实微调  | 906    |
  | 人类水平         | ~900   |
  +------------------+--------+

结论: 梦境训练可以达到接近人类的水平
```

### 4.2 VizDoom (Take Cover)

```
任务描述:
  - 第一人称射击
  - 躲避飞来的火球
  - 离散动作: 左, 右, 不动

结果:
  - 纯梦境训练的 agent 表现良好
  - 但发现了"作弊"问题 (见下文)
```

---

## 5. 关键发现

### 5.1 梦境训练确实可行

> "An agent trained inside of the hallucination can transfer policies back to the actual environment."

**验证了我们的理解**:
- 可以在想象中训练策略
- 策略能迁移到真实环境

### 5.2 简单 Controller 的必要性

> "We deliberately use a small controller with minimal parameters to prevent it from memorizing the game."

**论文明确指出**:
- 简单控制器是为了防止过拟合
- 防止 Controller 记住世界模型的缺陷

### 5.3 "作弊"问题

论文发现了一个有趣现象:

```
在 VizDoom 中:
  - Agent 学会了在某些区域"自杀"
  - 因为世界模型在这些区域预测不准
  - Agent 利用了模型的缺陷来"逃避"火球

这验证了我们讨论的:
  复杂 Controller -> 可能学会利用世界模型的漏洞
  简单 Controller -> 只能学到鲁棒的大方向策略
```

### 5.4 温度参数 τ

论文引入了"温度"来控制梦境的随机性:

```
采样时: z ~ N(μ, τ*σ)

τ < 1: 梦境更"确定"
       - 更容易训练
       - 但可能过于"乐观"

τ = 1: 标准设置

τ > 1: 梦境更"随机"
       - 更接近真实世界的不确定性
       - 训练更难，但更鲁棒
```

**有趣发现**:

> "Training in a more difficult environment with higher τ could make the agent more robust."

在 τ > 1 的"噩梦"中训练的 agent，在真实环境中表现更鲁棒！

---

## 6. MDN 细节

### 6.1 为什么 K=5？

```
K 的选择是经验性的:

K 太小: 无法表达复杂的多模态分布
K 太大: 参数多，可能过拟合

K=5 在论文任务上工作良好
```

### 6.2 K > 实际模态数会怎样？

```
如果真实分布只有 2 个峰，但 K=5:

  π_1 = 0.25 ─┐
  π_2 = 0.25 ─┴─> 合并表示峰 A

  π_3 = 0.25 ─┐
  π_4 = 0.25 ─┴─> 合并表示峰 B

  π_5 ≈ 0     ─> 退化 (权重趋近 0)

多余的分量会自动"合并"或"退化"
不会导致偏差，只是有些冗余
```

---

## 7. CMA-ES vs 梯度下降

### 7.1 为什么不用梯度下降？

```
梯度下降的问题:

1. 需要通过 M (世界模型) 反向传播
   - M 是随机的 (MDN 采样)
   - 梯度方差大

2. 长轨迹的梯度问题
   - 可能爆炸/消失
   - 100+ 步的梯度链很难优化

3. Controller 参数少 (867个)
   - 梯度信号弱
   - 可能不稳定
```

### 7.2 CMA-ES 的优势

```
1. 无需梯度
   - 直接评估"最终得分"
   - 不管中间过程

2. 对噪声鲁棒
   - 多次采样取平均
   - 自然处理随机性

3. 并行高效
   - 同时评估 N 个候选解
   - 适合分布式训练

4. 参数少时特别适合
   - 867 参数正好在 CMA-ES 的舒适区
```

### 7.3 类比

```
梯度下降: 盲人摸着斜坡走
          每一步都要感知坡度
          需要坡度信息可靠

CMA-ES:   往几个方向扔石头
          看哪个滚得最远
          然后往那边走
          不需要知道具体坡度
```

---

## 8. 论文的局限性

### 8.1 分阶段训练

```
V, M, C 分开训练:
  - V 和 M 是固定的
  - C 无法改进 V 和 M
  - 如果 V 或 M 有缺陷，C 只能"适应"

后来 Dreamer 改进:
  - 端到端联合训练
  - 所有组件同时优化
```

### 8.2 简单控制器的限制

```
线性 Controller:
  - 表达能力有限
  - 复杂任务可能不够

但这是有意为之:
  - 防止过拟合
  - Trade-off: 简单 vs 表达力
```

### 8.3 世界模型误差

```
梦境训练的根本问题:
  - 世界模型不完美
  - 误差会累积
  - Agent 可能学会"作弊"

解决方向:
  - 更好的世界模型 (Dreamer 的 RSSM)
  - 在线更新世界模型
  - 不确定性估计
```

---

## 9. 与后续工作的关系

```
World Models (2018)
      |
      v
PlaNet (2019)
  - 用 MPC 规划替代学习 Controller
  - RSSM: 确定性 + 随机性双路径
      |
      v
Dreamer (2020)
  - Actor-Critic 在想象中训练
  - 端到端联合优化
  - 更好的性能
      |
      v
DreamerV2 (2021)
  - 离散潜在变量
  - Atari 上超越人类
      |
      v
DreamerV3 (2023)
  - 通用架构
  - 无需任务特定调参
```

---

## 10. 核心 Takeaways

### 10.1 论文的核心贡献

```
1. 证明了"梦境训练"的可行性
2. 提出了 V-M-C 架构框架
3. 发现了简单 Controller 的重要性
4. 揭示了温度参数对鲁棒性的影响
5. 开创了 model-based RL 的新范式
```

### 10.2 验证我们的理解

```
✓ VAE 压缩高维输入
✓ MDN-RNN 学习环境规则
✓ 混合高斯表达多模态未来
✓ 简单 Controller 防止过拟合
✓ 梦境训练提高样本效率
✓ 模型误差是核心挑战
```

### 10.3 留下的问题

```
? 如何端到端训练？ -> Dreamer
? 如何处理更复杂的任务？ -> DreamerV2/V3
? 如何量化模型不确定性？ -> 模型集成
? 如何在线更新世界模型？ -> 持续学习
```

---

## 11. 下一步

```
当前 [完成]
  +-- 01: World Models 概念
  +-- 02: VAE 数学原理
  +-- 03: RNN/MDN 数学原理
  +-- 04: 论文精读

下一步
  +-- A: 代码实现 - 2_simple_world_model.py
  +-- D: 局限与发展 - 为什么需要 Dreamer?
```

---

*文档生成时间: 2024-12-09*

*学习方法: 苏格拉底式对话 + 论文对照*
