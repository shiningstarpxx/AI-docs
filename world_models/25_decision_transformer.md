# Decision Transformer：把 RL 变成序列建模

> Decision Transformer 代表了一条与 World Models 完全不同的路线：不学习世界如何运转，而是学习"好轨迹长什么样"。

## 1. 核心思想

### 1.1 范式转变

**传统 RL (World Models/Dreamer)**：
```
学习 P(s'|s, a)  →  在想象中规划  →  决策
"理解世界运转规律"
```

**Decision Transformer**：
```
学习 P(a|R̂, s, 历史)  →  条件生成动作
"模仿成功轨迹"
```

### 1.2 关键洞察

**把 RL 问题转换为序列建模问题**：

```
轨迹 τ = (R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₜ, sₜ, aₜ)

其中 R̂ₜ = Return-to-Go = Σᵢ₌ₜ^T rᵢ (从 t 到结束的累积奖励)

目标：学习 P(aₜ | R̂ₜ, sₜ, 历史)
```

**核心问题**：不是预测"最优动作"，而是回答"如果我想要 X 分，应该怎么做？"

---

## 2. 架构设计

### 2.1 输入序列

```
┌────────────────────────────────────────────────────────────────┐
│  时间步 t-2           时间步 t-1           时间步 t            │
│  ─────────           ─────────           ─────────            │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│  │ R̂₋₂ │ │ s₋₂ │ │ a₋₂ │ │ R̂₋₁ │ │ s₋₁ │ │ a₋₁ │ │ R̂ₜ  │ │ sₜ  │ │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ │
│     │      │      │      │      │      │      │      │      │
│     ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      │
│  ┌───────────────────────────────────────────────────────────┐│
│  │              Linear Embedding + Timestep PE               ││
│  └───────────────────────────────────────────────────────────┘│
│                              │                                │
│                              ▼                                │
│  ┌───────────────────────────────────────────────────────────┐│
│  │          GPT (Causal Self-Attention)                      ││
│  └───────────────────────────────────────────────────────────┘│
│                              │                                │
│                              ▼                                │
│                        预测 aₜ (动作位置)                      │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

**1. 嵌入层**
```python
# 三种 token 类型的嵌入
embed_return = nn.Linear(1, embed_dim)        # R̂ -> embedding
embed_state = nn.Linear(state_dim, embed_dim) # s -> embedding
embed_action = nn.Linear(act_dim, embed_dim)  # a -> embedding

# 时间步位置编码
embed_timestep = nn.Embedding(max_len, embed_dim)
```

**2. GPT 主干**
```python
# 标准 Transformer Decoder
transformer = nn.TransformerDecoder(
    nn.TransformerDecoderLayer(
        d_model=embed_dim,
        nhead=8,
        dim_feedforward=4*embed_dim
    ),
    num_layers=6
)
```

**3. 动作预测头**
```python
# 只在动作位置预测
predict_action = nn.Linear(embed_dim, act_dim)
```

### 2.3 训练过程

```python
def train_step(trajectories):
    # 1. 构建序列: (R̂, s, a, R̂, s, a, ...)
    # 2. 前向传播
    # 3. 只在动作位置计算 loss
    loss = MSE(predicted_action, true_action)  # 纯监督！
    return loss
```

**关键**：没有 TD learning，没有 Bellman 方程，纯监督学习！

---

## 3. 推理过程

### 3.1 条件生成

```python
def generate_action(model, history, target_return):
    """
    给定历史和目标回报，生成动作
    """
    # 构建输入序列
    sequence = []
    for t in range(len(history)):
        sequence.extend([
            history[t].return_to_go,
            history[t].state,
            history[t].action
        ])

    # 添加当前时间步
    sequence.extend([
        target_return,  # 目标 R̂
        current_state
    ])

    # 自回归生成动作
    action = model.predict_action(sequence)
    return action
```

### 3.2 使用方式

```
推理时：
1. 设定目标回报 R̂₀ = 目标分数 (比如 Atari 的 1000 分)
2. 每一步：
   - 输入: (R̂ₜ, sₜ, 历史)
   - 输出: aₜ
   - 执行 aₜ，获得 rₜ
   - 更新: R̂ₜ₊₁ = R̂ₜ - rₜ
```

---

## 4. Credit Assignment：DT vs TD

### 4.1 TD Learning 的困境

```
TD Learning 类似 RNN 的信息传播：

t=1    t=2    t=3    ...    t=99   t=100 ✓
 │      │      │              │      │
 └──────┴──────┴──────────────┴──────┘
           奖励必须逐步反向传播

V(s₉₉) ← r + γV(s₁₀₀)
V(s₉₈) ← r + γV(s₉₉)
...
需要 99 次迭代！误差逐步累积
```

### 4.2 Decision Transformer 的优势

```
DT 类似 Transformer 的信息传播：

t=1    t=2    t=3    ...    t=99   t=100
 │      │      │              │      │
 └──────┴──────┴──────────────┴──────┘
           │ Self-Attention │
           └────────────────┘
        任意位置直接建立关联！

s₁ 直接 attend 到 R̂₁ (最终回报)
一次前向传播，全局信息
```

### 4.3 稀疏奖励场景

**任务**：100 步迷宫，只有终点 +1 奖励

**TD Learning**：
```
步骤 1-99: r = 0
步骤 100:  r = 1

V(s₉₉) ← 0 + γ·1 = γ         第 1 轮学到
V(s₉₈) ← 0 + γ² = γ²         第 2 轮学到
...
V(s₁)  ← γ⁹⁹                  第 99 轮才学到！

+ 每轮估计误差累积...
```

**Decision Transformer**：
```
训练数据: (R̂=1, s₁, a₁), (R̂=1, s₂, a₂), ...

Self-Attention 直接学到:
  "R̂=1 时在 s₁ 执行 a₁"
  "R̂=1 时在 s₅₀ 执行 a₅₀"

一次训练，全局 credit assignment！
```

---

## 5. 避开 RL 的"致命三角"

### 5.1 RL 的致命三角 (Deadly Triad)

```
┌─────────────────────────────────────────────────────────┐
│  1. Function Approximation (神经网络估计)               │
│  2. Bootstrapping (用估计值更新估计值)                  │
│  3. Off-policy Learning (学习非当前策略的数据)          │
│                                                         │
│  三者同时存在 → 训练不稳定、发散                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 DT 如何绕过？

| 问题 | TD Learning | Decision Transformer |
|:---|:---|:---|
| Bootstrapping | V(s) ← r + γ·V̂(s') | 直接用真实的 R̂ₜ |
| Value Function | 需要学 V(s), Q(s,a) | **不学任何价值函数** |
| Off-policy | 复杂的重要性采样 | **天然支持** |

**DT 的本质**：纯监督学习，不是 RL！

---

## 6. World Models vs Decision Transformer

### 6.1 核心对比

| 维度 | World Models/Dreamer | Decision Transformer |
|:---|:---|:---|
| **核心思想** | 学习世界如何运转 | 学习好轨迹长什么样 |
| **学习目标** | 状态转移 P(s'｜s,a) | 动作生成 P(a｜R̂,s) |
| **是否学世界模型** | 是 | 否 |
| **能否 Planning** | 是（想象中规划） | 否 |
| **能否超越数据** | 是（想象中探索） | 否（受限于数据） |
| **Credit Assignment** | Bellman backup | Self-Attention |
| **稀疏奖励** | 困难 | 自然处理 |
| **在线学习** | 强 | 弱 |

### 6.2 哲学差异

```
World Models:
  "我理解这个世界怎么运转"
  → "让我想象一下如果这样做会怎样"
  → "找到最好的策略"

Decision Transformer:
  "我见过很多成功和失败的例子"
  → "告诉我你想要多少分"
  → "我按照成功例子的模式行动"
```

### 6.3 各自优势场景

**World Models 更适合**：
- 需要泛化到新场景
- 需要在线持续学习
- 数据量有限但计算充足
- 需要"创造性"探索

**Decision Transformer 更适合**：
- 有大量离线数据
- 稀疏奖励任务
- 不需要超越数据分布
- 追求训练稳定性

---

## 7. DT 的局限性

### 7.1 无法超越数据

```
DT 只能生成"数据中见过的"行为

如果数据集最高分是 500，DT 很难产生 600 分的策略

vs World Models/Dreamer:
可以在想象中探索新策略，可能发现数据中没有的更好策略
```

### 7.2 Return Conditioning 的挑战

```
推理时需要设定目标 R̂：

设太高 → 生成不出来的动作（数据中没见过这么高分）
设太低 → 表现不佳（按低分例子行动）

需要知道"合理的目标分数"是多少
```

### 7.3 不学习世界模型

```
DT 不理解"世界如何运转"
只学习"好轨迹长什么样"

无法做：
- 反事实推理（"如果当时那样做会怎样？"）
- Model-based planning
- 因果推断
```

---

## 8. 变体与发展

### 8.1 Online Decision Transformer (ODT)

```
问题：DT 是纯离线方法
解决：加入在线学习能力

1. 离线预训练 DT
2. 在线交互，收集新数据
3. 继续训练
```

### 8.2 Trajectory Transformer

```
更激进：把整个轨迹都建模为序列

不只预测动作，还预测状态和奖励
τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...)

可以做 planning：beam search 找最优轨迹
```

### 8.3 Q-Transformer

```
结合 DT 和 Q-learning

用 Transformer 架构
但训练目标是 Q 值而非监督学习
```

---

## 9. 实践建议

### 9.1 何时用 DT？

**适合**：
- 有高质量离线数据
- 稀疏奖励任务
- 想避免 RL 训练的不稳定
- 数据集覆盖了足够多的"好"轨迹

**不适合**：
- 需要在线持续改进
- 数据质量差或分布偏
- 需要超越数据集性能
- 需要理解因果关系

### 9.2 关键超参数

| 参数 | 建议值 | 说明 |
|:---|:---|:---|
| Context Length | 20-100 | 历史长度 |
| Embed Dim | 128-512 | 嵌入维度 |
| Num Layers | 3-6 | Transformer 层数 |
| Target Return | 数据集最高分的 0.8-1.2 | 推理时的目标回报 |

### 9.3 实现要点

```python
# 1. 数据预处理：计算 Return-to-Go
def compute_rtg(rewards):
    rtg = []
    running_sum = 0
    for r in reversed(rewards):
        running_sum += r
        rtg.append(running_sum)
    return list(reversed(rtg))

# 2. 序列构建：交错 (R̂, s, a)
def build_sequence(trajectory):
    seq = []
    for t in range(len(trajectory)):
        seq.extend([
            trajectory.rtg[t],
            trajectory.states[t],
            trajectory.actions[t]
        ])
    return seq

# 3. 推理：逐步降低目标回报
target_return = initial_target
for t in range(max_steps):
    action = model.predict(history, target_return, current_state)
    next_state, reward = env.step(action)
    target_return -= reward  # 关键：更新目标回报
```

---

## 10. 总结

### 10.1 DT 的核心贡献

1. **范式转换**：把 RL 变成序列建模
2. **稳定训练**：避开 RL 的致命三角
3. **稀疏奖励**：Self-Attention 自然处理
4. **简洁架构**：标准 GPT，无需 RL trick

### 10.2 与 World Models 的关系

```
两条不同的路线，解决类似问题：

World Models: 理解世界 → 想象 → 决策
              需要学习 P(s'|s,a)
              可以泛化和探索

Decision Transformer: 模仿成功 → 条件生成
                      只需要轨迹数据
                      受限于数据分布
```

### 10.3 未来方向

- **在线 DT**：结合离线预训练 + 在线微调
- **World Model + DT**：用世界模型生成数据，用 DT 学习策略
- **多任务 DT**：一个模型处理多种任务
- **层次化 DT**：高层目标 → 低层动作

---

## 参考资料

- **原论文**: Chen et al. (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling"
- **代码**: https://github.com/kzl/decision-transformer
- **综述**: https://arxiv.org/abs/2212.14194 "A Survey on Transformers in RL"

---

*最后更新: 2025-12-17*
