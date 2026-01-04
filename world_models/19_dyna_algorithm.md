# Dyna 架构：Model-Based RL 的经典范式

> Dyna 是理解 Model-Based RL 的关键起点，它优雅地将 direct RL 和 planning 统一在一个框架中。

## 1. 核心思想

### 1.1 从一个问题出发

**苏格拉底式提问**：

> Q: 如果每次和真实环境交互都要花钱（比如机器人磨损、自动驾驶事故风险），你还会像 DQN 那样疯狂试错吗？
>
> A: 不会。我会尽量减少真实交互，多用"脑内模拟"。

**Dyna 的解答**：

```
真实交互：收集数据 + 学习模型
脑内模拟：用模型生成虚拟经验 + 用虚拟经验训练策略
```

### 1.2 Dyna 架构图

```
            ┌──────────────────────────────────────────────┐
            │                 Dyna 架构                     │
            └──────────────────────────────────────────────┘

            真实环境                      内部模型
               │                            │
               ▼                            ▼
         ┌─────────┐                  ┌─────────┐
         │ 真实交互 │                  │Planning │
         └────┬────┘                  └────┬────┘
              │                            │
              ▼                            ▼
         ┌─────────┐                  ┌─────────┐
         │ 经验    │                  │ 模拟经验 │
         │ (s,a,r,s')│                │ (s,a,r̂,ŝ')│
         └────┬────┘                  └────┬────┘
              │                            │
              └─────────────┬──────────────┘
                            │
                            ▼
                      ┌─────────┐
                      │ 价值/策略 │
                      │   更新   │
                      └─────────┘
```

### 1.3 核心公式

**环境模型**：
$$\hat{s}', \hat{r} = f_\theta(s, a)$$

**价值更新**（Q-Learning）：
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**Dyna 的关键**：上述更新既用于真实经验，也用于模型生成的虚拟经验。

---

## 2. Dyna-Q 算法

### 2.1 伪代码

```python
# Dyna-Q 算法
def dyna_q(env, n_planning_steps):
    Q = defaultdict(zeros)  # Q 表
    Model = {}               # 环境模型: (s,a) -> (s',r)

    for episode in episodes:
        s = env.reset()

        while not done:
            # 1. 选择动作 (epsilon-greedy)
            a = epsilon_greedy(Q, s)

            # 2. 真实交互
            s', r, done = env.step(a)

            # 3. Direct RL: 从真实经验更新 Q
            Q[s][a] += alpha * (r + gamma * max(Q[s']) - Q[s][a])

            # 4. Model Learning: 更新环境模型
            Model[(s, a)] = (s', r)

            # 5. Planning: 用模型生成虚拟经验
            for _ in range(n_planning_steps):
                s_sim, a_sim = random_choice(Model.keys())
                s'_sim, r_sim = Model[(s_sim, a_sim)]

                # 从虚拟经验更新 Q
                Q[s_sim][a_sim] += alpha * (
                    r_sim + gamma * max(Q[s'_sim]) - Q[s_sim][a_sim]
                )

            s = s'
```

### 2.2 关键参数

| 参数 | 含义 | 典型值 |
|:---|:---|:---|
| `n_planning_steps` | 每步真实交互后的 planning 次数 | 5-50 |
| `alpha` | 学习率 | 0.1 |
| `gamma` | 折扣因子 | 0.95 |
| `epsilon` | 探索率 | 0.1 |

---

## 3. 实验结果

### 3.1 GridWorld 环境

```
S.....      S = 起点
......      G = 终点
......      # = 墙壁
.###..      . = 空地
......
.....G
```

### 3.2 性能对比

| 方法 | 收敛所需环境步数 | 样本效率 |
|:---|:---|:---|
| Q-Learning | ~3400 | 1x (baseline) |
| Dyna-Q (n=5) | ~2400 | **1.4x** |
| Dyna-Q (n=10) | ~2400 | **1.4x** |
| Dyna-Q (n=50) | ~2400 | **1.4x** |
| Prioritized Sweeping | ~2300 | **1.5x** |

### 3.3 学习曲线分析

**关键发现**：
1. **Planning 加速收敛**：Dyna-Q 在前 50 个 episode 就达到 0.83+ 的平均奖励，而 Q-Learning 只有 0.62
2. **边际效益递减**：n=5 和 n=50 的差距不大（简单环境中模型很快就准确了）
3. **Prioritized Sweeping 更高效**：优先更新 TD error 大的状态

---

## 4. Prioritized Sweeping

### 4.1 核心思想

> 不是随机选择 (s, a) 进行 planning，而是优先处理那些 Q 值变化可能很大的状态。

**直觉**：如果 Q(s', a') 刚刚大幅更新，那么所有能到达 s' 的前驱状态 s 的 Q 值也可能需要更新。

### 4.2 算法

```python
def prioritized_sweeping(env, n_planning_steps, theta=0.0001):
    Q = defaultdict(zeros)
    Model = {}
    Predecessors = defaultdict(list)  # 前驱关系
    PQueue = PriorityQueue()          # 优先队列

    for episode in episodes:
        s = env.reset()

        while not done:
            a = epsilon_greedy(Q, s)
            s', r, done = env.step(a)

            # 更新模型和前驱关系
            Model[(s, a)] = (s', r)
            Predecessors[s'].append((s, a, r))

            # 计算优先级 = |TD error|
            priority = abs(r + gamma * max(Q[s']) - Q[s][a])

            if priority > theta:
                PQueue.push((s, a), priority)

            # Planning: 处理优先级最高的 (s, a)
            for _ in range(n_planning_steps):
                if PQueue.empty():
                    break

                s_plan, a_plan = PQueue.pop()
                s'_plan, r_plan = Model[(s_plan, a_plan)]

                Q[s_plan][a_plan] += alpha * (
                    r_plan + gamma * max(Q[s'_plan]) - Q[s_plan][a_plan]
                )

                # 反向传播：更新所有前驱状态的优先级
                for s_pred, a_pred, r_pred in Predecessors[s_plan]:
                    priority = abs(r_pred + gamma * max(Q[s_plan]) - Q[s_pred][a_pred])
                    if priority > theta:
                        PQueue.push((s_pred, a_pred), priority)

            s = s'
```

---

## 5. 从 Dyna 到现代世界模型

### 5.1 演进路线

```
Dyna (1990)
    │ 简单表格式模型
    ▼
World Models (2018)
    │ 神经网络 VAE + RNN
    ▼
Dreamer (2020)
    │ RSSM + Actor-Critic
    ▼
DreamerV3 (2023)
    │ 离散潜在空间 + 通用超参数
    ▼
```

### 5.2 核心差异

| 维度 | Dyna | World Models | Dreamer |
|:---|:---|:---|:---|
| **状态表示** | 表格式 | VAE 潜在空间 | RSSM |
| **模型类型** | 确定性 | MDN (多模态) | 先验+后验 |
| **Planning** | Q-Learning | CMA-ES | Actor-Critic |
| **规模** | 小状态空间 | 视觉任务 | Atari/机器人 |

### 5.3 共同核心

**无论多复杂，本质都是 Dyna 的思想**：

1. **学习环境模型**：从真实数据学习 $P(s'|s, a)$
2. **用模型 planning**：在想象中生成虚拟经验
3. **从虚拟经验学习**：减少对真实环境的依赖

---

## 6. 模型误差的影响

### 6.1 核心问题

> 如果模型是错的，在错误的"梦境"中训练会怎样？

**Dyna 的假设**：模型足够准确（表格式模型 + 简单环境可以保证）

**现实挑战**：神经网络模型一定有误差，误差会累积

### 6.2 误差来源

```
真实轨迹:   s₀ → s₁ → s₂ → s₃ → ...
                ↓     ↓     ↓
模型轨迹:   s₀ → ŝ₁ → ŝ₂ → ŝ₃ → ...
                ε₁    ε₂    ε₃

复合误差: ε₃ ≈ f(ε₁, ε₂, ε₃) → 可能指数放大
```

### 6.3 解决方案

| 方案 | 描述 | 代表方法 |
|:---|:---|:---|
| **短视野 Planning** | 只在模型中 rollout 几步 | MBPO (k=5) |
| **模型集成** | 多个模型投票，减少单一模型偏差 | PE-TS |
| **不确定性惩罚** | 在高不确定区域悲观估计 | MOReL |
| **Dyna-style 混合** | 同时用真实和虚拟经验 | Dyna, MBPO |

---

## 7. 实践洞察

### 7.1 何时使用 Model-Based RL？

**适合场景**：
- 真实交互昂贵（机器人、自动驾驶）
- 环境动态相对简单可预测
- 有足够计算资源进行 planning

**不适合场景**：
- 环境高度随机（如某些 Atari 游戏）
- 动态极其复杂（如真实世界物理）
- 对实时性要求极高

### 7.2 关键 Trade-off

```
样本效率 ←─────────────────────────────→ 计算效率

Model-Free:                         Model-Based:
- 样本效率低                         - 样本效率高
- 计算简单                           - 需要训练模型 + planning
- 不需要模型假设                      - 受限于模型准确性
```

### 7.3 代码实践建议

```python
# 1. 从简单环境开始
env = GridWorld()  # 先验证算法正确性

# 2. 逐步增加 planning steps
for n in [0, 5, 10, 50]:
    agent = DynaQ(n_planning_steps=n)
    # 观察样本效率变化

# 3. 监控模型误差
def evaluate_model(model, real_env):
    # 比较模型预测 vs 真实转移
    pass

# 4. 可视化 Q 值和策略
visualize_policy(agent)
```

---

## 8. 总结

### 8.1 Dyna 的核心贡献

1. **统一框架**：将 direct RL 和 planning 优雅结合
2. **样本效率**：用计算换真实交互
3. **概念基础**：启发了后续所有世界模型方法

### 8.2 关键公式速查

| 概念 | 公式 |
|:---|:---|
| Q-Learning | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max Q(s',a') - Q(s,a)]$ |
| 模型学习 | $\hat{s}', \hat{r} = f_\theta(s, a)$ |
| Planning | 从 Model 采样 $(s,a)$，用 Q-Learning 更新 |
| 优先级 | $P(s,a) = \|r + \gamma \max Q(s') - Q(s,a)\|$ |

### 8.3 延伸阅读

- **原论文**: Sutton, R. S. (1990). "Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming"
- **综述**: Moerland et al. (2023). "Model-based Reinforcement Learning: A Survey"
- **实验代码**: `experiments/5_dyna_q.py`

---

*最后更新: 2025-12-17*
