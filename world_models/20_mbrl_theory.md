# Model-Based Reinforcement Learning：理论深度解析

> 深入理解 Model-Based RL 的理论基础、核心权衡与前沿发展。

## 1. MBRL 基本框架

### 1.1 核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model-Based RL 框架                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  环境模型     │    │   策略       │    │  价值函数    │       │
│  │ f̂(s,a)→s'    │    │  π(a|s)     │    │  V(s), Q(s,a)│       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Planning / Policy Optimization              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│                       与环境交互                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 形式化定义

**环境模型**：
$$\hat{s}_{t+1}, \hat{r}_t = f_\theta(s_t, a_t)$$

**确定性模型** vs **随机模型**：
```
确定性: s' = f(s, a)
随机性: s' ~ p(s'|s, a)
```

**完整的 MBRL 循环**：
```python
def mbrl_loop(env, model, policy):
    for iteration in range(max_iterations):
        # 1. 数据收集
        data = collect_data(env, policy)

        # 2. 模型学习
        model.train(data)

        # 3. 策略/价值优化 (使用模型)
        policy = improve_policy(model, policy)

        # 4. （可选）真实环境评估
        evaluate(env, policy)
```

---

## 2. Model-Free vs Model-Based 权衡

### 2.1 核心对比

| 维度 | Model-Free | Model-Based |
|:---|:---|:---|
| **样本效率** | 低（需要大量交互） | 高（模型可复用） |
| **计算开销** | 低（直接学策略） | 高（模型训练 + Planning） |
| **渐近性能** | 高（不受模型偏差影响） | 受限于模型准确性 |
| **泛化能力** | 有限（策略特定） | 更好（模型可迁移） |
| **实现复杂度** | 简单 | 复杂 |

### 2.2 样本效率分析

**定理（非正式）**：在模型准确的条件下，MBRL 的样本复杂度可以比 Model-Free 低若干数量级。

**直觉解释**：
```
Model-Free (Q-Learning):
  每个 (s, a, r, s') 只用一次

Model-Based (Dyna):
  真实 (s, a, r, s') → 训练模型
  模型 → 生成无限虚拟经验
  虚拟经验 → 训练策略

  n = 10 planning steps 意味着：
  1 次真实交互 ≈ 11 次学习更新
```

### 2.3 实验数据（CartPole 对比）

| 方法 | 收敛步数 | 样本效率 |
|:---|:---|:---|
| DQN (Model-Free) | ~20000 | 1x (baseline) |
| Simple World Model | ~4000 | **5x** |
| Mini Dreamer | ~3700 | **5.3x** |

---

## 3. 模型误差：MBRL 的核心挑战

### 3.1 误差类型

**1. 统计误差 (Statistical Error)**
```
原因：有限数据导致的估计偏差
解决：更多数据、正则化、数据增强
```

**2. 模型偏差 (Model Bias)**
```
原因：模型类（如神经网络）的表达能力限制
解决：更强大的模型架构、混合模型
```

**3. 分布偏移 (Distribution Shift)**
```
原因：策略改进后访问新状态，模型未见过
解决：持续数据收集、不确定性估计
```

### 3.2 复合误差 (Compounding Errors)

**问题**：多步预测时误差累积

```
真实轨迹:   s₀ → s₁ → s₂ → s₃ → ...
                 ↓     ↓     ↓
模型轨迹:   s₀ → ŝ₁ → ŝ₂ → ŝ₃ → ...
                 ε₁   ε₁+ε₂  ε₁+ε₂+ε₃

单步误差 ε 可能在 H 步后放大为 O(εH) 甚至 O(ε·γᴴ)
```

**数学分析**：

假设单步模型误差为 ε：
$$\|f_\theta(s,a) - f^*(s,a)\| \leq \epsilon$$

H 步后的预测误差上界：
$$\|\hat{s}_H - s_H\| \leq \sum_{t=0}^{H-1} L^{H-1-t} \cdot \epsilon = \frac{L^H - 1}{L - 1} \cdot \epsilon$$

其中 L 是环境的 Lipschitz 常数。

### 3.3 误差对策略的影响

**定理 (Policy Performance Bound, MBPO)**：

设 $\pi$ 是在模型 $\hat{P}$ 下的最优策略，则在真实环境 P 下的性能损失：

$$\eta[P, \pi] \geq \eta[\hat{P}, \pi] - \frac{2\gamma r_{max}}{(1-\gamma)^2} \cdot \epsilon_m$$

其中 $\epsilon_m = \max_{s,a} D_{TV}(\hat{P}(\cdot|s,a), P(\cdot|s,a))$

**实际意义**：
- 模型误差直接限制了策略性能上界
- 折扣因子 γ 越大，对误差越敏感
- 误差惩罚是二次的 $(1-\gamma)^{-2}$

---

## 4. 短视野 vs 长视野规划

### 4.1 视野选择的权衡

```
┌─────────────────────────────────────────────────────────────┐
│                 Planning Horizon 权衡                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  短视野 (H=1-5)              长视野 (H=15+)                  │
│  ├── 模型误差小               ├── 可利用长期信息              │
│  ├── 计算快                   ├── 更好的 credit assignment   │
│  ├── 但只看到近期收益          ├── 但复合误差大               │
│  └── 需要好的价值函数          └── 计算开销大                  │
│                                                              │
│             最优视野取决于：                                   │
│             1. 模型准确度                                     │
│             2. 任务时间尺度                                   │
│             3. 计算预算                                       │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 MBPO 的短视野策略

```python
# MBPO: Model-Based Policy Optimization
# 核心思想：用短视野模型 rollout 扩充数据

def mbpo_rollout(model, real_buffer, k=5):
    """
    从真实状态出发，只用模型展开 k 步
    """
    imagined_buffer = []

    for s in sample(real_buffer):
        trajectory = [s]
        for _ in range(k):  # 关键：只展开 k 步
            a = policy(s)
            s_next, r = model.predict(s, a)
            imagined_buffer.append((s, a, r, s_next))
            s = s_next

    return imagined_buffer
```

**MBPO 的关键洞察**：
1. 从**真实状态**出发（减少分布偏移）
2. 只展开**短视野** k=5 步（限制复合误差）
3. 用 **SAC** 等 model-free 方法优化（不依赖长期模型预测）

### 4.3 视野选择的理论指导

**定理 (Branching Time, MBPO)**：

最优 rollout 长度 k 应满足：
$$k^* = \arg\min_k \left[ \text{Model Error}(k) + \text{Value Estimation Error}(H-k) \right]$$

**实践经验**：
- 简单环境（CartPole）：k = 1-3
- 中等复杂度（MuJoCo）：k = 5
- 复杂环境（Atari 视觉）：k = 15（但需要强大模型）

---

## 5. 不确定性量化

### 5.1 为什么需要不确定性？

```
问题：模型在未见过的状态上可能完全错误

场景：
  训练数据分布: 状态空间的 30%
  策略探索到: 新的 70%

  在新区域，模型预测可能是任意的！
```

### 5.2 不确定性类型

**认知不确定性 (Epistemic Uncertainty)**
```
原因：数据不足导致的不确定性
特点：可以通过更多数据减少
估计方法：模型集成、Dropout、贝叶斯神经网络
```

**偶然不确定性 (Aleatoric Uncertainty)**
```
原因：环境本身的随机性
特点：无法通过更多数据减少
估计方法：预测分布而非点估计
```

### 5.3 模型集成 (Ensemble)

```python
class EnsembleModel:
    def __init__(self, n_models=5):
        self.models = [MLP() for _ in range(n_models)]

    def predict(self, s, a):
        """
        返回预测均值和方差
        """
        predictions = [m.predict(s, a) for m in self.models]

        mean = np.mean(predictions, axis=0)
        var = np.var(predictions, axis=0)  # 认知不确定性

        return mean, var

    def train(self, data):
        """
        每个模型用不同的数据子集训练（bootstrap）
        """
        for m in self.models:
            bootstrap_data = resample(data)
            m.train(bootstrap_data)
```

**集成的优势**：
- 预测方差反映模型不确定性
- 可用于悲观/乐观估计
- 实现简单，效果好

### 5.4 基于不确定性的决策

**1. 悲观估计 (Pessimistic Estimation)**
```python
# MOReL 风格
def pessimistic_value(model_ensemble, s, a):
    mean, var = model_ensemble.predict(s, a)
    return mean - beta * np.sqrt(var)  # 保守估计
```

**2. 乐观探索 (Optimistic Exploration)**
```python
# UCB 风格
def optimistic_value(model_ensemble, s, a):
    mean, var = model_ensemble.predict(s, a)
    return mean + beta * np.sqrt(var)  # 鼓励探索
```

**3. 不确定性惩罚 (Uncertainty Penalty)**
```python
# MBPO 风格
def penalized_reward(model_ensemble, s, a, r):
    _, var = model_ensemble.predict(s, a)
    return r - lambda_ * np.sqrt(var)  # 惩罚高不确定性区域
```

---

## 6. 关键算法全景

### 6.1 算法演进图谱

```
                    Model-Based RL 演进
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     基于规划           混合方法          纯模型方法
          │                │                │
       Dyna            MBPO            Dreamer
      (1990)          (2019)           (2020)
          │                │                │
    Prioritized       Model-free      DreamerV2
     Sweeping        + Model-based     (2021)
          │                │                │
       PILCO            PETS           DreamerV3
      (2011)           (2018)           (2023)
          │                │
       PlaNet          TD-MPC
      (2019)           (2022)
```

### 6.2 代表性算法对比

| 算法 | 年份 | 模型类型 | 规划方法 | 视野 | 特点 |
|:---|:---|:---|:---|:---|:---|
| **Dyna** | 1990 | 表格式 | Q-Learning | 1 | 开创性工作 |
| **PILCO** | 2011 | 高斯过程 | 策略梯度 | H | 样本超高效 |
| **PlaNet** | 2019 | RSSM | CEM | 15 | 视觉控制 |
| **MBPO** | 2019 | 集成 NN | SAC | 5 | 混合方法 |
| **Dreamer** | 2020 | RSSM | Actor-Critic | 15 | 想象中学习 |
| **DreamerV3** | 2023 | 离散 RSSM | Actor-Critic | 15 | 通用超参数 |

### 6.3 PILCO：极致样本效率

```python
"""
PILCO: Probabilistic Inference for Learning Control
核心：使用高斯过程 (GP) 作为动态模型
"""

def pilco_update(gp_model, policy):
    # 1. GP 给出预测分布（带不确定性）
    def rollout_distribution(s0):
        s = s0
        total_cost = 0
        for t in range(H):
            a = policy(s)
            s_mean, s_var = gp_model.predict(s, a)  # 解析传播
            s = GaussianDist(s_mean, s_var)
            total_cost += expected_cost(s)
        return total_cost

    # 2. 解析策略梯度（无需采样！）
    grad = analytic_gradient(rollout_distribution, policy)

    # 3. 更新策略
    policy.update(grad)
```

**PILCO 的局限**：
- GP 扩展性差（状态空间大时计算爆炸）
- 只适用于低维连续控制

### 6.4 PlaNet：视觉控制的突破

```
┌─────────────────────────────────────────────────────────────┐
│                      PlaNet 架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   观测 oₜ → Encoder → sₜ                                    │
│                        ↓                                     │
│   RSSM: (hₜ, sₜ, aₜ) → (hₜ₊₁, ŝₜ₊₁)                        │
│                        ↓                                     │
│   Planning: CEM 在潜在空间规划                               │
│             argmax_a E[Σ r̂ₜ]                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**CEM (Cross-Entropy Method)**：
```python
def cem_planning(model, s0, n_iter=10, n_samples=1000, elite_ratio=0.1):
    """
    在潜在空间中使用 CEM 规划
    """
    # 初始化动作分布
    action_mean = np.zeros((H, action_dim))
    action_var = np.ones((H, action_dim))

    for _ in range(n_iter):
        # 采样动作序列
        actions = np.random.normal(action_mean, action_var, (n_samples, H, action_dim))

        # 评估每个序列
        returns = []
        for action_seq in actions:
            r_total = model.simulate(s0, action_seq)
            returns.append(r_total)

        # 选择精英样本
        elite_idx = np.argsort(returns)[-int(n_samples * elite_ratio):]
        elite_actions = actions[elite_idx]

        # 更新分布
        action_mean = np.mean(elite_actions, axis=0)
        action_var = np.var(elite_actions, axis=0)

    return action_mean[0]  # 返回第一个动作
```

---

## 7. 前沿研究方向

### 7.1 World Models 与基础模型

**趋势**：将预训练的大模型用于世界建模

```
传统 MBRL:
  任务特定数据 → 任务特定模型 → 任务特定策略

基础模型时代:
  互联网规模数据 → 通用世界模型 → 零样本/少样本适应

例子:
  - Genie (2024): 从视频学习可控世界模型
  - Sora (2024): 视频生成即世界模拟
```

### 7.2 层次化世界模型

```
┌─────────────────────────────────────────────────────────────┐
│                   层次化世界模型                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   高层: 任务级抽象                                           │
│         "去厨房" → "打开冰箱" → "拿牛奶"                     │
│              │                                               │
│   中层: 技能级                                               │
│         "导航到位置 X" → 一系列低级动作                      │
│              │                                               │
│   底层: 物理级                                               │
│         电机控制、力矩计算                                    │
│                                                              │
│   优势:                                                      │
│   - 不同层次使用不同时间尺度                                  │
│   - 高层规划更抽象、更稳定                                    │
│   - 底层适应物理细节                                         │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 因果世界模型

**问题**：标准 MBRL 学习相关性，而非因果性

```
例子：
  观察: "灯亮" 总是伴随 "开关被按下"

  相关性模型: P(灯亮 | 开关按下) 高
  因果模型: 开关 → 灯

  区别在于干预 (Intervention):
  - 相关性: 如果观察到灯亮，推断开关被按
  - 因果: 如果按开关，灯会亮；如果灯亮（被阳光照亮），不能推断开关
```

**因果世界模型的优势**：
1. **反事实推理**：如果当时选择了不同的动作会怎样？
2. **迁移学习**：因果机制在不同环境中更稳定
3. **可解释性**：因果结构更易理解

### 7.4 离线 MBRL

**场景**：只有固定的历史数据，不能与环境交互

```python
# 离线 MBRL 的挑战

问题：
  数据集 D = {(s, a, r, s')} 来自旧策略 π_old
  新策略 π_new 可能访问 D 中没有的状态

解决方案:
  1. 悲观估计 (MOReL, MOPO)
     - 在不确定区域假设最坏情况

  2. 支持约束 (COMBO)
     - 约束策略不要离开数据分布太远

  3. 模型正则化
     - 在数据分布外惩罚模型预测
```

---

## 8. 实践指南

### 8.1 选择 MBRL 的时机

**适合 MBRL**：
- ✅ 真实交互昂贵（机器人、自动驾驶）
- ✅ 环境动态相对可预测
- ✅ 有充足的计算资源
- ✅ 需要良好的样本效率

**不适合 MBRL**：
- ❌ 环境高度随机
- ❌ 动态极其复杂
- ❌ 计算资源有限
- ❌ 需要极高的渐近性能

### 8.2 实现建议

**模型设计**：
```python
# 1. 使用概率模型
class ProbabilisticModel(nn.Module):
    def forward(self, s, a):
        # 输出分布参数，不是点估计
        mean, log_var = self.network(s, a)
        return mean, log_var

# 2. 模型集成
ensemble = [ProbabilisticModel() for _ in range(5)]

# 3. 正则化
loss = nll_loss + 0.01 * weight_decay + 0.1 * max_logvar_penalty
```

**Planning 策略**：
```python
# 1. 短视野 + 价值函数
H = 5  # 短视野
for t in range(H):
    r_total += gamma**t * r_t
r_total += gamma**H * value_function(s_H)  # 价值函数补全

# 2. 从真实状态出发
start_states = sample(replay_buffer)  # 不是随机初始化
```

### 8.3 调试技巧

**1. 验证模型质量**：
```python
def evaluate_model(model, test_data):
    errors = []
    for s, a, s_true in test_data:
        s_pred = model.predict(s, a)
        errors.append(np.linalg.norm(s_pred - s_true))

    print(f"单步预测误差: {np.mean(errors):.4f}")

    # 多步预测误差
    for H in [1, 5, 10, 20]:
        multi_step_error = evaluate_multi_step(model, test_data, H)
        print(f"{H}步预测误差: {multi_step_error:.4f}")
```

**2. 监控不确定性**：
```python
def plot_uncertainty(ensemble, states):
    _, variances = ensemble.predict(states)
    plt.hist(variances, bins=50)
    plt.xlabel("Prediction Variance")
    plt.title("Model Uncertainty Distribution")
```

**3. 可视化 rollout**：
```python
def visualize_rollout(model, real_env, policy, n_steps=50):
    # 对比真实轨迹和模型预测
    real_states, pred_states = [], []
    s = real_env.reset()

    for t in range(n_steps):
        a = policy(s)
        s_real = real_env.step(a)
        s_pred = model.predict(s, a)

        real_states.append(s_real)
        pred_states.append(s_pred)
        s = s_real

    plot_comparison(real_states, pred_states)
```

---

## 9. 总结

### 9.1 MBRL 的核心洞察

1. **样本效率来自模型复用**：一次交互的数据可以无限次生成虚拟经验

2. **模型误差是核心瓶颈**：所有 MBRL 方法都在解决这个问题

3. **短视野是实用的妥协**：在模型准确的范围内规划

4. **不确定性是关键信号**：知道什么是不知道的

### 9.2 关键公式速查

| 概念 | 公式 |
|:---|:---|
| 模型学习 | $\min_\theta \mathbb{E}[\|f_\theta(s,a) - s'\|^2]$ |
| 复合误差 | $\epsilon_H \leq \frac{L^H - 1}{L-1} \cdot \epsilon$ |
| 性能损失 | $\eta[P,\pi] \geq \eta[\hat{P},\pi] - O(\epsilon_m / (1-\gamma)^2)$ |
| 悲观估计 | $V_{pessimistic}(s) = \mu(s) - \beta \cdot \sigma(s)$ |

### 9.3 延伸阅读

**综述论文**：
- Moerland et al. (2023). "Model-based Reinforcement Learning: A Survey"
- Wang et al. (2019). "Benchmarking Model-Based Reinforcement Learning"

**经典论文**：
- PILCO (2011): 高斯过程动态模型
- PlaNet (2019): 视觉世界模型
- MBPO (2019): 短视野混合方法
- Dreamer (2020): 想象中的 Actor-Critic

**代码资源**：
- `experiments/5_dyna_q.py` - Dyna-Q 实现
- MBPO: https://github.com/jannerm/mbpo
- DreamerV3: https://github.com/danijar/dreamerv3

---

*最后更新: 2025-12-17*
