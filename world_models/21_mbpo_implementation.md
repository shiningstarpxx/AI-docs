# MBPO：Model-Based Policy Optimization 深度解析

> MBPO 代表了 Model-Based RL 的实用主义路线：用短视野模型 rollout 扩充数据，结合 model-free 方法优化策略。

## 1. 核心思想

### 1.1 MBPO 的定位

```
┌─────────────────────────────────────────────────────────────┐
│              Model-Based RL 的两种路线                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   纯 Model-Based (Dreamer):                                 │
│   ├── 完全在想象中训练                                       │
│   ├── 长视野 rollout (H=15)                                 │
│   ├── 依赖模型准确性                                         │
│   └── Actor-Critic in latent space                         │
│                                                              │
│   混合方法 (MBPO):                                          │
│   ├── 用模型扩充真实数据                                     │
│   ├── 短视野 rollout (k=1-5)                                │
│   ├── 限制模型误差影响                                       │
│   └── Model-free 算法优化 (SAC)                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 为什么需要 MBPO？

**问题**：长视野模型 rollout 会累积误差

```
真实轨迹:   s₀ → s₁ → s₂ → ... → s₁₅
模型轨迹:   s₀ → ŝ₁ → ŝ₂ → ... → ŝ₁₅
                 ε    2ε         15ε (误差累积!)

如果 ε = 5% 单步误差:
  15 步后: 误差可能达到 75%+
```

**MBPO 的解决方案**：
1. 从**真实状态**出发（而非模型生成的状态）
2. 只展开**短视野** k 步（限制误差累积）
3. 用 **SAC** 等 model-free 方法优化（不依赖长期模型预测）

---

## 2. 算法框架

### 2.1 MBPO 伪代码

```python
def mbpo(env, n_iterations):
    # 初始化
    D_env = ReplayBuffer()      # 真实数据
    D_model = ReplayBuffer()    # 模型生成数据
    policy = SAC()              # 策略网络
    model = EnsembleModel(n=7)  # 概率集成模型

    for iteration in range(n_iterations):
        # 1. 环境交互：收集真实数据
        for _ in range(E):  # E 步环境交互
            a = policy.sample(s)
            s', r, done = env.step(a)
            D_env.add(s, a, r, s', done)
            s = s'

        # 2. 模型训练：用真实数据训练模型
        model.train(D_env)

        # 3. 模型 Rollout：从真实状态出发，生成虚拟数据
        for s in D_env.sample_states(M):  # M 个起始状态
            for k in range(rollout_length):  # 短视野 k 步
                a = policy.sample(s)
                s', r = model.predict(s, a)
                D_model.add(s, a, r, s')
                s = s'

        # 4. 策略优化：用混合数据训练
        for _ in range(G):  # G 步梯度更新
            batch = sample_mixed(D_env, D_model, ratio)
            policy.update(batch)  # SAC 更新
```

### 2.2 关键组件

**1. 概率集成模型 (Probabilistic Ensemble)**
```python
class EnsembleModel:
    def __init__(self, n_models=7, elite_size=5):
        self.models = [ProbabilisticMLP() for _ in range(n_models)]
        self.elite_size = elite_size

    def train(self, data):
        # 每个模型用不同的 bootstrap 数据训练
        for i, model in enumerate(self.models):
            bootstrap_data = data.bootstrap_sample()
            model.train(bootstrap_data)

        # 选择验证损失最低的 elite 模型
        self.elites = select_top_k(self.models, k=self.elite_size)

    def predict(self, s, a):
        # 随机选择一个 elite 模型预测
        model = random.choice(self.elites)
        mean, var = model.predict(s, a)
        s_next = mean + sqrt(var) * randn()  # 采样
        return s_next
```

**2. 短视野 Rollout**
```python
def model_rollout(model, policy, start_states, k):
    """
    从真实状态出发，用模型展开 k 步

    关键：
    - start_states 来自真实经验
    - k 很小 (1-5)，限制误差累积
    """
    rollouts = []
    for s in start_states:
        for step in range(k):
            a = policy.sample(s)
            s_next, r = model.predict(s, a)
            rollouts.append((s, a, r, s_next))
            s = s_next
    return rollouts
```

**3. 混合缓冲区采样**
```python
def sample_mixed(D_env, D_model, real_ratio=0.05):
    """
    从真实和模型数据中按比例采样

    默认：5% 真实数据 + 95% 模型数据
    """
    batch_size = 256
    n_real = int(batch_size * real_ratio)
    n_model = batch_size - n_real

    real_batch = D_env.sample(n_real)
    model_batch = D_model.sample(n_model)

    return concat(real_batch, model_batch)
```

---

## 3. 理论分析

### 3.1 分支时间 (Branching Time)

**核心问题**：rollout 长度 k 应该选多大？

**定理 (MBPO Paper)**：

假设模型误差 $\epsilon_m$，策略在模型下的回报为 $\eta_{\hat{M}}[\pi]$，则真实回报的下界为：

$$\eta_M[\pi] \geq \eta_{\hat{M}}[\pi] - C \cdot \epsilon_m \cdot k$$

其中 $C$ 是与折扣因子和奖励范围相关的常数。

**直觉**：
- k 越大，利用模型的信息越多
- 但 k 越大，误差项 $C \cdot \epsilon_m \cdot k$ 越大
- 存在最优 k* 平衡两者

### 3.2 最优 Rollout 长度

```
┌─────────────────────────────────────────────────────────────┐
│           Rollout 长度 k 的选择                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   误差                                                       │
│   ↑                                                          │
│   │      ╱ 模型误差 (随 k 增加)                              │
│   │     ╱                                                    │
│   │    ╱                                                     │
│   │   ╱                                                      │
│   │  ╱  ＼                                                   │
│   │ ╱    ＼ 价值估计误差 (随 k 减少)                         │
│   │╱      ＼                                                 │
│   └─────────────────────────────→ k                         │
│        k*                                                    │
│                                                              │
│   最优 k*: 平衡模型误差和价值估计误差                        │
└─────────────────────────────────────────────────────────────┘
```

**实践指导**：
- MuJoCo 任务：k = 1-5
- 简单环境：k = 1-3
- 复杂环境：需要实验确定

### 3.3 样本效率分析

**MBPO vs SAC 样本效率**：

| 环境 | SAC | MBPO | 提升 |
|:---|:---|:---|:---|
| HalfCheetah | 3M steps | 300K steps | **10x** |
| Hopper | 1M steps | 125K steps | **8x** |
| Walker2d | 3M steps | 300K steps | **10x** |
| Ant | 3M steps | 300K steps | **10x** |

---

## 4. 实现细节

### 4.1 模型架构

```python
class ProbabilisticMLP(nn.Module):
    """
    输出高斯分布的 MLP

    预测 P(s', r | s, a) = N(μ, σ²)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=200):
        super().__init__()

        input_dim = state_dim + action_dim
        output_dim = state_dim + 1  # 状态 + 奖励

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # 或 ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # 输出均值和方差
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

        # 方差范围限制
        self.min_logvar = nn.Parameter(torch.ones(output_dim) * -10)
        self.max_logvar = nn.Parameter(torch.ones(output_dim) * 0.5)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)

        mean = self.mean_head(h)
        logvar = self.logvar_head(h)

        # 软约束方差范围
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

    def loss(self, s, a, target):
        """
        负对数似然损失
        """
        mean, logvar = self.forward(s, a)
        var = torch.exp(logvar)

        # NLL = 0.5 * (log(var) + (target - mean)² / var)
        nll = 0.5 * (logvar + (target - mean) ** 2 / var)

        # 正则化：惩罚过大/过小的方差
        reg = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())

        return nll.mean() + reg
```

### 4.2 SAC 策略优化

```python
class SAC:
    """
    Soft Actor-Critic

    目标：最大化 E[Σ r + α H(π)]
    """
    def __init__(self, state_dim, action_dim):
        # Actor: 输出动作分布
        self.actor = GaussianPolicy(state_dim, action_dim)

        # Critic: 双 Q 网络
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)

        # 目标网络
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # 温度参数 (自动调节)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.target_entropy = -action_dim

    def update(self, batch):
        s, a, r, s_next, done = batch

        # 1. Critic 更新
        with torch.no_grad():
            a_next, log_prob = self.actor.sample(s_next)
            q1_next = self.q1_target(s_next, a_next)
            q2_next = self.q2_target(s_next, a_next)
            q_next = torch.min(q1_next, q2_next) - self.alpha * log_prob
            target = r + gamma * (1 - done) * q_next

        q1_loss = F.mse_loss(self.q1(s, a), target)
        q2_loss = F.mse_loss(self.q2(s, a), target)

        # 2. Actor 更新
        a_new, log_prob = self.actor.sample(s)
        q1_new = self.q1(s, a_new)
        q2_new = self.q2(s, a_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        # 3. 温度更新
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        # 4. 更新目标网络 (软更新)
        soft_update(self.q1_target, self.q1, tau=0.005)
        soft_update(self.q2_target, self.q2, tau=0.005)
```

### 4.3 完整训练循环

```python
def train_mbpo(env, config):
    # 初始化
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = EnsembleModel(state_dim, action_dim, n_models=7)
    policy = SAC(state_dim, action_dim)

    D_env = ReplayBuffer(capacity=1_000_000)
    D_model = ReplayBuffer(capacity=1_000_000)

    # 预热：收集随机数据
    s = env.reset()
    for _ in range(5000):
        a = env.action_space.sample()
        s_next, r, done, _ = env.step(a)
        D_env.add(s, a, r, s_next, done)
        s = s_next if not done else env.reset()

    # 主训练循环
    total_steps = 0
    for epoch in range(config.n_epochs):

        # 1. 环境交互
        for _ in range(config.env_steps_per_epoch):
            a = policy.sample(s)
            s_next, r, done, _ = env.step(a)
            D_env.add(s, a, r, s_next, done)
            s = s_next if not done else env.reset()
            total_steps += 1

        # 2. 模型训练
        model.train(D_env, epochs=config.model_train_epochs)

        # 3. 模型 Rollout
        rollout_length = min(
            config.max_rollout_length,
            rollout_schedule(epoch)  # 可以随训练增加
        )

        start_states = D_env.sample_states(config.n_rollout_starts)
        model_data = model_rollout(model, policy, start_states, rollout_length)
        D_model.add_batch(model_data)

        # 4. 策略优化
        for _ in range(config.policy_updates_per_epoch):
            batch = sample_mixed(D_env, D_model, real_ratio=0.05)
            policy.update(batch)

        # 评估
        if epoch % config.eval_interval == 0:
            eval_reward = evaluate(env, policy)
            print(f"Epoch {epoch} | Steps {total_steps} | Reward {eval_reward:.1f}")
```

---

## 5. 超参数指南

### 5.1 关键超参数

| 参数 | 含义 | 建议值 |
|:---|:---|:---|
| `n_models` | 集成模型数量 | 7 |
| `elite_size` | 使用的精英模型数 | 5 |
| `rollout_length` | 模型展开步数 | 1-5 |
| `real_ratio` | 真实数据比例 | 0.05 |
| `model_train_freq` | 模型训练频率 | 250 steps |
| `rollout_batch_size` | 每次 rollout 的起始状态数 | 100K |

### 5.2 Rollout 长度调度

```python
def rollout_schedule(epoch, min_length=1, max_length=15, start_epoch=20):
    """
    随训练进行逐渐增加 rollout 长度

    理由：模型在训练初期不准确，应该用短 rollout
    """
    if epoch < start_epoch:
        return min_length

    progress = (epoch - start_epoch) / (100 - start_epoch)
    return int(min_length + progress * (max_length - min_length))
```

---

## 6. MBPO vs 其他方法

### 6.1 与 Dreamer 对比

| 维度 | MBPO | Dreamer |
|:---|:---|:---|
| **模型类型** | 集成 MLP | RSSM (潜在空间) |
| **Rollout 长度** | 1-5 步 | 15 步 |
| **策略优化** | SAC (model-free) | Actor-Critic (in imagination) |
| **起始状态** | 真实状态 | 潜在状态 |
| **适用场景** | 状态空间任务 | 视觉任务 |
| **样本效率** | 高 | 更高 |
| **实现复杂度** | 中等 | 高 |

### 6.2 与 Dyna 对比

| 维度 | Dyna | MBPO |
|:---|:---|:---|
| **模型** | 表格式/简单 | 神经网络集成 |
| **Rollout** | 1 步 | 多步 (1-5) |
| **策略优化** | Q-Learning | SAC |
| **不确定性** | 无 | 集成方差 |
| **规模** | 小状态空间 | 连续控制 |

---

## 7. 实践建议

### 7.1 常见问题

**1. 模型不准确导致性能下降**
```
症状：策略在模型中表现好，真实环境差
解决：
- 减小 rollout 长度
- 增加真实数据比例
- 增加模型集成数量
```

**2. 训练不稳定**
```
症状：性能波动大
解决：
- 使用方差下界约束
- 增加模型训练 epochs
- 降低策略学习率
```

**3. 样本效率没有提升**
```
症状：和 SAC 差不多
解决：
- 检查模型损失是否收敛
- 增加 rollout 数量
- 确保从真实状态开始 rollout
```

### 7.2 调试技巧

```python
# 1. 监控模型误差
def evaluate_model_error(model, D_env, n_samples=1000):
    errors = []
    for s, a, r, s_next, _ in D_env.sample(n_samples):
        s_pred, r_pred = model.predict(s, a)
        error = np.linalg.norm(s_pred - s_next)
        errors.append(error)
    return np.mean(errors), np.std(errors)

# 2. 可视化预测
def visualize_predictions(model, trajectory):
    real_states = [t[0] for t in trajectory]
    pred_states = []

    s = real_states[0]
    for t in trajectory:
        a = t[1]
        s_pred, _ = model.predict(s, a)
        pred_states.append(s_pred)
        s = s_pred  # 用预测继续

    plot_comparison(real_states, pred_states)

# 3. 检查数据分布
def check_data_distribution(D_env, D_model):
    env_states = np.array([d[0] for d in D_env.sample(1000)])
    model_states = np.array([d[0] for d in D_model.sample(1000)])

    print(f"Env data: mean={env_states.mean(0)}, std={env_states.std(0)}")
    print(f"Model data: mean={model_states.mean(0)}, std={model_states.std(0)}")
```

---

## 8. 总结

### 8.1 MBPO 的核心贡献

1. **实用的混合方法**：结合 model-based 的样本效率和 model-free 的稳定性
2. **短视野策略**：通过限制 rollout 长度控制模型误差
3. **理论保证**：提供了性能下界的理论分析
4. **通用性**：可以与多种 model-free 算法结合

### 8.2 关键公式速查

| 概念 | 公式 |
|:---|:---|
| 性能下界 | $\eta_M[\pi] \geq \eta_{\hat{M}}[\pi] - C \cdot \epsilon_m \cdot k$ |
| 模型损失 | $\mathcal{L} = \mathbb{E}[\log \sigma^2 + \frac{(s' - \mu)^2}{\sigma^2}]$ |
| SAC 目标 | $J(\pi) = \mathbb{E}[\sum_t r_t + \alpha H(\pi(\cdot|s_t))]$ |

### 8.3 延伸阅读

**论文**：
- Janner et al. (2019). "When to Trust Your Model: Model-Based Policy Optimization"
- Chua et al. (2018). "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS)

**代码**：
- 官方实现: https://github.com/jannerm/mbpo
- 简化实现: https://github.com/Xingyu-Lin/mbpo_pytorch

---

*最后更新: 2025-12-18*
