# 好奇心驱动探索：内在动机与世界模型

> 当外部奖励稀疏或缺失时，如何让智能体主动探索？好奇心驱动探索提供了一种基于内在动机的解决方案。

## 1. 为什么需要好奇心？

### 1.1 稀疏奖励问题

```
┌─────────────────────────────────────────────────────────────┐
│                   稀疏奖励困境                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   环境: 100 步迷宫，只有终点 +1 奖励                         │
│                                                              │
│   随机探索:                                                  │
│   ├── P(到达终点) ≈ (1/4)^100 ≈ 0                          │
│   ├── 几乎所有轨迹奖励 = 0                                   │
│   └── 没有学习信号！                                         │
│                                                              │
│   需要: 内在动机 (Intrinsic Motivation)                     │
│   ├── 不依赖外部奖励                                         │
│   ├── 鼓励探索新颖状态                                       │
│   └── 提供持续的学习信号                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 内在动机的类型

| 类型 | 原理 | 代表方法 |
|:---|:---|:---|
| **预测误差** | 奖励预测困难的状态 | ICM, RND |
| **信息增益** | 奖励减少不确定性的动作 | VIME, Plan2Explore |
| **状态覆盖** | 奖励访问新状态 | Count-based, Hash-based |
| **技能发现** | 奖励学习多样技能 | DIAYN, VIC |

---

## 2. ICM：Intrinsic Curiosity Module

### 2.1 核心思想

```
┌─────────────────────────────────────────────────────────────┐
│                     ICM 架构                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   观测 sₜ, sₜ₊₁                                              │
│        ↓                                                     │
│   ┌─────────┐                                               │
│   │ Encoder │  φ(s) - 特征提取                              │
│   └────┬────┘                                               │
│        ↓                                                     │
│   φ(sₜ), φ(sₜ₊₁)                                            │
│        │                                                     │
│   ┌────┴────┐        ┌──────────┐                           │
│   │         │        │          │                           │
│   ↓         ↓        ↓          │                           │
│ ┌─────────────┐  ┌─────────────┐│                           │
│ │ Inverse     │  │ Forward     ││                           │
│ │ Model       │  │ Model       ││                           │
│ │ â = g(φₜ,φₜ₊₁)│  │ φ̂ₜ₊₁=f(φₜ,a)││                           │
│ └─────────────┘  └─────────────┘│                           │
│        │                │       │                           │
│        ↓                ↓       │                           │
│   Inverse Loss      Forward Error = Curiosity Reward        │
│   (训练 encoder)    rᵢ = ||φ̂ₜ₊₁ - φₜ₊₁||²                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数学公式

**Inverse Model (逆模型)**：
$$\hat{a}_t = g(φ(s_t), φ(s_{t+1}); θ_I)$$

**Inverse Loss**：
$$L_I = \mathbb{E}[\text{CE}(\hat{a}_t, a_t)]$$

**Forward Model (前向模型)**：
$$\hat{φ}(s_{t+1}) = f(φ(s_t), a_t; θ_F)$$

**Intrinsic Reward**：
$$r^i_t = \frac{η}{2} \|\hat{φ}(s_{t+1}) - φ(s_{t+1})\|^2$$

**Total Reward**：
$$r_t = r^e_t + β \cdot r^i_t$$

### 2.3 为什么需要逆模型？

**问题**：如果直接用像素空间的预测误差作为奖励会怎样？

```
像素空间问题:
  - 树叶飘动 → 高预测误差 → 高奖励
  - 电视噪声 → 高预测误差 → 高奖励
  - 这些与智能体的动作无关！

逆模型的作用:
  - 只编码与动作相关的特征
  - 过滤掉环境中的随机性
  - 学习"可控"的状态表示
```

### 2.4 实现代码

```python
class ICM(nn.Module):
    """
    Intrinsic Curiosity Module
    """
    def __init__(self, state_dim, action_dim, feature_dim=256):
        super().__init__()

        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # 逆模型: (φₜ, φₜ₊₁) → â
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        # 前向模型: (φₜ, a) → φ̂ₜ₊₁
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, s, a, s_next):
        # 编码
        phi = self.encoder(s)
        phi_next = self.encoder(s_next)

        # 逆模型预测
        a_pred = self.inverse_model(torch.cat([phi, phi_next], dim=-1))

        # 前向模型预测
        phi_next_pred = self.forward_model(torch.cat([phi, a], dim=-1))

        return a_pred, phi_next_pred, phi_next

    def intrinsic_reward(self, s, a, s_next):
        """计算内在奖励"""
        _, phi_next_pred, phi_next = self.forward(s, a, s_next)
        reward = 0.5 * (phi_next_pred - phi_next.detach()).pow(2).sum(dim=-1)
        return reward

    def loss(self, s, a, s_next):
        """计算训练损失"""
        a_pred, phi_next_pred, phi_next = self.forward(s, a, s_next)

        # 逆模型损失 (分类)
        inverse_loss = F.cross_entropy(a_pred, a)

        # 前向模型损失 (回归)
        forward_loss = 0.5 * (phi_next_pred - phi_next.detach()).pow(2).mean()

        return inverse_loss + forward_loss
```

---

## 3. RND：Random Network Distillation

### 3.1 核心思想

比 ICM 更简单：用一个固定的随机网络作为"目标"，训练另一个网络去预测它。

```
┌─────────────────────────────────────────────────────────────┐
│                     RND 架构                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   观测 s                                                     │
│       │                                                      │
│       ├────────────────┬────────────────┐                   │
│       ↓                ↓                │                   │
│   ┌─────────┐     ┌─────────┐           │                   │
│   │ Target  │     │Predictor│           │                   │
│   │ Network │     │ Network │           │                   │
│   │ (固定)   │     │ (可训练) │           │                   │
│   └────┬────┘     └────┬────┘           │                   │
│        ↓               ↓                │                   │
│     f(s)           f̂(s)                │                   │
│        │               │                │                   │
│        └───────┬───────┘                │                   │
│                ↓                        │                   │
│      r_intrinsic = ||f(s) - f̂(s)||²    │                   │
│                                                              │
│   直觉:                                                      │
│   - 见过的状态: 预测准确, 低奖励                             │
│   - 新状态: 预测不准, 高奖励                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数学原理

**Target Network** (固定随机权重)：
$$f(s; θ^*) \text{ where } θ^* \sim \mathcal{N}(0, I)$$

**Predictor Network** (可训练)：
$$\hat{f}(s; θ)$$

**Intrinsic Reward**：
$$r^i_t = \|f(s_t) - \hat{f}(s_t)\|^2$$

**Predictor Loss**：
$$L = \mathbb{E}_{s \sim D}[\|f(s) - \hat{f}(s)\|^2]$$

### 3.3 为什么 RND 有效？

**核心洞察**：神经网络泛化能力有限

```
训练数据分布 D_train:
  - Predictor 在 D_train 上预测准确
  - 误差 ≈ 0

新状态 s_new ∉ D_train:
  - Predictor 无法泛化
  - 误差 >> 0
  - 高内在奖励！

本质上是一种"访问计数"的神经网络近似
```

### 3.4 实现代码

```python
class RND(nn.Module):
    """
    Random Network Distillation
    """
    def __init__(self, state_dim, feature_dim=512):
        super().__init__()

        # 目标网络 (固定)
        self.target = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        # 冻结目标网络
        for param in self.target.parameters():
            param.requires_grad = False

        # 预测网络 (可训练)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        # 奖励归一化
        self.reward_mean = 0
        self.reward_std = 1

    def forward(self, s):
        target_features = self.target(s)
        predicted_features = self.predictor(s)
        return target_features, predicted_features

    def intrinsic_reward(self, s):
        """计算内在奖励"""
        with torch.no_grad():
            target_features = self.target(s)
        predicted_features = self.predictor(s)

        reward = (target_features - predicted_features).pow(2).sum(dim=-1)

        # 归一化
        reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)

        return reward

    def update_normalization(self, rewards):
        """更新奖励归一化统计量"""
        self.reward_mean = 0.99 * self.reward_mean + 0.01 * rewards.mean()
        self.reward_std = 0.99 * self.reward_std + 0.01 * rewards.std()

    def loss(self, s):
        """预测网络损失"""
        target_features, predicted_features = self.forward(s)
        return (target_features.detach() - predicted_features).pow(2).mean()
```

---

## 4. Plan2Explore：世界模型 + 好奇心

### 4.1 核心思想

在世界模型框架下进行好奇心驱动探索：

```
┌─────────────────────────────────────────────────────────────┐
│                  Plan2Explore 框架                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   阶段 1: 无奖励探索                                         │
│   ├── 用世界模型探索环境                                     │
│   ├── 内在奖励 = 世界模型的预测不确定性                      │
│   └── 最大化信息增益                                         │
│                                                              │
│   阶段 2: 零样本/少样本任务适应                              │
│   ├── 冻结世界模型                                           │
│   ├── 给定任务奖励                                           │
│   └── 快速学习策略                                           │
│                                                              │
│   优势:                                                      │
│   - 探索与任务解耦                                           │
│   - 可迁移到多种下游任务                                     │
│   - 利用世界模型的规划能力                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 信息增益作为内在奖励

**集成模型的不确定性**：

```python
def information_gain_reward(model_ensemble, s, a):
    """
    用模型集成的分歧作为内在奖励

    直觉：模型不确定的地方，探索价值高
    """
    predictions = []
    for model in model_ensemble:
        s_next = model.predict(s, a)
        predictions.append(s_next)

    # 计算预测方差 (认知不确定性)
    predictions = torch.stack(predictions)
    variance = predictions.var(dim=0).mean()

    return variance
```

### 4.3 与 Dreamer 结合

```python
class Plan2Explore:
    """
    基于 Dreamer 的好奇心探索
    """
    def __init__(self, env):
        # 世界模型 (RSSM)
        self.world_model = RSSM(...)

        # 探索策略 (在想象中训练)
        self.explorer = ActorCritic(...)

        # 模型集成 (用于不确定性估计)
        self.ensemble = [RSSM(...) for _ in range(5)]

    def explore(self, n_steps):
        """无奖励探索阶段"""
        for step in range(n_steps):
            # 选择动作 (最大化信息增益)
            a = self.explorer.act(s)

            # 环境交互
            s_next, _, done = env.step(a)

            # 计算内在奖励 (模型不确定性)
            r_intrinsic = self.compute_disagreement(s, a)

            # 训练世界模型
            self.world_model.train(s, a, s_next)

            # 在想象中训练探索策略
            self.train_explorer_in_imagination(r_intrinsic)

    def compute_disagreement(self, s, a):
        """计算模型分歧作为奖励"""
        predictions = [m.predict(s, a) for m in self.ensemble]
        return torch.stack(predictions).var(dim=0).mean()

    def adapt_to_task(self, task_reward_fn, n_steps):
        """快速适应下游任务"""
        # 冻结世界模型
        self.world_model.eval()

        # 训练任务特定策略
        task_policy = ActorCritic(...)
        for step in range(n_steps):
            # 在想象中用任务奖励训练
            self.train_in_imagination(task_policy, task_reward_fn)

        return task_policy
```

---

## 5. 实验对比

### 5.1 Montezuma's Revenge

最经典的稀疏奖励基准测试：

| 方法 | 平均得分 | 说明 |
|:---|:---|:---|
| DQN | ~0 | 几乎无法学习 |
| A3C | ~0 | 几乎无法学习 |
| ICM | ~2500 | 显著提升 |
| RND | ~8000 | 更好的探索 |
| Go-Explore | ~35000+ | 人工设计的归档机制 |

### 5.2 探索效率对比

```
┌─────────────────────────────────────────────────────────────┐
│              状态覆盖率 vs 训练步数                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  覆盖率                                                      │
│   ↑                                                          │
│   │     ╱ RND                                                │
│   │    ╱                                                     │
│   │   ╱  ╱ ICM                                               │
│   │  ╱  ╱                                                    │
│   │ ╱  ╱   ╱ Random                                          │
│   │╱  ╱   ╱                                                  │
│   └─────────────────────────────────→ 步数                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 与世界模型的关系

### 6.1 互补关系

```
世界模型                    好奇心探索
    ↓                           ↓
预测未来状态               发现新状态
    ↓                           ↓
在想象中规划               提供学习信号
    ↓                           ↓
    └───────────┬───────────────┘
                ↓
        高效探索 + 高效规划
```

### 6.2 结合方式

**1. 模型误差作为奖励** (ICM 风格)
```python
r_intrinsic = ||world_model.predict(s, a) - s_next||²
```

**2. 模型不确定性作为奖励** (Plan2Explore 风格)
```python
r_intrinsic = ensemble_variance(s, a)
```

**3. 信息增益** (贝叶斯风格)
```python
r_intrinsic = KL[p_new(model) || p_old(model)]
```

---

## 7. 实践指南

### 7.1 选择哪种方法？

| 场景 | 推荐方法 | 原因 |
|:---|:---|:---|
| 视觉任务 | ICM | 学习动作相关特征 |
| 状态空间任务 | RND | 简单高效 |
| 需要迁移 | Plan2Explore | 探索与任务解耦 |
| 结合世界模型 | 模型不确定性 | 自然融合 |

### 7.2 关键超参数

| 参数 | 含义 | 建议值 |
|:---|:---|:---|
| `intrinsic_coef` | 内在奖励系数 β | 0.01-0.1 |
| `feature_dim` | 特征维度 | 256-512 |
| `normalize_reward` | 是否归一化 | True |
| `clip_reward` | 奖励裁剪 | [-5, 5] |

### 7.3 常见问题

**1. "Noisy-TV" 问题**
```
问题：智能体被随机性吸引（如电视噪声）
解决：
- 使用 RND 代替纯预测误差
- ICM 的逆模型可以过滤无关随机性
```

**2. 内在奖励消失**
```
问题：训练后期内在奖励趋近于 0
解决：
- 归一化内在奖励
- 使用适当的衰减策略
```

**3. 探索-利用平衡**
```
问题：过度探索，忽略外部奖励
解决：
- 动态调整 β 系数
- 使用课程学习
```

---

## 8. 总结

### 8.1 核心洞察

1. **稀疏奖励需要内在动机**：纯外部奖励在稀疏场景下无法学习
2. **预测误差 ≈ 新颖性**：难以预测的状态可能是新状态
3. **与世界模型自然结合**：模型的不确定性就是探索信号
4. **探索与任务解耦**：先探索，后适应

### 8.2 方法对比速查

| 方法 | 内在奖励 | 优点 | 缺点 |
|:---|:---|:---|:---|
| ICM | 前向预测误差 | 过滤随机性 | 需要训练逆模型 |
| RND | 随机网络蒸馏误差 | 简单高效 | 可能被随机性吸引 |
| Plan2Explore | 模型集成分歧 | 与世界模型结合 | 计算开销大 |

### 8.3 延伸阅读

**论文**：
- Pathak et al. (2017). "Curiosity-driven Exploration by Self-Supervised Prediction" (ICM)
- Burda et al. (2019). "Exploration by Random Network Distillation" (RND)
- Sekar et al. (2020). "Planning to Explore via Self-Supervised World Models" (Plan2Explore)

**代码**：
- ICM: https://github.com/pathak22/noreward-rl
- RND: https://github.com/openai/random-network-distillation

---

*最后更新: 2025-12-18*
