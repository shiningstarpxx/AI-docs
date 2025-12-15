# World Models vs Dreamer：深度对比分析

> 从 2018 到 2023，基于世界模型的强化学习演进

## 核心思想对比

### 共同目标

两者都在回答同一个问题：**如何在"脑中"学习，减少与真实环境的交互？**

```
传统 Model-Free RL:
  真实环境交互 → 获得奖励 → 更新策略
  问题：需要海量交互，样本效率低

Model-Based RL (World Models/Dreamer):
  真实环境交互 → 学习世界模型 → 在想象中训练策略
  优势：可以在"梦中"无限练习
```

### 哲学差异

```
World Models (2018):
  "先理解世界，再学习决策"
  分阶段：感知(V) → 记忆(M) → 决策(C)
  强调模块化和可解释性

Dreamer (2020-2023):
  "理解世界是为了更好地决策"
  端到端：世界模型和策略联合优化
  强调性能和通用性
```

---

## 架构对比

### 整体流程

```
World Models:
┌─────────────────────────────────────────────────────────┐
│  Stage 1        Stage 2           Stage 3              │
│  ────────       ────────          ────────             │
│  收集数据  →    VAE + MDN-RNN  →  CMA-ES 优化          │
│  (随机策略)     (分开训练)        (无梯度，梦中)        │
└─────────────────────────────────────────────────────────┘
     ↓                ↓                  ↓
  10000 rollouts   固定后不变        线性 Controller

Dreamer:
┌─────────────────────────────────────────────────────────┐
│              持续交替训练                                │
│  ┌────────────────┐    ┌────────────────┐              │
│  │  World Model   │ ←→ │  Actor-Critic  │              │
│  │  (RSSM)        │    │  (神经网络)     │              │
│  └────────────────┘    └────────────────┘              │
│         ↑                      ↓                       │
│         └──── 真实环境交互 ←────┘                       │
└─────────────────────────────────────────────────────────┘
```

### 动态模型对比

```
MDN-RNN (World Models):
────────────────────────
         a_{t-1}
            ↓
  z_{t-1} → LSTM → MDN → p(z_t)

  - 纯随机传递：所有信息必须通过 z
  - 输出：高斯混合分布 (5 个 Gaussian)
  - 问题：长期记忆困难，信息瓶颈


RSSM (Dreamer):
────────────────────────
         a_{t-1}
            ↓
  h_{t-1} → GRU → h_t     (确定性路径)
    ↓              ↓
  z_{t-1}        z_t      (随机性路径)

  - 双路径：h 保持记忆，z 建模随机性
  - 先验 p(z|h) + 后验 q(z|h,o)
  - 优势：长期记忆 + 训练稳定
```

### 状态表示对比

| 方面 | World Models | Dreamer V1 | DreamerV2/V3 |
|:---|:---|:---|:---|
| 潜在变量类型 | 连续 (32-dim) | 连续 (30-dim) | 离散 (32×32) |
| 分布 | 高斯混合 (MDN) | 高斯 | 分类 (Categorical) |
| 采样方式 | 从混合分布采样 | 重参数化 | Straight-Through |
| 确定性状态 | 无 | 有 (h) | 有 (h) |

---

## 策略优化对比

### World Models: CMA-ES (无梯度)

```python
# 进化策略：只看最终奖励
for generation in range(300):
    # 1. 采样 64 个控制器参数
    population = cma.ask()  # 64 组参数

    # 2. 每个控制器在梦中跑 16 条轨迹
    fitness = []
    for params in population:
        rewards = [dream_rollout(params) for _ in range(16)]
        fitness.append(mean(rewards))

    # 3. 根据 fitness 更新分布
    cma.tell(population, fitness)

# 特点：
# - 无梯度，只需要 fitness 值
# - 可以跑很长轨迹 (1000 步)
# - 中间误差被平均掉
# - 但需要大量采样，效率低
```

### Dreamer: 策略梯度 (有梯度)

```python
# Actor-Critic：精确梯度更新
for step in training:
    # 1. 从真实状态出发，想象 15 步
    states, rewards = imagine_trajectory(horizon=15)

    # 2. 用 Critic 估计未来价值
    values = critic(states)
    returns = compute_lambda_returns(rewards, values, λ=0.95)

    # 3. 策略梯度更新
    actor_loss = -returns.mean()  # 最大化回报
    actor_loss.backward()  # 梯度反向传播！

# 特点：
# - 有梯度，每步都有精确改进方向
# - 只需短视野 (15 步)
# - Critic 估计长期价值
# - 高效但梯度误差会累积
```

### 为什么视野长度不同？

```
           误差累积方式

CMA-ES:    ε₁ + ε₂ + ... + ε₁₀₀₀ → 平均误差（线性累积）
           可以容忍，因为只看总和

策略梯度:   ε₁ × ε₂ × ... × ε₁₅ → 梯度误差（可能指数累积）
           必须控制步数，否则梯度爆炸/消失
```

---

## 训练流程对比

### World Models: 三阶段流水线

```
阶段 1: 数据收集 (几小时)
────────────────────────
随机策略 → 环境 → 收集 10000 条轨迹
完成后固定，不再更新

阶段 2: 模型训练 (几小时)
────────────────────────
VAE 训练 (10 epochs) → 固定
MDN-RNN 训练 (20 epochs) → 固定
完成后不再更新

阶段 3: 控制器优化 (几小时)
────────────────────────
在梦中用 CMA-ES 进化
300 代 × 64 个体 × 16 rollouts

总时间：~10 小时 (CarRacing)
特点：各阶段独立，可并行开发
```

### Dreamer: 持续交替训练

```
while not done:
    # 1. 真实环境交互
    action = actor(state)
    next_state, reward = env.step(action)
    replay_buffer.add(state, action, reward, next_state)

    # 2. 世界模型更新 (每步或每 N 步)
    batch = replay_buffer.sample()
    world_model.train(batch)

    # 3. 想象 + Actor-Critic 更新
    imagined_trajectories = world_model.imagine(batch)
    actor.train(imagined_trajectories)
    critic.train(imagined_trajectories)

总时间：根据任务复杂度
特点：三个组件持续联合优化
```

---

## 关键技术细节对比

### 1. KL 散度处理

```
World Models:
  VAE 标准 KL：D_KL(q(z|x) || N(0,1))
  先验是固定的标准正态分布
  容易后验坍缩

Dreamer V1:
  D_KL(posterior || prior)
  先验是学习的 p(z|h)
  但还是单向 KL

DreamerV2/V3:
  KL Balancing:
  L = 0.8 × D_KL(sg(q) || p)   # 让先验更强
    + 0.2 × D_KL(q || sg(p))   # 让后验规整

  + Free Bits: max(KL, 1.0)    # 防止过度正则化
```

### 2. 价值估计

```
World Models:
  无价值估计
  必须跑完整轨迹才能评估

Dreamer:
  Critic 网络 V(s) 估计未来累积奖励

  λ-Returns 平衡：
  - λ=0: 完全信任 Critic (高偏差)
  - λ=1: 完全用真实奖励 (高方差)
  - λ=0.95: 短期真实奖励 + 长期 Critic 估计
```

### 3. 奖励尺度处理

```
World Models:
  任务特定，需要调参

DreamerV3:
  symlog 变换：symlog(x) = sign(x) × log(|x| + 1)

  效果：
  - symlog(1) = 0.69
  - symlog(100) = 4.62
  - symlog(10000) = 9.21

  不同尺度任务压缩到相似范围
```

---

## 性能对比

### 样本效率

| 任务 | World Models | Dreamer V1 | DreamerV2 | DreamerV3 |
|:---|:---|:---|:---|:---|
| CarRacing | ~900 分 | - | - | - |
| Atari (200M steps) | 未测试 | 115% 人类 | 200% 人类 | SOTA |
| DMControl | 未测试 | SOTA | SOTA | SOTA |
| Minecraft 钻石 | 未测试 | 失败 | 失败 | **首次成功** |

### 计算效率

```
World Models (CarRacing):
  数据收集: 10000 rollouts × 1000 steps = 10M frames
  训练时间: ~10 小时 (单 GPU)
  参数量: VAE + RNN + Controller ≈ 几 MB

DreamerV3:
  训练: 可在 100k-200M 步内收敛
  参数量: ~100M (取决于配置)
  更高效但需要更多计算资源
```

---

## 优缺点总结

### World Models

**优点：**
- 模块化清晰，易于理解和调试
- 各阶段可独立开发
- CMA-ES 对奖励函数变化鲁棒
- 可解释性强（可以可视化梦境）

**缺点：**
- 分阶段训练，无法联合优化
- 线性控制器表达能力有限
- CMA-ES 扩展性差（参数多时效率低）
- 世界模型固定后无法适应新情况

### Dreamer

**优点：**
- 端到端联合优化
- 神经网络策略，表达能力强
- 策略梯度高效
- 持续学习，适应环境变化

**缺点：**
- 训练过程复杂，调试困难
- 梯度误差累积限制想象视野
- 对超参数敏感（V3 之前）
- 黑盒程度更高

---

## 演进路线图

```
2018: World Models
      │
      │ 问题：分阶段训练、线性控制器
      ▼
2019: PlaNet
      │ 改进：RSSM、CEM 规划
      │ 问题：每步都要规划，太慢
      ▼
2020: Dreamer V1
      │ 改进：Actor-Critic、想象中训练
      │ 问题：连续潜在空间不稳定
      ▼
2021: DreamerV2
      │ 改进：离散潜在空间、KL Balancing
      │ 问题：需要针对任务调参
      ▼
2023: DreamerV3
      │ 改进：symlog、固定超参数
      │ 成就：首次解决 Minecraft 钻石任务
      ▼
未来: ???
      - 更统一的架构？
      - 目标导向的世界模型？
      - 与大语言模型结合？
```

---

## 核心洞察

### 1. 确定性 vs 随机性的分离

```
World Models 的教训：纯随机传递信息会丢失
RSSM 的解决方案：h 记忆 + z 随机

这是一个普适原则：
- Transformer 的残差连接也是"确定性路径"
- 很多序列模型都有类似设计
```

### 2. 短视野 + 价值估计 > 长视野

```
有了 Critic，不需要想象太远
V(s) 压缩了"未来所有信息"

这是 TD 学习的核心思想：
用估计值 bootstrap，减少方差
```

### 3. 离散表示更稳定

```
连续空间容易"漂移"和"坍缩"
离散空间有明确的边界

现实世界很多概念本就是离散的
```

### 4. 三阶段割裂是待解决的问题

```
观察 → 建模 → 决策 分离

理想情况：世界模型应该直接为决策服务
不是"重建准确"，而是"决策有用"

这是未来研究的重要方向
```

---

## 实践建议

### 何时用 World Models 风格？

- 简单任务、快速原型
- 需要可解释性
- 奖励函数可能变化
- 计算资源有限

### 何时用 Dreamer 风格？

- 复杂任务、追求性能
- 有足够计算资源
- 需要持续学习/适应
- 追求样本效率

### 代码资源

```
World Models:
- 原始实现: https://github.com/hardmaru/WorldModelsExperiments
- 本项目复现: world_models/experiments/3_car_racing_world_model.py

Dreamer:
- 官方 DreamerV3: https://github.com/danijar/dreamerv3
- PyTorch 复现: https://github.com/NM512/dreamerv3-torch
```

---

## 参考文献

1. Ha & Schmidhuber. "World Models" (2018)
2. Hafner et al. "Learning Latent Dynamics for Planning from Pixels" (PlaNet, 2019)
3. Hafner et al. "Dream to Control" (Dreamer, 2020)
4. Hafner et al. "Mastering Atari with Discrete World Models" (DreamerV2, 2021)
5. Hafner et al. "Mastering Diverse Domains through World Models" (DreamerV3, 2023)

---

> **下一步**: 前沿探索 - Genie, JEPA 与新一代世界模型 → `12_future_world_models.md`
