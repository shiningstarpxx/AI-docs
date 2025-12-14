# 世界模型技术演进：从想象到理解
> 聚焦 2018-2024 年世界模型的关键技术突破

本文档基于历史演进视角，梳理世界模型（World Models）从开创性探索到实用化落地的技术里程碑。每个阶段都回答：**解决了什么问题？如何解决的？**

---

## 目录

- [0. 为什么需要世界模型？](#0-为什么需要世界模型)
- [1. World Models (2018): 开创性的"梦境学习"](#1-world-models-2018-开创性的梦境学习)
- [2. PlaNet (2019): 从想象到规划](#2-planet-2019-从想象到规划)
- [3. Dreamer (2020-2023): 从规划到策略学习](#3-dreamer-2020-2023-从规划到策略学习)
- [4. Genie (2024): 从视频学习可交互世界](#4-genie-2024-从视频学习可交互世界)
- [5. 技术对比与未来展望](#5-技术对比与未来展望)

---

## 0. 为什么需要世界模型？

### 0.1 强化学习的样本效率困境

**问题核心**：传统 Model-Free RL 需要海量环境交互

```ascii
任务对比：

监督学习 (ImageNet分类):
数据：1.2M 图像 (离线，可并行)
训练：数小时
样本效率：高 ✓

强化学习 (Atari DQN):
数据：200M 帧 (在线，串行) ≈ 833小时游戏
训练：10天
样本效率：低 ✗

真实机器人：
数据：数月持续交互
成本：硬件损耗 + 安全风险
样本效率：极低 ✗✗
```

**形式化分析**：

学习 Q 函数 $Q(s,a)$ 需要多次访问状态-动作对：
$$
\text{样本需求} \propto |\mathcal{S}| \times |\mathcal{A}| \times H
$$

Atari Breakout 示例：
- 状态空间：$\sim 10^{60}$ (像素组合)
- DQN 收敛：200M frames
- 人类学习：10 分钟 $\rightarrow$ **样本效率差距 100×+**

### 0.2 人类的关键优势：心智模拟

```ascii
人类学习过程：

1. 观察 2分钟 → 理解游戏规则 (建立世界模型)
2. 心智模拟   → "如果我这样操作会怎样？"
3. 快速试错   → 在想象中规划
4. 实际执行   → 验证并修正模型

关键：大部分学习发生在"想象"中，而非真实试错！
```

**世界模型的承诺**：
$$
\begin{align}
\text{真实环境：} \quad & s_{t+1}, r_t \sim p(s',r | s_t, a_t) \quad \text{(昂贵)} \\
\text{世界模型：} \quad & \hat{s}_{t+1}, \hat{r}_t \sim \hat{p}(s',r | s_t, a_t) \quad \text{(免费)}
\end{align}
$$

### 0.3 核心挑战

虽然理论优势明显，但实际面临：

| 挑战 | 具体问题 | 后果 |
|:---|:---|:---|
| **模型误差累积** | 单步误差 $\epsilon$ → H步误差 $H \cdot \epsilon$ | 长期规划失效 |
| **高维观测压缩** | 64×64×3 图像 → 低维表征 | 信息丢失 |
| **随机性建模** | 环境内在不确定性 | 预测模糊 |
| **训练稳定性** | 高方差、过拟合 | 泛化能力差 |

**这些问题的解决，正是 2018-2024 年技术演进的核心主线。**

---

## 1. World Models (2018): 开创性的"梦境学习"

**技术贡献**：V-M-C 解耦架构 + 梦境训练

### 1.1 为什么需要这个贡献 (Why)

**问题 1：端到端训练的复杂度爆炸**

传统端到端 RL（如 PPO、A3C）：

```ascii
[像素] → [大型CNN] → [策略网络] → [动作]
  ↓          ↓            ↓           ↓
12288维   数百万参数   高方差    训练不稳定

训练时间：2-3天
样本需求：100M+ frames
收敛稳定性：差 (高方差)
```

**问题 2：探索效率低**

- 需要在真实环境中大量试错
- CarRacing-v0：PPO 需要 100M 环境步骤 ≈ 50 小时游戏时间
- 能否在"想象"中训练？

### 1.2 解决方案 (How)

**核心思想：模块化解耦 + 梦境学习**

```ascii
V-M-C 三模块架构：

┌─────────────────────────────────────┐
│ V (Vision) - VAE                    │
│ 目标：64×64×3 → z∈R³²  (压缩385×)   │
│ 训练：无监督重构                     │
└─────────────────────────────────────┘
              ↓ z_t
┌─────────────────────────────────────┐
│ M (Memory) - MDN-RNN                │
│ 目标：(z_t, a_t) → z_{t+1}分布      │
│ 训练：监督学习 (预测下一帧)          │
│ 关键：MDN 建模多模态不确定性         │
└─────────────────────────────────────┘
              ↓ h_t (隐藏状态)
┌─────────────────────────────────────┐
│ C (Controller) - 线性策略           │
│ 目标：(z_t, h_t) → a_t              │
│ 训练：在V+M构建的"梦境"中训练 ⭐     │
│ 参数：仅867个 (极简)                │
│ 方法：CMA-ES 进化算法               │
└─────────────────────────────────────┘
```

### 1.3 关键技术细节

**A. MDN (Mixture Density Network) 的必要性**

为什么不用简单的确定性预测？

```ascii
场景：赛车前方有障碍物

单高斯预测 (失败):           MDN 多模态 (成功):
z_{t+1} = μ (固定)          z_{t+1} ~ π₁N(μ₁,σ₁) + π₂N(μ₂,σ₂)
   ↓                            ↓
预测"左转+右转"的平均值        预测两种可能: 左转 OR 右转
→ 模糊的中间状态 (撞墙!)      → 清晰的多种可能 ✓
```

**B. 梦境训练的优势**

```python
# 传统 RL
for step in range(100_000_000):  # 100M 真实步骤
    action = policy(obs)
    obs', reward = env.step(action)  # 昂贵！
    train_policy(obs, action, reward)

# World Models 梦境训练
# 阶段1: 收集少量数据
collect_10k_trajectories()  # 仅10k轨迹

# 阶段2: 训练 V+M
train_vae()      # 离线训练
train_mdn_rnn()  # 离线训练

# 阶段3: 在梦境中训练策略
for generation in range(100):
    for individual in population:
        reward = evaluate_in_dream(V, M, individual)  # 免费！
    population = evolve(population)

样本效率提升：~100× ⭐
```

### 1.4 实验结果

**CarRacing-v0 性能对比**

| 方法 | 环境步骤 | 训练时间 | 得分 | 样本效率 |
|:---|:---|:---|:---|:---|
| PPO (baseline) | 100M | 2-3天 | 850 ± 45 | 1× |
| A3C | 80M | 2天 | 820 ± 60 | 1.25× |
| **World Models** | **10k轨迹** | **13小时** | **906 ± 21** | **~100×** ⭐ |

**消融实验：模块必要性**

| 配置 | 得分 | 说明 |
|:---|:---|:---|
| 完整 V-M-C | **906 ± 21** | Baseline |
| 移除 M (V→C直连) | 450 ± 67 | 无时序建模，反应式策略 |
| 单高斯 RNN (替代MDN) | 702 ± 89 | 无法捕获多模态 |
| 大型Controller (1000参数) | 890 ± 34 | 过拟合到梦境 |

**关键洞察**：
- ✅ 解耦降低复杂度：V、M、C 独立优化
- ✅ 简单策略防过拟合：867 参数足够
- ✅ MDN 建模不确定性：多模态预测至关重要

---

## 2. PlaNet (2019): 从想象到规划

**技术贡献**：RSSM 动态模型 + 在线规划 (MPC)

### 2.1 为什么需要这个贡献 (Why)

**问题 1：World Models 的策略泛化问题**

```ascii
World Models 工作流：
1. 离线收集数据
2. 训练固定的 V+M
3. 在梦境中训练固定策略 C
4. 部署（无进一步学习）

局限：
• 策略过拟合到不完美的世界模型
• 无法在线适应环境变化
• 模型误差无法纠正
```

**问题 2：传统 RNN 的表达能力不足**

```python
# 确定性 RNN (如 World Models 的 LSTM)
h_t = f(h_{t-1}, a_t, z_t)  # 完全确定性

问题：无法显式建模环境的随机性

例子 - 抛硬币：
h_t 编码"即将抛硬币"
→ 下一状态应该是正面/反面（50%概率）
→ 但 RNN 只能输出固定的 h_{t+1}
→ 模型困惑，预测模糊
```

### 2.2 解决方案 (How)

**核心创新 1：RSSM (Recurrent State Space Model)**

$$
\begin{align}
\text{确定性路径：} \quad & h_t = f(h_{t-1}, s_{t-1}, a_{t-1}) \\
\text{随机路径：} \quad & s_t \sim p(s_t | h_t) \\
\text{观测模型：} \quad & o_t \sim p(o_t | h_t, s_t) \\
\text{奖励模型：} \quad & r_t \sim p(r_t | h_t, s_t)
\end{align}
$$

```ascii
RSSM 架构：

    [h_{t-1}, s_{t-1}, a_{t-1}]
              ↓
         ┌─────────┐
         │   RNN   │  h_t (确定性：历史信息)
         └─────────┘
              ↓
         ┌─────────┐
         │ Latent  │  s_t ~ N(μ,σ) (随机性：不确定性)
         └─────────┘
              ↓
    ┌──────────┴──────────┐
    ↓                     ↓
[Observation]         [Reward]
  o_t ~ p(·|h,s)       r_t ~ p(·|h,s)

优势：
✓ h_t 捕获长期依赖
✓ s_t 建模随机性
✓ 分离可预测 vs 不可预测
```

**核心创新 2：在线规划 (MPC)**

不训练固定策略，而是每步重新规划：

```python
def select_action_with_mpc(model, state, horizon=12):
    """
    Model Predictive Control (MPC)
    每步都用 CEM 优化未来 H 步动作序列
    """
    # 初始化动作分布
    mean = zeros(horizon, action_dim)
    std = ones(horizon, action_dim)
    
    # CEM 优化（迭代 10 轮）
    for iteration in range(10):
        # 采样 1000 条动作序列
        action_seqs = sample(mean, std, n=1000)
        
        # 用模型想象每条序列的结果
        returns = []
        for actions in action_seqs:
            total_return = imagine_rollout(model, state, actions)
            returns.append(total_return)
        
        # 选择精英序列，更新分布
        elite_actions = top_10_percent(action_seqs, returns)
        mean = elite_actions.mean()
        std = elite_actions.std()
    
    # 执行第一个动作，下一步重新规划
    return mean[0]
```

### 2.3 形式化表达与数据

**A. RSSM vs 传统 RNN 对比**

| 维度 | 传统 RNN | RSSM (PlaNet) |
|:---|:---|:---|
| 状态表示 | $h_t = f(h_{t-1}, z_t, a_t)$ | $h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$ <br> $s_t \sim p(s_t \| h_t)$ |
| 随机性建模 | ❌ 隐式（通过多个时间步） | ✅ 显式（潜在变量 $s_t$） |
| 多模态预测 | MDN 输出层 | 潜在空间本身多模态 |
| 训练稳定性 | 中等 | 更高（KL 正则化） |

**B. DMControl Suite 实验结果**

| 任务 | PlaNet (5K steps) | SAC (5K steps) | SAC (500K steps) | 样本效率提升 |
|:---|:---|:---|:---|:---|
| cheetah-run | **680** | 120 | 920 | **5.7×** at 5K |
| walker-walk | **920** | 200 | 940 | **4.6×** at 5K |
| finger-spin | **950** | 600 | 980 | **1.6×** at 5K |

**关键洞察**：
- PlaNet 在 5K 步骤内接近 SAC 的 500K 步骤性能
- 样本效率提升 2-6 倍
- 但：每步决策需要 CEM 优化（120k 次模型前向传播）

**C. 规划视野 (Horizon) 的影响**

```ascii
Planning Horizon H 对性能的影响 (walker-walk):

H=1:    400 分  [无长期规划]
H=3:    650 分
H=6:    820 分
H=12:   920 分  ← 最优 ⭐
H=20:   900 分  [误差累积]
H=30:   750 分  [模型不准，规划失效]

教训：太短无效，太长误差累积
```

---

## 3. Dreamer (2020-2023): 从规划到策略学习

**技术贡献**：在潜在空间中的 Actor-Critic

### 3.1 为什么需要这个贡献 (Why)

**问题：规划的计算瓶颈**

PlaNet 每步决策需要 CEM 优化：

```python
CEM 计算成本：
- 种群：1000 个动作序列
- 迭代：10 轮
- 视野：12 步

每步前向传播次数 = 1000 × 10 × 12 = 120,000 次 ⚠️

后果：
• 推理慢（~1秒/动作）
• 无法实时控制
• 无法扩展到长视野
```

**问题 2：离线规划 vs 在线学习**

```ascii
PlaNet (规划):               理想方案 (策略学习):
每步都优化 → 计算昂贵        训练一次 → 部署快速
无法积累经验 → 重复计算       策略改进 → 持续学习
固定模型 → 无法适应          在线学习 → 持续改进
```

### 3.2 解决方案 (How)

**核心思想：在想象中学习 Actor-Critic**

```ascii
Dreamer 工作流：

阶段 A: 环境交互
  真实环境 → 经验池 (持续收集)

阶段 B: 世界模型学习
  经验池 → 训练 RSSM (重构观测、预测奖励)

阶段 C: 行为学习（关键创新 ⭐）
  1. 从经验池采样起始状态 s₀
  2. 用 Actor 在想象中展开 H=15 步：
     a_t ~ π(·|s_t)
     s_{t+1} ~ RSSM(·|s_t, a_t)
     r_t ~ p_reward(s_t)
  3. 在想象轨迹上训练：
     • Critic: V(s) = 𝔼[Σ γᵗ r_t]
     • Actor: ∇_θ 𝔼[V(s)]
```

**关键优势**：

| 维度 | PlaNet (规划) | Dreamer (策略学习) |
|:---|:---|:---|
| 每步计算 | 120k 次前向传播 (CEM) | 1 次前向传播 (Actor) |
| 推理速度 | ~1秒/动作 | ~10ms/动作 (**100×**) |
| 长期规划 | H=12 (误差累积) | H=15+ (策略学习弥补) |
| 在线学习 | ❌ 固定模型 | ✅ 持续改进 |

### 3.3 Dreamer 系列演进

#### DreamerV1 (2020): 基础架构

```python
# 连续潜在变量
s_t ~ N(μ, σ)  # 30维高斯分布

# 策略梯度 + λ-return
L_actor = -𝔼[Σ λᵗ V(s_t)]
```

#### DreamerV2 (2021): 离散潜在空间

**核心改进**：
$$
\begin{align}
\text{V1 (连续):} \quad & s_t \sim \mathcal{N}(\mu, \sigma) \in \mathbb{R}^{30} \\
\text{V2 (离散):} \quad & s_t = [s_t^1, s_t^2, ..., s_t^{32}] \\
& \text{其中每个 } s_t^i \sim \text{Cat}(32) \\
& \text{总表征空间：} 32^{32} \approx 2^{160} \text{ (巨大！)}
\end{align}
$$

**为什么离散更好？**

```ascii
连续高斯 (V1):               离散类别 (V2):
表达能力：有限               表达能力：指数级
多模态：依赖混合高斯          多模态：天然支持
训练稳定性：Posterior collapse  训练稳定性：Free bits 防崩溃
```

**Atari 100K 基准结果**：

| 算法 | 人类归一化得分 | 训练数据 | 样本效率 |
|:---|:---|:---|:---|
| DQN | 0.52 | 200M frames | 1× |
| Rainbow | 0.71 | 200M frames | 1× |
| **DreamerV2** | **1.15** | 100k frames | **2000×** ⭐ |

**突破**：仅 100K frames (约 2 小时游戏) 超越人类水平 (1.0)！

#### DreamerV3 (2023): 通用世界模型

**目标**：单组超参数适用于所有任务

**关键技术**：
1. **Symlog 预测**：处理不同尺度奖励
   $$\text{symlog}(x) = \text{sign}(x) \cdot \log(|x| + 1)$$
   
2. **Free Bits**：防止 KL 崩溃
   $$\max(\text{KL}[q \| p], \beta)$$
   
3. **Layer Norm**：所有网络归一化

**通用性验证**：

| 任务类型 | 环境 | DreamerV3 得分 | 说明 |
|:---|:---|:---|:---|
| 离散动作 | Atari (55 games) | 1.22 (超人类) | ✓ |
| 连续控制 | DMC (20 tasks) | 980/1000 | ✓ |
| 3D 导航 | Minecraft | Diamond 15% | ✓ (极难) |
| 机器人 | Meta-World | 95% 成功率 | ✓ |

**突破**：首个真正的通用世界模型算法！无需针对任务调参 ⭐⭐⭐

### 3.4 消融实验：关键组件贡献

| 配置 | Atari 得分 | 说明 |
|:---|:---|:---|
| **DreamerV2 完整** | **1.15** | Baseline |
| 去除 Imagination（直接 RL） | 0.72 | 样本效率暴跌 |
| 连续潜在变量（V1） | 0.98 | 表达能力不足 |
| 去除 Critic | 0.85 | 高方差，不稳定 |
| 短想象视野 (H=5) | 0.92 | 长期规划差 |

---

## 4. Genie (2024): 从视频学习可交互世界

**技术贡献**：无动作标注的世界模型学习

### 4.1 为什么需要这个贡献 (Why)

**问题：动作标注的瓶颈**

前述方法（World Models、PlaNet、Dreamer）都需要：
```python
数据格式：(观测 o_t, 动作 a_t, 奖励 r_t)
                    ↑
                  必需！
```

**局限性**：
- 机器人：需要安装传感器记录动作
- 游戏：需要修改引擎导出按键
- 人类行为：无法获取"动作标注"

**机会**：互联网上有海量视频数据（YouTube、游戏直播），但**没有动作标注**！

```ascii
可用数据：
• YouTube: 800M 视频
• Twitch 游戏直播: 每天 15M 小时
• 自然视频: 无限

问题：能否仅从视频 [o_1, o_2, ..., o_T] 学习世界模型？
```

### 4.2 解决方案 (How)

**核心思想：潜在动作模型 (Latent Action Model)**

```ascii
Genie 架构：

输入：视频序列 [o_1, o_2, ..., o_T]  (无动作标注)

┌──────────────────────────────────────────┐
│ 模块 1: Video Tokenizer (ST)            │
│ o_t → z_t ∈ {0,...,8191}                │
│ (压缩到离散 token)                       │
└──────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────┐
│ 模块 2: Latent Action Model (LAM) ⭐     │
│ (z_t, z_{t+1}) → â_t ∈ {0,...,7}        │
│ 自监督学习：从帧对推断动作                │
└──────────────────────────────────────────┘
                ↓
┌──────────────────────────────────────────┐
│ 模块 3: Dynamics Model (Transformer)    │
│ (z_t, â_t) → z_{t+1}                    │
│ 预测下一帧                               │
└──────────────────────────────────────────┘

联合训练（闭环一致性）：
z_t → â_t → ẑ_{t+1} 应该接近 z_{t+1}
```

**关键创新：自监督动作推断**

```python
# 训练 LAM
for (z_t, z_{t+1}) in video:
    # 推断隐含动作
    a_hat = LAM(z_t, z_{t+1})
    
    # 用推断的动作预测下一帧
    z_pred = Dynamics(z_t, a_hat)
    
    # 闭环损失
    loss = ||z_pred - z_{t+1}||²

# 推理时（生成可玩游戏）
z = initial_frame
for step in range(1000):
    action = user_input()  # 用户输入动作！
    z = Dynamics(z, action)
    display(decode(z))
```

### 4.3 实验结果

**Coinrun 游戏生成**：

```ascii
训练：
• 数据：30K 小时游戏视频 (无动作标注)
• 模型：11B 参数

能力：
• 生成可玩游戏（接受动作输入）
• 物理一致性：角色跳跃、碰撞、重力
• 长时间一致性：保持 20+ 帧
• 动作可控性：8 个离散动作清晰分离

局限：
• 分辨率：160×90 (较低)
• 时间：~20 帧后逐渐模糊
• 物理：复杂交互偶尔不一致
```

**对比 Dreamer**：

| 维度 | Dreamer | Genie |
|:---|:---|:---|
| 输入数据 | (o, a, r) | 仅 o (视频) |
| 动作空间 | 预定义 | 自动发现 |
| 训练数据 | 1M frames | 30K hours |
| 应用场景 | 已知动作的 RL | 视频生成、模仿学习 |

---

## 5. 技术对比与未来展望

### 5.1 里程碑总结

| 模型 | 年份 | 核心痛点 | 关键贡献 | 样本效率提升 |
|:---|:---|:---|:---|:---|
| **World Models** | 2018 | 端到端训练复杂 | V-M-C 解耦 + 梦境学习 | **100×** |
| **PlaNet** | 2019 | 策略过拟合 | RSSM + 在线规划 (MPC) | **2-6×** |
| **Dreamer** | 2020 | 规划计算瓶颈 | 在想象中学习策略 | **2000×** |
| **DreamerV2** | 2021 | 表达能力不足 | 离散潜在空间 | 超人类 (Atari) |
| **DreamerV3** | 2023 | 任务特定调参 | 通用架构 + Symlog | 跨域泛化 |
| **Genie** | 2024 | 动作标注瓶颈 | 潜在动作模型 | 无监督学习 |

### 5.2 技术演进脉络

```ascii
2018 → 2019 → 2020 → 2021 → 2023 → 2024
  |      |      |      |      |      |
解耦   规划   策略   离散   通用   无监督
  ↓      ↓      ↓      ↓      ↓      ↓
梦境   MPC   Actor  表达力  泛化   视频
学习         Critic
```

### 5.3 关键技术对比

**A. 动态建模方式**

| 模型 | 状态表示 | 随机性建模 | 训练稳定性 |
|:---|:---|:---|:---|
| World Models | RNN + MDN | 输出层多模态 | 中等 |
| PlaNet/Dreamer | RSSM (h + s) | 潜在变量 | 高 (KL 正则) |
| DreamerV2/V3 | RSSM + 离散 | 分类分布 | 更高 |

**B. 策略优化方式**

| 模型 | 方法 | 每步计算 | 推理速度 | 在线学习 |
|:---|:---|:---|:---|:---|
| World Models | CMA-ES | 固定策略 | 快 | ❌ |
| PlaNet | CEM 规划 | 120k 前向 | 慢 (~1s) | ✅ (模型) |
| Dreamer | Actor-Critic | 1 次前向 | 快 (~10ms) | ✅ (全部) |

**C. 样本效率对比**

```ascii
达到相同性能所需环境步骤：

Model-Free (SAC):  ████████████████████ 500K steps
PlaNet:            ████ 100K steps (5×)
Dreamer:           █ 10K steps (50×)
DreamerV2:         ▌ 5K steps (100×)
```

### 5.4 未来方向

**1. 多模态融合**
```
视觉 + 语言 + 触觉 → 统一世界模型
应用：具身智能、机器人
```

**2. 长时间一致性**
```
当前：10-20 步
目标：1000+ 步 (分钟级视频)
挑战：误差累积、计算效率
```

**3. 物理理解**
```
超越像素：理解因果、物理定律
应用：科学发现、自动驾驶
```

**4. 大规模预训练**
```
类比 GPT: 互联网视频预训练
迁移学习：零样本/少样本适应
```

**5. 可解释性**
```
当前：黑盒潜在空间
目标：可解释的概念表征
应用：医疗诊断、自动驾驶安全
```

---

## 参考文献

### 核心论文

1. **World Models** - Ha & Schmidhuber (2018)  
   arXiv:1803.10122  
   [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

2. **PlaNet** - Hafner et al. (2019)  
   "Learning Latent Dynamics for Planning from Pixels"  
   arXiv:1811.04551

3. **Dreamer** - Hafner et al. (2020)  
   "Dream to Control: Learning Behaviors by Latent Imagination"  
   arXiv:1912.01603

4. **DreamerV2** - Hafner et al. (2021)  
   "Mastering Atari with Discrete World Models"  
   arXiv:2010.02193

5. **DreamerV3** - Hafner et al. (2023)  
   "Mastering Diverse Domains through World Models"  
   arXiv:2301.04104

6. **Genie** - Bruce et al. (2024)  
   "Genie: Generative Interactive Environments"  
   arXiv:2402.15391

### 相关综述

- "Model-Based Reinforcement Learning: A Survey" - Moerland et al. (2023)
- "Deep Learning for Video Prediction" - Oprea et al. (2020)

---

*最后更新: 2025-12-08*
