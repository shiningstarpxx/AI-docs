# Diffusion World Models 深度解析

## 1. 引言：为什么用扩散模型做世界模型？

### 1.1 回顾：现有世界模型的建模方式

| 方法 | 代表工作 | 建模方式 | 优势 | 局限 |
|:---|:---|:---|:---|:---|
| **VAE-based** | World Models, Dreamer | 连续高斯潜在空间 | 训练稳定 | 可能模糊 |
| **Discrete** | DreamerV2/V3 | 离散分类潜在变量 | 清晰边界 | 信息瓶颈 |
| **Autoregressive** | Genie | Token 序列预测 | 灵活 | 计算开销大 |
| **Diffusion** | DIAMOND, UniSim | 迭代去噪 | 高质量生成 | 推理较慢 |

### 1.2 扩散模型的独特优势

```
传统世界模型 (VAE/Discrete):
  单次前向传播 → 直接输出预测
  优点：快
  缺点：难以建模复杂多模态分布

扩散世界模型:
  多步迭代去噪 → 逐步精炼预测
  优点：可以建模任意复杂分布
  缺点：需要多次前向传播
```

**核心洞察**：
- 扩散模型不假设特定的输出分布形式
- 通过迭代过程逐步"雕刻"出复杂分布
- 天然适合多模态未来预测（T字路口问题）

---

## 2. 扩散模型基础回顾

### 2.1 DDPM 核心机制

#### 前向过程（加噪）

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

给定 $x_0$，可以直接采样任意时刻：
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

其中 $\bar{\alpha}_t = \prod_{i=1}^{t} (1 - \beta_i)$

```
x_0 (原始图像)
  │  添加少量噪声
  ▼
x_1
  │  继续加噪
  ▼
x_2
  ...
  │
  ▼
x_T ≈ N(0, I)  (纯噪声)
```

#### 反向过程（去噪）

学习预测噪声 $\epsilon_\theta(x_t, t)$：
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

其中：
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

```
x_T (纯噪声)
  │  预测噪声 ε_θ(x_T, T)
  │  去除噪声
  ▼
x_{T-1}
  │  预测噪声 ε_θ(x_{T-1}, T-1)
  ▼
  ...
  │
  ▼
x_0 (重建图像)
```

#### 训练目标

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**直觉**：训练网络预测"加入的噪声是什么"

### 2.2 Score Function 视角

Score Function 定义：
$$s(x) = \nabla_x \log p(x)$$

扩散模型本质是在学习 score function：
$$\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t} \log p(x_t)$$

**几何直觉**：
- Score function 指向数据分布的高概率区域
- 去噪过程就是沿着 score 的方向"爬坡"

### 2.3 Classifier-Free Guidance (CFG)

条件生成时增强条件的影响：

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing))$$

- $c$: 条件（如动作）
- $w$: guidance scale（通常 1.5-7.5）
- $\varnothing$: 无条件（训练时随机 drop condition）

**对世界模型的意义**：可以控制动作条件的强度

---

## 3. Diffusion 用于世界模型

### 3.1 Action-Conditioned Diffusion

#### 基本框架

```
输入: 当前状态 s_t, 动作 a_t
      ↓
      [扩散过程]
      ↓
输出: 下一状态 s_{t+1} 的分布
```

#### 条件注入方式

**方式 1：Concatenation**
```python
# 将动作 concat 到输入
input = torch.cat([x_t, action_embedding], dim=-1)
noise_pred = model(input, t)
```

**方式 2：Cross-Attention**
```python
# 动作作为 cross-attention 的 key/value
noise_pred = model(x_t, t, context=action_embedding)
```

**方式 3：AdaLN (Adaptive Layer Norm)**
```python
# 用动作调制 layer norm 的 scale 和 shift
scale, shift = action_mlp(action)
x = scale * layer_norm(x) + shift
```

### 3.2 多步预测的挑战

#### 问题：误差累积

```
单步预测: s_0 → s_1   (误差 ε_1)
          s_1 → s_2   (误差 ε_2，但 s_1 已有误差)
          ...
          s_n 的误差 ≈ 累积

长轨迹预测 vs 单帧预测的 trade-off
```

#### 解决方案

**方案 1：自回归展开（最常见）**
```
s_0 →[去噪]→ s_1 →[去噪]→ s_2 → ...

每一步都完整运行扩散过程
慢但简单
```

**方案 2：直接预测多帧**
```
输入: s_0, [a_0, a_1, ..., a_k]
输出: [s_1, s_2, ..., s_{k+1}]

一次扩散过程预测整个序列
快但难以训练
```

**方案 3：Latent Diffusion**
```
s_t → Encoder → z_t → [扩散] → z_{t+1} → Decoder → s_{t+1}

在压缩的潜在空间做扩散
平衡速度和质量
```

### 3.3 与 RL 的结合方式

#### 方式 1：作为环境模拟器

```
真实环境交互 → 训练扩散世界模型
                    ↓
              在世界模型中 rollout
                    ↓
              训练 RL agent (PPO, SAC, etc.)
```

#### 方式 2：作为 Planner

```
当前状态 s_0
    ↓
扩散模型采样多条可能轨迹
    ↓
评估每条轨迹的累积奖励
    ↓
选择最优轨迹的第一个动作
```

---

## 4. 代表工作

### 4.1 DIAMOND (2024)

**Diffusion for World Modeling: Visual Details Matter in Atari**
*Alonso, Jelley, Storkey, Micheli, Pearce, Kanervisto, Fleuret (NeurIPS 2024)*

#### 核心问题

```
离散潜在空间的局限 (DreamerV2/V3):
  - 压缩成离散 token 会丢失信息
  - 某些游戏中，小细节（如交通灯、远处行人）对决策很重要
  - 增加离散 token 数量 → 计算成本剧增

DIAMOND 的解决方案:
  - 直接在像素空间做扩散
  - 保留所有视觉细节
  - 利用扩散模型的高质量生成能力
```

#### 架构：时间维度的巧妙设计

```
┌─────────────────────────────────────────────────────────┐
│  DIAMOND: 双时间轴设计                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  环境时间 t (横轴):                                     │
│    x_{t-1} → x_t → x_{t+1} → ...                       │
│                                                         │
│  扩散时间 τ (纵轴):                                     │
│    x^T_t (噪声) → x^{T-1}_t → ... → x^0_t (干净)       │
│                                                         │
│  想象过程:                                              │
│    1. 给定历史 x^0_{<t} 和动作 a_{<t}                  │
│    2. 从噪声 x^T_t 开始                                 │
│    3. 调用 D_θ 逐步去噪: x^τ_t → x^{τ-1}_t            │
│    4. 得到干净的下一帧 x^0_t                            │
│    5. 策略 π_φ 选择动作 a_t                            │
│    6. 自回归地继续到 t+1                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 关键技术细节

**1. Score-based Diffusion**

```
前向过程 (加噪):
  dx = f(x, τ)dτ + g(τ)dw

反向过程 (去噪):
  dx = [f(x, τ) - g(τ)² ∇_x log p_τ(x)]dτ + g(τ)dw̄

核心: 学习 score function ∇_x log p_τ(x)
用单个网络 S_θ(x, τ) 估计所有时刻的 score
```

**2. 条件生成**

```
条件: 历史帧 x_{<t}, 动作序列 a_{<t}
目标: 生成 p(x_t | x_{<t}, a_{<t})

通过 cross-attention 或 concatenation 注入条件
```

**3. 训练循环**

```
while training:
    1. 在真实环境收集数据 (agent 交互)
    2. 训练扩散世界模型 (所有收集的数据)
    3. 在想象中训练 RL agent
       - 从真实帧开始
       - 扩散模型生成未来帧
       - 计算奖励和价值
```

#### 为什么视觉细节重要？

```
案例: Atari Boxing

DreamerV3 (离散潜在):
  - 压缩后丢失对手拳头的精确位置
  - agent 无法准确判断躲避时机

DIAMOND (像素扩散):
  - 保留拳头的像素级位置
  - agent 可以更精确地躲避

结论: 在需要精确视觉信息的任务中，扩散模型优势明显
```

#### 额外应用: 交互式神经游戏引擎

```
DIAMOND 还在 Counter-Strike: Global Offensive 上训练:
  - 87 小时的静态游戏视频
  - 学习成可交互的神经游戏引擎
  - 玩家可以在生成的环境中"玩"游戏

这展示了扩散世界模型作为通用模拟器的潜力
```

#### 实验结果

| 方法 | Atari 100K | 说明 |
|:---|:---|:---|
| SimPLe | 0.44 | 早期世界模型 |
| DreamerV3 | 1.03 | 离散潜在空间 |
| **DIAMOND** | **1.46** | 扩散世界模型 |

**资源**: https://diamond-wm.github.io (代码、agent、可玩世界模型)

### 4.2 UniSim (2023)

**Learning Interactive Real-World Simulators**

#### 核心思想

- 从互联网视频学习通用世界模拟器
- 支持语言、动作等多种条件控制
- 可以模拟真实世界物理

#### 架构

```
┌─────────────────────────────────────────────────────┐
│  UniSim Architecture                                │
├─────────────────────────────────────────────────────┤
│  条件输入:                                          │
│  - 语言指令: "pick up the cup"                     │
│  - 低层动作: [dx, dy, gripper]                     │
│  - 初始帧: 视频的第一帧                            │
│        ↓                                            │
│  Video Diffusion Model (3D U-Net)                  │
│        ↓                                            │
│  输出: 未来视频帧序列                               │
└─────────────────────────────────────────────────────┘
```

#### 关键创新

1. **多级动作空间**
   - 高层：语言指令
   - 中层：导航目标
   - 低层：关节力矩

2. **从视频学习物理**
   - 不需要模拟器
   - 从真实视频中学习

3. **交互式生成**
   - 支持实时控制
   - 可以改变动作观察结果

### 4.3 其他相关工作

#### Diffusion Policy (2023)

```
不是世界模型，而是直接用扩散模型做策略

观测 → 扩散过程 → 动作序列

优点：
- 多模态动作分布
- 适合模仿学习
```

#### DWM (Diffusion World Model, 2023)

```
将扩散模型与 Dreamer 框架结合

RSSM 的 VAE → 扩散模型

在连续控制任务上表现良好
```

---

## 5. 对比分析

### 5.1 VAE-based vs Diffusion-based 世界模型

| 维度 | VAE-based (Dreamer) | Diffusion-based |
|:---|:---|:---|
| **分布假设** | 高斯 (或离散) | 无假设 |
| **生成质量** | 可能模糊 | 高质量、清晰 |
| **多模态** | 需要特殊设计 | 天然支持 |
| **推理速度** | 快 (单次前向) | 慢 (多步去噪) |
| **训练稳定性** | 需要调 KL 权重 | 相对稳定 |
| **内存占用** | 较低 | 较高 |

### 5.2 为什么 DIAMOND 能超越 DreamerV3？

```
DreamerV3 的设计选择:
  - 离散潜在空间 (32x32 categorical)
  - 信息瓶颈
  - 无法表达精细视觉细节

DIAMOND 的发现:
  - 在 Atari 中，某些游戏的像素细节很重要
  - 比如小球的精确位置
  - 扩散模型可以重建这些细节
```

### 5.3 计算效率 vs 生成质量 Trade-off

```
                    生成质量
                       ↑
                       │    ★ Diffusion (full)
                       │
                       │         ★ Diffusion (few-step)
                       │
                       │    ★ DreamerV3
                       │
                       │ ★ VAE (简单)
                       │
                       └────────────────────→ 推理速度

权衡策略:
1. 使用 Latent Diffusion 减少计算
2. 使用少步采样 (DDIM, Consistency Models)
3. 只在关键决策点使用扩散模型
```

---

## 6. 实践建议

### 6.1 何时选择 Diffusion World Model

**适合场景**：
- 需要高保真视觉预测（机器人操作、自动驾驶）
- 存在多模态未来（复杂决策点）
- 计算资源充足
- 离线 RL 场景（可以慢慢生成数据）

**不适合场景**：
- 实时决策要求高
- 简单低维状态空间
- 计算资源受限

### 6.2 加速推理的技巧

1. **少步采样**：DDIM, DPM-Solver
2. **Consistency Models**：一步生成
3. **Latent Diffusion**：在压缩空间操作
4. **模型蒸馏**：将扩散模型蒸馏为快速模型

---

## 7. 开放问题与未来方向

### 7.1 开放问题

1. **推理速度**：如何让扩散世界模型足够快用于实时控制？
2. **长期预测**：如何避免多步预测的误差累积？
3. **与 RL 的深度集成**：如何让扩散模型和策略学习联合优化？

### 7.2 未来方向

1. **Consistency World Models**：用 Consistency Models 加速
2. **3D Diffusion**：直接在 3D 空间做世界模型
3. **多模态条件**：语言 + 动作 + 目标 统一条件
4. **Foundation World Models**：类似 GPT-4 的通用世界模型

---

## 8. 参考资料

### 论文

- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
- **DIAMOND**: Alonso et al., "Diffusion for World Modeling: Visual Details Matter in Atari", 2024
- **UniSim**: Yang et al., "Learning Interactive Real-World Simulators", 2023
- **Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", 2023

### 代码

- DIAMOND: https://github.com/eloialonso/diamond
- Diffusion Policy: https://github.com/real-stanford/diffusion_policy

---

## 9. 总结

### 核心要点

1. **扩散模型的优势**：无分布假设、高质量生成、天然多模态
2. **关键挑战**：推理速度、长期预测、与 RL 集成
3. **DIAMOND 的贡献**：证明视觉细节在 RL 中很重要，扩散模型可以捕捉

### 一句话总结

> **扩散世界模型用迭代去噪换取更精确的未来预测，在视觉细节重要的任务上表现出色。**
