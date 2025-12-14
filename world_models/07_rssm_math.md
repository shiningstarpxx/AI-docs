# RSSM 数学原理详解

> Recurrent State-Space Model: Dreamer 系列的核心

## 目录

1. [状态空间模型回顾](#状态空间模型回顾)
2. [RSSM 的设计动机](#rssm-的设计动机)
3. [数学形式化](#数学形式化)
4. [变分推断与 ELBO](#变分推断与-elbo)
5. [离散 vs 连续潜在变量](#离散-vs-连续潜在变量)
6. [训练目标详解](#训练目标详解)
7. [实现细节](#实现细节)

---

## 状态空间模型回顾

### 经典状态空间模型 (SSM)

状态空间模型是描述动态系统的标准框架：

$$
\begin{aligned}
\text{状态转移:} \quad s_{t+1} &= f(s_t, a_t) + \epsilon_t \\
\text{观测生成:} \quad o_t &= g(s_t) + \delta_t
\end{aligned}
$$

其中：
- $s_t$: 隐藏状态（不可直接观测）
- $o_t$: 观测（可观测的传感器数据）
- $a_t$: 动作
- $\epsilon_t, \delta_t$: 噪声

### 概率表示

$$
\begin{aligned}
p(s_{t+1} | s_t, a_t) &\quad \text{状态转移分布} \\
p(o_t | s_t) &\quad \text{观测似然}
\end{aligned}
$$

### 经典方法的局限

| 方法 | 假设 | 局限性 |
|:---|:---|:---|
| 卡尔曼滤波 | 线性高斯 | 无法处理非线性系统 |
| 粒子滤波 | 非参数 | 高维空间效率低 |
| HMM | 离散状态 | 状态空间有限 |

**深度学习的解决方案**: 用神经网络参数化 $f$ 和 $g$

---

## RSSM 的设计动机

### 为什么需要确定性 + 随机性？

**World Models 的纯随机方法**：

```
z_t → LSTM → z_{t+1}
     纯随机传递
```

问题：
1. **信息瓶颈**: 所有信息必须通过随机变量 z 传递
2. **长期记忆困难**: 随机采样导致信息损失
3. **训练不稳定**: KL 散度难以平衡

**RSSM 的双路径设计**：

```
h_t → GRU → h_{t+1}     (确定性，无损传递)
 │            │
 ▼            ▼
z_t          z_{t+1}    (随机性，建模不确定性)
```

优势：
1. **确定性路径 h**: 像传统 RNN 一样无损传递信息
2. **随机性路径 z**: 捕获环境的随机性和不确定性
3. **分工明确**: h 负责记忆，z 负责建模变化

### 直觉理解

想象一个视频游戏：

```
确定性部分 h (可预测):
- 角色当前位置
- 已收集的物品
- 游戏进度

随机性部分 z (不可预测):
- 敌人是否出现
- 宝箱里有什么
- 环境随机事件
```

---

## 数学形式化

### RSSM 完整定义

设状态 $s_t = (h_t, z_t)$，其中：
- $h_t \in \mathbb{R}^H$: 确定性隐状态
- $z_t$: 随机潜在变量（连续或离散）

**状态转移模型**：

$$
\begin{aligned}
\text{确定性路径:} \quad h_t &= f_\theta(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{先验分布:} \quad z_t &\sim p_\theta(z_t | h_t) \\
\end{aligned}
$$

**观测模型**：

$$
\begin{aligned}
\text{后验分布:} \quad z_t &\sim q_\phi(z_t | h_t, o_t) \\
\text{观测解码:} \quad o_t &\sim p_\theta(o_t | h_t, z_t) \\
\text{奖励预测:} \quad r_t &\sim p_\theta(r_t | h_t, z_t) \\
\end{aligned}
$$

### 图模型表示

```
                 先验 p(z|h)
                     ↓
a_{t-1}         a_t           a_{t+1}
  ↓               ↓               ↓
┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
│h_0│ → │h_1│ → │h_2│ → │h_3│ → │h_4│  确定性路径
└─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
  │       │       │       │       │
  ↓       ↓       ↓       ↓       ↓
┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
│z_0│   │z_1│   │z_2│   │z_3│   │z_4│  随机路径
└─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
  │       │       │       │       │
  ↓       ↓       ↓       ↓       ↓
┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
│o_0│   │o_1│   │o_2│   │o_3│   │o_4│  观测
└───┘   └───┘   └───┘   └───┘   └───┘
                  ↑
             后验 q(z|h,o)
```

### 各分布的参数化

**1. 确定性路径 (GRU)**

$$h_t = \text{GRU}_\theta([z_{t-1}, a_{t-1}], h_{t-1})$$

```python
# PyTorch 实现
h_t = self.gru(
    torch.cat([z_prev.flatten(-2), action], dim=-1),
    h_prev
)
```

**2. 先验分布 (Prior)**

连续版本 (Dreamer V1)：
$$p_\theta(z_t | h_t) = \mathcal{N}(\mu_\theta(h_t), \sigma_\theta(h_t))$$

离散版本 (DreamerV2/V3)：
$$p_\theta(z_t | h_t) = \prod_{i=1}^{K} \text{Categorical}(\pi_\theta^i(h_t))$$

其中 $K=32$ 个独立的 32 类分类分布。

**3. 后验分布 (Posterior)**

$$q_\phi(z_t | h_t, o_t) = q_\phi(z_t | h_t, \text{Enc}_\phi(o_t))$$

后验利用了真实观测 $o_t$，因此比先验更准确。

---

## 变分推断与 ELBO

### 目标：最大化观测序列的似然

给定观测序列 $o_{1:T}$ 和动作序列 $a_{1:T}$，我们想最大化：

$$\log p_\theta(o_{1:T} | a_{1:T})$$

但这个积分难以计算（涉及对所有 $z_{1:T}$ 积分）。

### 变分下界 (ELBO)

引入近似后验 $q_\phi(z_{1:T} | o_{1:T}, a_{1:T})$：

$$
\log p_\theta(o_{1:T} | a_{1:T}) \geq \mathbb{E}_{q_\phi} \left[ \sum_{t=1}^{T} \log p_\theta(o_t | h_t, z_t) \right] - D_{KL}(q_\phi \| p_\theta)
$$

### RSSM 的分解

由于 RSSM 的马尔可夫结构，ELBO 可以按时间步分解：

$$
\mathcal{L} = \sum_{t=1}^{T} \left[
    \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(o_t | h_t, z_t)]}_{\text{重建项}}
    - \underbrace{D_{KL}(q_\phi(z_t | h_t, o_t) \| p_\theta(z_t | h_t))}_{\text{KL 项}}
\right]
$$

### 各项解释

**重建项**：
- 让模型能从 $(h_t, z_t)$ 重建观测 $o_t$
- 确保潜在状态包含足够信息

**KL 项**：
- 让后验 $q(z|h,o)$ 接近先验 $p(z|h)$
- 确保想象时（只用先验）与训练时（用后验）一致

```
训练时: 用后验 q(z|h,o)，有真实观测指导
想象时: 用先验 p(z|h)，无观测，纯预测

KL 项确保两者一致！
```

---

## 离散 vs 连续潜在变量

### 连续潜在变量 (Dreamer V1)

$$z_t \sim \mathcal{N}(\mu_\theta(h_t), \text{diag}(\sigma_\theta(h_t)^2))$$

**重参数化采样**：
$$z_t = \mu_\theta(h_t) + \sigma_\theta(h_t) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**KL 散度闭式解**：
$$D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### 离散潜在变量 (DreamerV2/V3)

$$z_t = [z_t^1, z_t^2, \ldots, z_t^{32}], \quad z_t^i \sim \text{Categorical}(\pi^i)$$

每个 $z_t^i$ 是 32 类的 one-hot 向量。

**Straight-Through Gradient**：

```python
def straight_through_categorical(logits):
    """
    前向：离散采样 (argmax)
    反向：连续梯度 (softmax)
    """
    probs = F.softmax(logits, dim=-1)

    # 前向：one-hot (不可微)
    indices = probs.argmax(dim=-1)
    one_hot = F.one_hot(indices, num_classes=probs.shape[-1]).float()

    # 反向：让梯度流过 probs
    z = one_hot + probs - probs.detach()

    return z
```

**离散 KL 散度**：

$$D_{KL}(q \| p) = \sum_{k=1}^{K} q_k \log \frac{q_k}{p_k}$$

### 为什么离散更好？

| 方面 | 连续 | 离散 |
|:---|:---|:---|
| 表达离散概念 | 困难 | 自然 |
| KL 平衡 | 容易后验坍缩 | 更稳定 |
| 信息容量 | 无界 | 有界 (log K) |
| 采样效率 | 需要重参数化 | 直接采样 |

---

## 训练目标详解

### 完整损失函数

DreamerV3 的世界模型损失：

$$
\mathcal{L}_{\text{world}} = \mathcal{L}_{\text{pred}} + \beta \cdot \mathcal{L}_{\text{dyn}} + \beta \cdot \mathcal{L}_{\text{rep}}
$$

其中：

**1. 预测损失 (Prediction Loss)**

$$\mathcal{L}_{\text{pred}} = -\mathbb{E}_{q_\phi}[\log p_\theta(o_t | s_t) + \log p_\theta(r_t | s_t) + \log p_\theta(c_t | s_t)]$$

- 重建观测 $o_t$
- 预测奖励 $r_t$
- 预测继续信号 $c_t$ (是否终止)

**2. 动态损失 (Dynamics Loss)**

$$\mathcal{L}_{\text{dyn}} = \max(D_{KL}[sg(q_\phi(z_t | h_t, o_t)) \| p_\theta(z_t | h_t)], \text{free})$$

- $sg()$: stop gradient
- 只更新先验网络 $p_\theta$
- `free`: free bits，防止过度正则化

**3. 表示损失 (Representation Loss)**

$$\mathcal{L}_{\text{rep}} = \max(D_{KL}[q_\phi(z_t | h_t, o_t) \| sg(p_\theta(z_t | h_t))], \text{free})$$

- 只更新后验网络 $q_\phi$ 和编码器
- 让后验向先验靠拢

### KL Balancing 解释

为什么要分开 $\mathcal{L}_{\text{dyn}}$ 和 $\mathcal{L}_{\text{rep}}$？

```
传统做法: D_KL(q || p) 同时更新 q 和 p
问题: 后验 q 可能"躲避"到先验 p 覆盖的区域，损失信息

DreamerV2/V3 的做法:
- L_dyn: 固定 q，让 p 去拟合 q → 先验变得更有信息
- L_rep: 固定 p，让 q 向 p 靠拢 → 后验更规整

通过调整比例 (通常 1:1)，平衡两个方向
```

### Free Bits

$$\mathcal{L}_{KL} = \max(D_{KL}, \text{free\_bits})$$

当 KL < free_bits 时，不惩罚 → 允许一定的后验自由度

```python
def free_bits_kl(kl, free_bits=1.0):
    """
    只惩罚超过 free_bits 的 KL
    """
    return torch.clamp(kl, min=free_bits)
```

---

## 实现细节

### 完整 RSSM 类

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, kl_divergence


class RSSM(nn.Module):
    """
    Recurrent State-Space Model

    状态 s_t = (h_t, z_t)
    - h_t: 确定性隐状态 (GRU)
    - z_t: 随机潜在变量 (离散分类)
    """

    def __init__(
        self,
        hidden_size: int = 512,
        state_size: int = 32,
        num_categories: int = 32,
        action_size: int = 6,
        embed_size: int = 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_categories = num_categories
        self.state_dim = state_size * num_categories  # 展平后的维度

        # 确定性路径: GRU
        self.gru = nn.GRUCell(self.state_dim + action_size, hidden_size)

        # 先验网络: h -> z 的 logits
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.state_dim),
        )

        # 后验网络: (h, embed) -> z 的 logits
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_size + embed_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.state_dim),
        )

    def initial_state(self, batch_size: int, device):
        """初始化状态"""
        return {
            'h': torch.zeros(batch_size, self.hidden_size, device=device),
            'z': torch.zeros(batch_size, self.state_size, self.num_categories, device=device),
        }

    def get_state_feature(self, state):
        """获取用于后续网络的状态特征 [h, z_flat]"""
        h, z = state['h'], state['z']
        z_flat = z.reshape(z.shape[0], -1)
        return torch.cat([h, z_flat], dim=-1)

    def imagine_step(self, state, action):
        """
        想象一步 (只用先验，不需要观测)
        用于 Actor-Critic 在想象中训练
        """
        h_prev = state['h']
        z_prev = state['z'].reshape(state['z'].shape[0], -1)

        # 确定性路径
        gru_input = torch.cat([z_prev, action], dim=-1)
        h = self.gru(gru_input, h_prev)

        # 先验采样
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.reshape(-1, self.state_size, self.num_categories)
        z = self._straight_through_categorical(prior_logits)

        return {
            'h': h,
            'z': z,
            'prior_logits': prior_logits,
        }

    def observe_step(self, state, action, embed):
        """
        观测一步 (用后验，需要编码后的观测)
        用于世界模型训练
        """
        h_prev = state['h']
        z_prev = state['z'].reshape(state['z'].shape[0], -1)

        # 确定性路径
        gru_input = torch.cat([z_prev, action], dim=-1)
        h = self.gru(gru_input, h_prev)

        # 先验
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.reshape(-1, self.state_size, self.num_categories)

        # 后验 (条件于观测)
        posterior_input = torch.cat([h, embed], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)
        posterior_logits = posterior_logits.reshape(-1, self.state_size, self.num_categories)

        # 从后验采样 (训练时)
        z = self._straight_through_categorical(posterior_logits)

        return {
            'h': h,
            'z': z,
            'prior_logits': prior_logits,
            'posterior_logits': posterior_logits,
        }

    def _straight_through_categorical(self, logits):
        """Straight-Through 梯度估计器"""
        probs = F.softmax(logits, dim=-1)
        indices = probs.argmax(dim=-1)
        one_hot = F.one_hot(indices, self.num_categories).float()
        # Straight-through: 前向用 one_hot，反向用 probs 梯度
        return one_hot + probs - probs.detach()

    def kl_loss(self, prior_logits, posterior_logits, free_bits=1.0, balance=0.8):
        """
        计算 KL 散度损失 (带 free bits 和 balancing)
        """
        prior_probs = F.softmax(prior_logits, dim=-1)
        posterior_probs = F.softmax(posterior_logits, dim=-1)

        # KL(posterior || prior) 对每个类别
        kl = (posterior_probs * (
            torch.log(posterior_probs + 1e-8) - torch.log(prior_probs + 1e-8)
        )).sum(dim=-1)  # sum over categories

        kl = kl.sum(dim=-1)  # sum over state dimensions

        # Free bits: 只惩罚超过阈值的部分
        kl = torch.clamp(kl, min=free_bits)

        # KL balancing
        # dynamics loss: 更新 prior
        dyn_loss = (posterior_probs.detach() * (
            torch.log(posterior_probs.detach() + 1e-8) - torch.log(prior_probs + 1e-8)
        )).sum(dim=(-2, -1))

        # representation loss: 更新 posterior
        rep_loss = (posterior_probs * (
            torch.log(posterior_probs + 1e-8) - torch.log(prior_probs.detach() + 1e-8)
        )).sum(dim=(-2, -1))

        return balance * dyn_loss.mean() + (1 - balance) * rep_loss.mean()


# ========== 使用示例 ==========
if __name__ == "__main__":
    batch_size = 32
    action_size = 6
    embed_size = 1024

    rssm = RSSM(hidden_size=512, state_size=32, num_categories=32,
                action_size=action_size, embed_size=embed_size)

    # 初始化
    state = rssm.initial_state(batch_size, device='cpu')

    # 模拟一步
    action = torch.randn(batch_size, action_size)
    embed = torch.randn(batch_size, embed_size)  # 来自 CNN encoder

    # 观测步（训练时）
    new_state = rssm.observe_step(state, action, embed)
    print(f"h shape: {new_state['h'].shape}")  # [32, 512]
    print(f"z shape: {new_state['z'].shape}")  # [32, 32, 32]

    # 想象步（Actor-Critic 训练时）
    imagined_state = rssm.imagine_step(state, action)
    print(f"Imagined h shape: {imagined_state['h'].shape}")

    # KL 损失
    kl = rssm.kl_loss(
        new_state['prior_logits'],
        new_state['posterior_logits']
    )
    print(f"KL loss: {kl.item():.4f}")
```

### 训练循环伪代码

```python
def train_dreamer_step(world_model, actor, critic, replay_buffer, config):
    """Dreamer 单步训练"""

    # ========== 1. 世界模型训练 ==========
    batch = replay_buffer.sample(config.batch_size, config.seq_len)

    # 编码观测
    embeds = world_model.encoder(batch.observations)

    # RSSM 前向传播
    states = []
    state = world_model.rssm.initial_state(config.batch_size, device)

    for t in range(config.seq_len):
        state = world_model.rssm.observe_step(
            state, batch.actions[:, t], embeds[:, t]
        )
        states.append(state)

    # 计算损失
    features = [world_model.rssm.get_state_feature(s) for s in states]
    features = torch.stack(features, dim=1)

    # 重建损失
    recon = world_model.decoder(features)
    recon_loss = F.mse_loss(recon, batch.observations)

    # 奖励预测损失
    pred_rewards = world_model.reward_head(features)
    reward_loss = F.mse_loss(pred_rewards, batch.rewards)

    # KL 损失
    kl_loss = sum(world_model.rssm.kl_loss(
        s['prior_logits'], s['posterior_logits']
    ) for s in states) / len(states)

    world_model_loss = recon_loss + reward_loss + config.kl_scale * kl_loss
    world_model.optimizer.zero_grad()
    world_model_loss.backward()
    world_model.optimizer.step()

    # ========== 2. Actor-Critic 训练 ==========
    # 从真实状态出发想象
    with torch.no_grad():
        start_state = random.choice(states)

    # 想象 H 步
    imagined_states = [start_state]
    imagined_rewards = []
    state = start_state

    for t in range(config.imagination_horizon):
        action = actor(world_model.rssm.get_state_feature(state))
        state = world_model.rssm.imagine_step(state, action)
        reward = world_model.reward_head(world_model.rssm.get_state_feature(state))

        imagined_states.append(state)
        imagined_rewards.append(reward)

    # 计算 λ-returns
    features = [world_model.rssm.get_state_feature(s) for s in imagined_states]
    values = critic(torch.stack(features, dim=1))
    returns = compute_lambda_returns(imagined_rewards, values, config.gamma, config.lambda_)

    # Critic 损失
    critic_loss = F.mse_loss(values[:, :-1], returns.detach())

    # Actor 损失 (最大化回报)
    actor_loss = -returns.mean()

    # 更新
    actor.optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor.optimizer.step()

    critic.optimizer.zero_grad()
    critic_loss.backward()
    critic.optimizer.step()

    return {
        'world_model_loss': world_model_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'actor_loss': actor_loss.item(),
        'critic_loss': critic_loss.item(),
    }
```

---

## 总结

### RSSM 的核心创新

1. **双路径设计**: 确定性 + 随机性分离
2. **先验-后验结构**: 支持想象与观测
3. **端到端训练**: 世界模型与策略联合优化

### 关键数学

| 组件 | 公式 |
|:---|:---|
| 确定性路径 | $h_t = \text{GRU}([z_{t-1}, a_{t-1}], h_{t-1})$ |
| 先验 | $p(z_t \mid h_t) = \prod_i \text{Cat}(\pi^i(h_t))$ |
| 后验 | $q(z_t \mid h_t, o_t) = \prod_i \text{Cat}(\pi^i(h_t, e_t))$ |
| ELBO | $\log p(o) \geq \mathbb{E}_q[\log p(o \mid z)] - D_{KL}(q \| p)$ |

### 从 World Models 到 RSSM 的演进

```
World Models        PlaNet/Dreamer
────────────        ──────────────
VAE 编码            CNN 编码 + RSSM
MDN-RNN (纯随机)    RSSM (确定+随机)
无先验-后验区分      明确的先验-后验
分阶段训练          端到端训练
```

---

## 参考文献

1. Hafner et al. "Learning Latent Dynamics for Planning from Pixels" (PlaNet, 2019)
2. Hafner et al. "Dream to Control" (Dreamer, 2020)
3. Hafner et al. "Mastering Atari with Discrete World Models" (DreamerV2, 2021)
4. Hafner et al. "Mastering Diverse Domains through World Models" (DreamerV3, 2023)
5. Kingma & Welling "Auto-Encoding Variational Bayes" (VAE, 2014)

---

> **下一步**: World Models vs Dreamer 详细对比实验 → `08_comparison_experiments.md`
