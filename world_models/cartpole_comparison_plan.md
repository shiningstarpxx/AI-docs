# CartPole-v1 世界模型对比实验

## 🎯 实验目标

在 MacBook MPS 上复现并对比三种方法，清晰展示世界模型的优势。

---

## 📋 任务选择：CartPole-v1

### 为什么选 CartPole？

```python
环境特点：
├─ 状态空间：4维 [x, ẋ, θ, θ̇]  ← 低维，易可视化
├─ 动作空间：2维 {左, 右}      ← 离散，易建模
├─ 奖励：每步 +1 (最多 500)   ← 稀疏但清晰
├─ 求解标准：连续 100 轮平均 ≥ 475
└─ 训练时间：<30 分钟          ← MPS 友好 ⭐

对比 Atari/DMC：
• 无需 CNN（省显存）
• 无需 MuJoCo（兼容性好）
• 快速迭代（适合调试）
```

---

## 🔬 实验设计

### 对比三种方法

| 方法 | 架构 | 预期样本效率 | 训练时间 |
|:---|:---|:---|:---|
| **DQN** (Baseline) | Q-Network | 100% | ~15分钟 |
| **Simple World Model** | VAE + LSTM + Linear | 30% (3×) | ~20分钟 |
| **Mini Dreamer** | RSSM + Actor-Critic | 20% (5×) | ~25分钟 |

### 核心对比维度

```ascii
1. 样本效率
   ├─ X轴：环境交互步数
   ├─ Y轴：平均回报
   └─ 目标：展示 World Model 用更少步骤达到 475

2. 训练稳定性
   ├─ 方差分析（5次重复实验）
   └─ 收敛速度

3. 模型质量
   ├─ 1步预测误差
   ├─ 5步预测误差
   └─ 可视化想象轨迹 vs 真实轨迹

4. 计算效率
   ├─ Wall-clock time
   ├─ 内存占用
   └─ MPS 利用率
```

---

## 🛠️ 实现细节

### 1. Baseline: DQN

```python
"""
标准 DQN 实现（PyTorch + MPS）
"""

class DQN:
    def __init__(self):
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        ).to('mps')
        
        self.optimizer = torch.optim.Adam(lr=1e-3)
        self.memory = ReplayBuffer(10000)
    
    def train_step(self, batch):
        # 标准 DQN 更新
        states, actions, rewards, next_states, dones = batch
        
        current_q = self.network(states).gather(1, actions)
        next_q = self.target_network(next_states).max(1)[0]
        target_q = rewards + 0.99 * next_q * (1 - dones)
        
        loss = F.mse_loss(current_q, target_q)
        loss.backward()
        self.optimizer.step()

# 预期性能：
# - 1000 episodes 收敛
# - 环境步骤：~50k steps
```

### 2. Simple World Model (World Models 简化版)

```python
"""
VAE + LSTM + 进化策略（简化实现）
"""

class SimpleWorldModel:
    def __init__(self):
        # V: 状态编码器（简化，无需 VAE）
        self.encoder = nn.Linear(4, 16).to('mps')
        
        # M: LSTM 动态模型
        self.lstm = nn.LSTM(16 + 2, 32, batch_first=True).to('mps')
        self.predictor = nn.Linear(32, 16).to('mps')
        
        # C: 线性策略（极简）
        self.controller = nn.Linear(16, 2).to('mps')
    
    def train_world_model(self, trajectories):
        """阶段 1+2: 训练 V+M"""
        for traj in trajectories:
            states, actions = traj
            
            # 编码
            z = self.encoder(states)  # (T, 16)
            
            # LSTM 预测
            inputs = torch.cat([z[:-1], actions[:-1]], dim=-1)
            h, _ = self.lstm(inputs.unsqueeze(0))
            z_pred = self.predictor(h.squeeze(0))
            
            # 预测损失
            loss = F.mse_loss(z_pred, z[1:])
            loss.backward()
    
    def train_controller_in_dream(self):
        """阶段 3: 在梦境中训练 C"""
        # 简化：用 CMA-ES 或者梯度优化
        for _ in range(100):
            # 想象 rollout
            z = self.encoder(initial_state)
            total_reward = 0
            
            for t in range(500):
                action = self.controller(z)
                z_next = self.imagine_next(z, action)
                reward = self.predict_reward(z_next)
                total_reward += reward
                z = z_next
            
            # 优化策略
            (-total_reward).backward()

# 预期性能：
# - 300 episodes 收敛 (3× faster)
# - 环境步骤：~15k steps
```

### 3. Mini Dreamer (Dreamer 核心简化版)

```python
"""
RSSM + Actor-Critic in Latent Space
"""

class MiniDreamer:
    def __init__(self):
        # RSSM: 双路径状态空间模型
        self.rnn = nn.GRUCell(16 + 2, 32).to('mps')  # 确定性
        self.posterior = nn.Linear(32 + 4, 16).to('mps')  # 随机
        self.prior = nn.Linear(32, 16).to('mps')
        
        # 观测/奖励模型
        self.obs_decoder = nn.Linear(32 + 16, 4).to('mps')
        self.reward_model = nn.Linear(32 + 16, 1).to('mps')
        
        # Actor-Critic
        self.actor = nn.Linear(32 + 16, 2).to('mps')
        self.critic = nn.Linear(32 + 16, 1).to('mps')
    
    def imagine_trajectory(self, start_h, start_s, horizon=15):
        """在想象中展开轨迹"""
        trajectory = []
        h, s = start_h, start_s
        
        for t in range(horizon):
            # Actor 采样动作
            logits = self.actor(torch.cat([h, s], dim=-1))
            action = Categorical(logits=logits).sample()
            
            # RSSM 预测下一状态
            h = self.rnn(torch.cat([s, action], dim=-1), h)
            s = self.prior(h)  # 简化：不采样
            
            # 预测奖励
            reward = self.reward_model(torch.cat([h, s], dim=-1))
            
            trajectory.append((h, s, action, reward))
        
        return trajectory
    
    def train_behavior(self, initial_states):
        """在想象中训练 Actor-Critic"""
        for s0 in initial_states:
            # 想象轨迹
            traj = self.imagine_trajectory(s0, horizon=15)
            
            # 计算 λ-return
            returns = compute_lambda_return(traj)
            
            # 训练 Critic
            for (h, s, _, _), G in zip(traj, returns):
                value = self.critic(torch.cat([h, s], dim=-1))
                critic_loss = (value - G) ** 2
                critic_loss.backward()
            
            # 训练 Actor
            for (h, s, a, _), G in zip(traj, returns):
                value = self.critic(torch.cat([h, s], dim=-1)).detach()
                advantage = G - value
                
                logits = self.actor(torch.cat([h, s], dim=-1))
                log_prob = Categorical(logits=logits).log_prob(a)
                actor_loss = -log_prob * advantage
                actor_loss.backward()

# 预期性能：
# - 200 episodes 收敛 (5× faster)
# - 环境步骤：~10k steps
```

---

## 📊 评估指标

### 1. 主要指标

```python
metrics = {
    # 样本效率
    'episodes_to_solve': int,  # 达到 475 所需轮数
    'env_steps_to_solve': int,  # 真实环境交互步数
    
    # 训练稳定性
    'final_reward_mean': float,
    'final_reward_std': float,
    'convergence_speed': float,  # 0.1 → 0.9 性能所需步数
    
    # 模型质量（仅 World Model）
    '1step_prediction_mse': float,
    '5step_prediction_mse': float,
    
    # 计算效率
    'wall_clock_time': float,  # 秒
    'memory_peak': float,  # MB
    'mps_utilization': float,  # %
}
```

### 2. 可视化

```python
# 图 1: 学习曲线
plt.plot(env_steps, rewards_dqn, label='DQN')
plt.plot(env_steps, rewards_wm, label='Simple WM')
plt.plot(env_steps, rewards_dreamer, label='Mini Dreamer')
plt.xlabel('Environment Steps')
plt.ylabel('Average Reward (100 episodes)')
plt.title('Sample Efficiency Comparison')

# 图 2: 想象 vs 真实轨迹
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
states_real = collect_real_trajectory()
states_imagined = model.imagine_trajectory()

for i, (label, dim) in enumerate([
    ('Position', 0), ('Velocity', 1), 
    ('Angle', 2), ('Angular Velocity', 3)
]):
    axes[i//2, i%2].plot(states_real[:, dim], label='Real', alpha=0.7)
    axes[i//2, i%2].plot(states_imagined[:, dim], label='Imagined', alpha=0.7)
    axes[i//2, i%2].set_title(label)
    axes[i//2, i%2].legend()

# 图 3: 预测误差累积
horizons = [1, 3, 5, 10, 15]
errors = [compute_prediction_error(h) for h in horizons]
plt.plot(horizons, errors)
plt.xlabel('Prediction Horizon')
plt.ylabel('MSE')
plt.title('Compounding Error Analysis')
```

---

## 🚀 实验流程

### Step 1: 环境搭建（10 分钟）

```bash
# 创建虚拟环境
conda create -n world_models python=3.10
conda activate world_models

# 安装依赖
pip install torch torchvision  # MPS 支持
pip install gymnasium[classic-control]
pip install matplotlib seaborn pandas
pip install tqdm tensorboard

# 验证 MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Step 2: 实现三种方法（2-3 天）

```
Day 1: DQN baseline
  ├─ 实现 DQN 核心算法
  ├─ 调试至收敛
  └─ 记录 baseline 性能

Day 2: Simple World Model
  ├─ 实现 VAE/编码器
  ├─ 实现 LSTM 动态模型
  ├─ 梯度优化策略（简化 CMA-ES）
  └─ 对比 DQN

Day 3: Mini Dreamer
  ├─ 实现 RSSM
  ├─ 实现 Actor-Critic
  ├─ 想象训练循环
  └─ 完整对比
```

### Step 3: 运行实验（1 天）

```python
# 每种方法运行 5 次
for method in ['dqn', 'simple_wm', 'mini_dreamer']:
    for seed in range(5):
        run_experiment(
            method=method,
            seed=seed,
            max_episodes=1000,
            device='mps'
        )

# 预期总时间：~3-4 小时
```

### Step 4: 分析结果（1 天）

```python
# 加载所有实验结果
results = load_all_results()

# 统计分析
summary = {
    'method': [],
    'episodes_to_solve': [],
    'env_steps_to_solve': [],
    'final_reward': [],
    'wall_time': []
}

for method in ['dqn', 'simple_wm', 'mini_dreamer']:
    data = results[method]
    summary['method'].append(method)
    summary['episodes_to_solve'].append(data['episodes'].mean())
    # ... 其他指标

# 生成报告
generate_comparison_report(summary)
```

---

## 📈 预期结果

### 样本效率对比

```ascii
环境步骤 (达到 475 reward):

DQN:           ████████████████████ 50k steps
Simple WM:     ██████ 15k steps (3.3× faster) ⭐
Mini Dreamer:  ████ 10k steps (5× faster) ⭐⭐

关键洞察：
✓ World Model 显著减少真实环境交互
✓ Dreamer 的 Actor-Critic > World Models 的进化策略
✓ RSSM 的双路径设计提升预测准确度
```

### 训练时间对比

```ascii
Wall-clock time (MacBook Pro M1, 16GB):

DQN:           ███████ ~15 分钟
Simple WM:     █████████ ~20 分钟 (模型训练开销)
Mini Dreamer:  ██████████ ~25 分钟 (RSSM 复杂度)

说明：
• World Model 前期训练模型耗时
• 但环境交互少 → 总时间接近
• 实际应用中环境交互才是瓶颈
```

### 模型质量

```ascii
预测误差 (MSE):

Horizon      LSTM    RSSM
1-step:      0.05    0.03  ← RSSM 更准确
3-step:      0.15    0.10
5-step:      0.35    0.22
10-step:     0.80    0.50  ← 长期预测差距扩大

原因：RSSM 显式建模随机性，避免模式平均
```

---

## 💡 扩展实验（可选）

### 1. 消融实验

```python
ablation_studies = {
    'A1': 'RSSM vs 纯 RNN',
    'A2': '想象视野 H=5/10/15',
    'A3': '离散 vs 连续潜在变量',
    'A4': 'Actor-Critic vs 进化策略',
}

# 每个消融实验 30 分钟
```

### 2. 可视化分析

```python
visualizations = [
    'latent_space_evolution',  # t-SNE 降维
    'attention_weights',        # RSSM 关注什么
    'imagination_quality',      # 想象 vs 真实
    'policy_behavior',          # 决策边界
]
```

### 3. 升级到 Pendulum-v1

```python
# 连续控制任务
# 状态：3维
# 动作：1维连续
# 训练时间：~1 小时
```

---

## 🎓 学习价值

通过这个实验，你将理解：

1. **为什么需要世界模型**：样本效率提升 3-5 倍
2. **RSSM 的优势**：双路径设计 > 单一 RNN
3. **想象训练的威力**：在心智模拟中学习
4. **实际工程权衡**：模型训练成本 vs 环境交互成本

---

## 📦 代码结构

```
world_models_cartpole/
├── agents/
│   ├── dqn.py           # DQN baseline
│   ├── world_model.py   # Simple World Model
│   └── dreamer.py       # Mini Dreamer
├── models/
│   ├── rssm.py          # RSSM 实现
│   ├── networks.py      # 通用网络层
│   └── replay.py        # Replay buffer
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── visualize.py         # 可视化工具
└── configs/
    ├── dqn.yaml
    ├── world_model.yaml
    └── dreamer.yaml
```

---

## ⏱️ 时间估算

| 阶段 | 任务 | 时间 |
|:---|:---|:---|
| 准备 | 环境搭建 | 10 分钟 |
| 实现 | DQN | 4 小时 |
| 实现 | Simple WM | 6 小时 |
| 实现 | Mini Dreamer | 8 小时 |
| 实验 | 运行 5 次重复 | 4 小时 |
| 分析 | 结果分析+可视化 | 4 小时 |
| **总计** | | **~2-3 天** |

---

## 🚦 Ready to Start?

建议实现顺序：

1. ✅ **DQN** (最简单，建立 baseline)
2. ✅ **Simple World Model** (理解梦境训练)
3. ✅ **Mini Dreamer** (核心创新)

每个方法调试通过后再进行对比实验，避免同时调试多个模型。

祝实验顺利！🎉
