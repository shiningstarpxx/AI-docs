# Simple World Model V2 改进说明

## 🔧 改进策略（方案 A）

针对 V1 版本的失败（梦境 103 vs 真实 17），进行三个核心改进：

### 1. 数据收集改进 ✨

**V1 问题**: 纯随机策略，平均奖励只有 20 分
```python
# V1: 纯随机
action = env.action_space.sample()  # 性能极差
```

**V2 改进**: 先用 DQN 预训练，再用 ε-greedy 收集数据
```python
# 阶段 0: 预训练 DQN (100 episodes)
self.pretrain_dqn()  # 达到 100+ 性能

# 阶段 1: 用策略收集 (ε: 0.5 → 0.1)
action = self.dqn.get_action(state, epsilon)  # 高质量数据
```

**预期效果**: 数据平均奖励从 20 → 100+

---

### 2. 世界模型改进 ✨

**V1 问题**: 容量不足，无法准确建模
```python
# V1: 单层 LSTM, 128 hidden
self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
```

**V2 改进**: 2层 LSTM + 256 hidden + 更多训练
```python
# V2: 2层 LSTM, 256 hidden, dropout
self.lstm = nn.LSTM(
    hidden_size=256,
    num_layers=2,
    dropout=0.1,
    batch_first=True
)

# 训练轮次: 100 → 200 epochs
# 每轮更新: 50 → 100 batches
```

**预期效果**: 世界模型预测误差显著降低

---

### 3. 控制器改进 ✨

**V1 问题**: 线性控制器表达能力太弱
```python
# V1: 线性策略 + CMA-ES
self.weights = np.random.randn(action_dim, input_dim)
action = argmax(W @ state)  # 太简单
```

**V2 改进**: 神经网络 + 梯度优化
```python
# V2: 3层神经网络
self.network = nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, action_dim)
)

# 在梦境中梯度优化（而非进化）
# 5000 steps, batch_size=32, horizon=50
```

**预期效果**: 策略表达能力大幅提升

---

## 📊 预期性能对比

| 版本 | 数据质量 | 模型容量 | 控制器 | 预期性能 |
|:---|:---|:---|:---|:---|
| **V1** | 20 分 | 128-1层 | 线性 | 17 分 ❌ |
| **V2** | 100+ 分 | 256-2层 | 神经网络 | 150+ 分 ✅ |

---

## ⏱️ 训练时间估算

```
阶段 0: DQN 预训练      ~5 分钟
阶段 1: 数据收集        ~8 分钟
阶段 2: 世界模型训练    ~15 分钟
阶段 3: 梦境策略训练    ~10 分钟
--------------------------------
总计:                   ~40 分钟
```

---

## 🎯 成功标准

**基本目标**: 真实环境评估 > 100 分
- 证明世界模型不是完全失败
- 高质量数据 + 强模型 = 可用

**理想目标**: 真实环境评估 > 150 分
- 接近 DQN baseline (193 分)
- 展示 World Model 的可行性

---

## 🔍 关键技术点

### 梯度优化 vs 进化算法

**V1 (CMA-ES)**:
- ✅ 无需梯度，简单
- ❌ 采样效率低
- ❌ 高维参数空间效果差

**V2 (梯度优化)**:
- ✅ 直接优化
- ✅ 样本效率高
- ⚠️ 需要世界模型可微

### 梦境训练的梯度流

```python
# 关键：世界模型预测保留梯度
for t in range(horizon):
    # 策略选择动作（需要梯度）
    logits = controller(state)
    action = sample(logits)
    
    # 世界模型预测（保留梯度！）
    with torch.enable_grad():
        next_state, reward = world_model(state, action)
    
    total_reward += reward
    state = next_state.detach()  # 下一步断开

# 优化策略
loss = -total_reward.mean()
loss.backward()  # 梯度通过世界模型回传
```

---

## 📈 监控指标

### 阶段 0: DQN 预训练
- 目标: 最后 20 episodes 平均 > 100
- ε 衰减: 1.0 → 0.1

### 阶段 1: 数据收集
- 目标: 平均奖励 > 100
- ε 范围: 0.5 → 0.1 (保持探索)

### 阶段 2: 世界模型
- Loss 下降: 应该稳定收敛到 < 0.01
- 检查: 状态预测 MSE

### 阶段 3: 梦境训练
- Dream reward 应该逐步上升
- 最终评估: > 100 分

---

## 🚨 可能的问题

### 问题 1: 世界模型仍然不准确

**症状**: Loss 收敛但评估性能差
**原因**: 数据分布偏差
**解决**: 增加数据量，或使用更多样的策略收集

### 问题 2: 梯度优化不稳定

**症状**: Dream reward 震荡
**原因**: 学习率过大，或梯度爆炸
**解决**: 降低 controller_lr, 增加梯度裁剪

### 问题 3: 性能仍不如 DQN

**说明**: 这是正常的！
- World Model 优势在于样本效率
- CartPole 本身不太适合 World Model
- 但至少应该 > 100 分

---

## 🎓 经验总结

### 成功经验

1. **数据质量 >> 数据量**
   - 100 条好数据 > 1000 条差数据

2. **模型容量很重要**
   - 不要吝啬参数量
   - CartPole 虽简单，但动态敏感

3. **梯度优化 > 进化算法**
   - 对于可微模型，梯度更高效

### 失败教训

1. **不要在错误的模型中训练**
   - V1 的世界模型太差，策略越训练越错

2. **CartPole 对 World Model 不友好**
   - 状态空间小但动态复杂
   - 更适合视觉任务或物理仿真

3. **调试比实现更重要**
   - V1 代码没问题，但配置不对
   - 参数调优占 50% 时间

---

## 📚 如果V2仍然失败...

### 备选方案

**方案 B1**: 简化目标
- 不追求性能，重点演示原理
- 文档说明 CartPole 的局限性

**方案 B2**: 换任务
- MountainCar: 稀疏奖励，更需要规划
- Pendulum: 连续控制，物理直觉强

**方案 B3**: 使用真实数据
- 直接用 DQN 收集 500 分的轨迹
- 展示"如果模型完美"的理论上界

---

**当前状态**: V2 训练中 (PID: 71873)

**预计完成**: ~40 分钟

**实时监控**: `./monitor_training.sh`
