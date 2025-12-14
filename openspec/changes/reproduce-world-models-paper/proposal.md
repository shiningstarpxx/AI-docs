# Change: 复现 World Models (Ha & Schmidhuber, 2018) 论文实验

## Why

为了深入理解 Model-Based RL 的核心思想，通过亲手复现 World Models 论文的实验来验证：
1. 世界模型 (VAE + MDN-RNN) 能否准确建模环境动态
2. 在"梦境"中训练策略是否能提高样本效率
3. Model-Based 方法相比 Model-Free (DQN) 的优劣势

## What Changes

### 实验设计
- **CartPole-v1 对比实验**: 三种方法（DQN, Simple World Model, Full World Model）
- **CarRacing-v3 复现**: 完整论文设置的 VAE + MDN-LSTM + CMA-ES

### 实现的组件
1. **Vision Model (V)**: VAE 状态编码器
2. **Memory Model (M)**: MDN-LSTM 预测状态分布
3. **Controller (C)**: 线性策略 + CMA-ES 进化优化

### 关键发现
- CartPole 环境可能不适合 World Model 演示（状态空间小但动态敏感）
- 数据收集质量对世界模型训练至关重要
- Dream vs Reality gap 是核心挑战

## Impact

- 新增实验代码: `world_models/experiments/`
- 新增技术文档: World Models 数学推导和代码解读
- 为后续 Dreamer 系列研究打下基础

## 当前状态

**进行中** - 实验已完成，需要分析结果和改进
