## ADDED Requirements

### Requirement: World Models 论文复现实验框架

实验框架 SHALL 支持对比三种强化学习方法:
1. DQN (Model-Free Baseline)
2. Simple World Model (LSTM + CMA-ES)
3. Full World Model (VAE + MDN-LSTM + CMA-ES)

#### Scenario: CartPole-v1 环境对比
- **WHEN** 在 CartPole-v1 环境上运行三种方法
- **THEN** 记录样本效率、最终性能、训练稳定性指标
- **AND** 生成可视化训练曲线

#### Scenario: CarRacing-v3 论文复现
- **WHEN** 使用论文完整设置运行 CarRacing 实验
- **THEN** 实现 VAE(32-dim) + MDN-LSTM(256 hidden, 5 gaussians)
- **AND** 使用 CMA-ES 优化线性控制器

### Requirement: Vision Model (V) - VAE 实现

系统 SHALL 实现变分自编码器用于状态/图像编码:
- 编码器: 输入 → μ, log_σ (潜在分布参数)
- 解码器: z → 重建输入
- 损失: 重建损失 + β * KL散度

#### Scenario: CartPole 状态编码
- **WHEN** 输入 4 维 CartPole 状态
- **THEN** 编码到 16 维潜在空间
- **AND** 能够准确重建原始状态

#### Scenario: CarRacing 图像编码
- **WHEN** 输入 64x64 RGB 图像
- **THEN** 编码到 32 维潜在空间 (论文设置)

### Requirement: Memory Model (M) - MDN-LSTM 实现

系统 SHALL 实现混合密度网络-LSTM 用于预测状态转移分布:
- 输入: (z_t, a_t, h_{t-1})
- 输出: P(z_{t+1}) = Σ π_i * N(μ_i, σ_i²)
- 同时预测 reward 和 done 信号

#### Scenario: 状态分布预测
- **WHEN** 给定当前潜在状态 z 和动作 a
- **THEN** 输出 K 个高斯分量的混合分布
- **AND** 能够从分布中采样下一状态

#### Scenario: 梦境展开 (Dream Rollout)
- **WHEN** 从初始状态开始想象
- **THEN** 能够生成完整的想象轨迹
- **AND** 轨迹长度可配置 (默认 200-1000 步)

### Requirement: Controller (C) - CMA-ES 优化

系统 SHALL 实现线性控制器并使用 CMA-ES 进化算法优化:
- 控制器: action = argmax(W @ [z, h])
- 优化: CMA-ES 无梯度搜索

#### Scenario: 梦境适应度评估
- **WHEN** 在世界模型中评估控制器
- **THEN** 运行多次 rollout 取平均
- **AND** 返回适应度分数

#### Scenario: 真实环境评估
- **WHEN** 将训练好的控制器部署到真实环境
- **THEN** 评估实际性能
- **AND** 对比梦境性能以量化 Dream-Reality Gap
