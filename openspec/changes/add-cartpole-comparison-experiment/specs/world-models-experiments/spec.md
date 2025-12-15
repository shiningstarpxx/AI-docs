## ADDED Requirements

### Requirement: CartPole Comparison Experiment
系统 SHALL 包含在 CartPole 环境上的对比实验，验证 Model-Based RL 相对于 Model-Free RL 的样本效率优势。

#### Scenario: DQN Baseline Training
- **WHEN** 运行 DQN baseline 实验
- **THEN** 应记录完整的学习曲线
- **AND** 应记录达到目标分数 (195) 所需的交互步数

#### Scenario: World Model Training
- **WHEN** 运行 World Model 实验
- **THEN** 应使用更少的真实交互达到相同目标
- **AND** 应记录世界模型预测准确率

#### Scenario: Comparison Visualization
- **WHEN** 生成对比报告
- **THEN** 应包含学习曲线对比图
- **AND** 应包含样本效率对比柱状图
- **AND** 应包含各方法优缺点分析
