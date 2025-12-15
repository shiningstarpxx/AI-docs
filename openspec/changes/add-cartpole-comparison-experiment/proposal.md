# Change: CartPole 对比实验 - DQN vs World Model vs Mini-Dreamer

## Why

理论学习需要实验验证。在 CartPole 这个简单环境上进行对比实验可以：
- 直观验证 World Models 的样本效率优势
- 理解不同方法的实际表现和局限
- 积累 Model-Based RL 的实践经验

实验设计：
1. DQN (Model-Free baseline)
2. Simple World Model (VAE + MLP dynamics)
3. Mini-Dreamer (简化版 RSSM + Actor-Critic)

## What Changes

- 实现三种方法在 CartPole-v1 上的对比实验
- 关键指标：样本效率 (达到目标所需的真实交互次数)
- 可视化：学习曲线、样本效率对比图

## Impact

- Affected specs: world-models-experiments
- Affected files: `world_models/experiments/` 目录下的实验代码
- Dependencies: 当前 CarRacing 训练完成后执行
- Priority: 5 (最后执行，等训练完成)
