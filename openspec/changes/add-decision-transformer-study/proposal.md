# Change: Decision Transformer 研究 - 把 RL 变成序列建模

## Why

Decision Transformer 代表了一种范式转变：将强化学习问题重新框架为序列建模问题。这与我们研究的 World Models 路线形成有趣对比：
- World Models: 学习环境模型 → 在想象中规划
- Decision Transformer: 直接学习 (状态, 动作, 回报) 序列的生成模型

理解这一方向有助于把握"统一 Transformer 架构"的前沿趋势。

## What Changes

- 添加 Decision Transformer 深度解析文档
- 对比分析：World Models vs Decision Transformer 的哲学差异
- 理解 Return-Conditioned 生成的核心机制
- 探讨 Trajectory Transformer、Gato 等后续工作

## Impact

- Affected specs: world-models-research
- Affected files: `world_models/14_decision_transformer.md`
- Dependencies: 无，可独立进行
- Priority: 1 (优先级最高)
