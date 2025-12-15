# Change: 因果世界模型研究 - 从相关性到因果性

## Why

当前的世界模型（包括 Dreamer、Genie）学习的是相关性，而非因果性：
- 相关模型：观察到 A 后 B 发生 → 预测 A→B
- 因果模型：理解 A 是否真正导致 B

因果理解是通向真正"世界理解"的关键：
- 支持反事实推理："如果当时选择了 B 会怎样？"
- 更好的泛化：因果关系在分布变化下更鲁棒
- 更高效的规划：理解干预 (intervention) 的效果

## What Changes

- 添加因果世界模型深度解析文档
- 因果推断基础：SCM, do-calculus, 反事实
- 因果发现在世界模型中的应用
- 代表工作：Causal World Models, CausalCity 等

## Impact

- Affected specs: world-models-research
- Affected files: `world_models/16_causal_world_models.md`
- Dependencies: 基础因果推断概念
- Priority: 3
