# Change: Diffusion World Models 研究 - 用扩散模型做世界模型

## Why

扩散模型 (Diffusion Models) 在图像/视频生成领域取得了巨大成功 (DALL-E, Stable Diffusion, Sora)。将扩散模型应用于世界模型是最新的研究热点：
- 扩散模型天然建模多模态分布，适合不确定性建模
- 可以生成高质量的未来预测
- 与 Dreamer 的离散潜在空间形成对比

## What Changes

- 添加 Diffusion World Models 深度解析文档
- 分析扩散模型如何建模环境动态
- 对比：VAE/离散 vs 扩散的世界模型
- 代表工作：UniSim, DIAMOND, Diffusion World Model 等

## Impact

- Affected specs: world-models-research
- Affected files: `world_models/15_diffusion_world_models.md`
- Dependencies: 基础扩散模型知识
- Priority: 2
