# Tasks: World Models 研究深化任务清单

## 任务状态说明

- `[x]` 已完成
- `[ ]` 待完成
- `[~]` 进行中

---

## Phase 1: 完成进行中项目 (本周)

### 1.1 CarRacing 实验完成
- [~] CMA-ES 控制器训练 (Generation 70/300)
- [ ] 真实环境评估 (目标: ~900 分)
- [ ] 实验结果分析与文档化

### 1.2 MPS/GPU 优化
- [ ] 批量化 dream rollouts 实现
- [ ] 向量化 RNN 推理
- [ ] 性能基准测试

### 1.3 文档标记更新
- [ ] 更新 learning_plan.md 完成状态
- [ ] 更新 TODO.md 进度
- [ ] 同步各 OpenSpec 变更状态

---

## Phase 2: 阶段三 Model-Based RL 深入 (下周)

### 2.1 Dyna 算法实现
- [ ] GridWorld 环境搭建
- [ ] 表格式 Q-learning 基线
- [ ] Dyna-Q 实现
- [ ] 优先级扫描 (Prioritized Sweeping)
- [ ] Planning steps 对比实验
- [ ] 文档: `19_dyna_algorithm.md`

### 2.2 Model-Based RL 理论深化
- [ ] 阅读 MBRL Survey (Moerland et al., 2023)
- [ ] 模型误差分析笔记
- [ ] 短视野 vs 长视野规划对比
- [ ] 文档: `20_mbrl_theory.md`

---

## Phase 3: MBPO 与探索 (两周内)

### 3.1 MBPO 实现
- [ ] 概率集成模型 (Probabilistic Ensemble)
- [ ] MuJoCo 环境配置 (HalfCheetah)
- [ ] SAC 策略优化集成
- [ ] 短视野 rollout (k=5) 实现
- [ ] 混合缓冲区管理
- [ ] 样本效率对比 (vs SAC baseline)
- [ ] 文档: `21_mbpo_implementation.md`

### 3.2 好奇心驱动探索
- [ ] ICM (Intrinsic Curiosity Module) 实现
- [ ] MiniGrid 稀疏奖励环境
- [ ] 内在奖励可视化
- [ ] 探索覆盖率分析
- [ ] 文档: `22_curiosity_exploration.md`

---

## Phase 4: 视频预测与扩散模型 (月内)

### 4.1 视频预测基线
- [ ] Moving MNIST 数据准备
- [ ] ConvLSTM 基线实现
- [ ] VAE 视频预测版本
- [ ] 评估指标 (PSNR, SSIM, LPIPS)
- [ ] 文档: `23_video_prediction.md`

### 4.2 扩散世界模型探索
- [ ] DIAMOND 论文精读
- [ ] 扩散模型在 RL 中的应用理解
- [ ] 小规模实验 (如有资源)
- [ ] 文档: `24_diffusion_wm_deep.md`

---

## Phase 5: 前沿方向 (持续)

### 5.1 Decision Transformer
- [ ] 论文精读 + 笔记
- [ ] 与 World Models 对比分析
- [ ] 序列建模视角的 RL 理解
- [ ] 文档: `25_decision_transformer.md`

### 5.2 因果世界模型深入
- [ ] Causal Confusion 问题实验
- [ ] 因果推断基础学习
- [ ] 文档更新: 16_causal_world_models.md

### 5.3 多模态融合
- [ ] RT-X / Gato 架构学习
- [ ] Vision-Language-Action 统一表示
- [ ] 概念性文档: `26_multimodal_wm.md`

---

## 已完成任务归档

### 阶段一: 基础理论
- [x] VAE 数学原理 (02_vae_math.md)
- [x] RNN/MDN 数学原理 (03_rnn_mdn_math.md)

### 阶段二: 经典世界模型
- [x] World Models 论文精读 (04_paper_review.md)
- [x] World Models 代码实现 (experiments/3_car_racing_world_model.py)
- [x] Dreamer 系列研究 (06_dreamer_series.md)
- [x] RSSM 数学原理 (07_rssm_math.md)
- [x] DreamerV3 代码走读 (17_dreamerv3_code_walkthrough.md)
- [x] CartPole 对比实验 (18_experiment_report.md)

### 阶段四: 视频生成 (概念)
- [x] Sora/Genie 研究 (12_future_world_models.md)
- [x] JEPA 对比 (13_genie_jepa_comparison.md)
- [x] Diffusion 世界模型 (15_diffusion_world_models.md)

### 技术分享
- [x] World_Models_Deep_Dive.md (92页)
- [x] World_Models_Presentation.md (标准版)
- [x] World_Models_Presentation_short.md (精简版)
- [x] World_Models_Presentation_long.md (完整版)

---

## 优先级矩阵

| 任务 | 重要性 | 紧急度 | 预计时间 |
|:---|:---|:---|:---|
| CarRacing 完成 | 高 | 高 | ~3天 |
| MPS 优化 | 中 | 中 | 1天 |
| Dyna 实现 | 高 | 中 | 3天 |
| MBPO 实现 | 高 | 低 | 1周 |
| 视频预测 | 中 | 低 | 1周 |
| Decision Transformer | 中 | 低 | 3天 |

---

## 里程碑

### 短期 (1周)
- [ ] CarRacing 实验完成
- [ ] Dyna 算法实现

### 中期 (1月)
- [ ] MBPO 实现并验证
- [ ] 视频预测基线完成
- [ ] 文档体系达到 25+ 篇

### 长期 (3月)
- [ ] 10 个实践项目完成 8 个
- [ ] 原创性研究方向确定
- [ ] 技术博客系列发布
