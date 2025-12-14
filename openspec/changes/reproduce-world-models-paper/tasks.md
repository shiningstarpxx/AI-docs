# 任务清单: World Models 论文复现

## 1. 基础设施 ✅
- [x] 1.1 搭建实验环境 (PyTorch + Gymnasium)
- [x] 1.2 创建虚拟环境和依赖管理
- [x] 1.3 设计实验目录结构

## 2. CartPole 对比实验

### 2.1 DQN Baseline ✅
- [x] 实现 DQN 算法 (`1_baseline_dqn.py`)
- [x] 训练 800 episodes
- [x] 结果: 193.3 ± 128.9 (性能不稳定，但可作为 baseline)

### 2.2 Simple World Model ✅
- [x] 实现 LSTM 世界模型 (`2_simple_world_model.py`)
- [x] 实现 CMA-ES 控制器优化
- [x] 训练并评估
- [x] 结果: 梦境 ~103 vs 真实 ~17 (**严重模型偏差**)

### 2.3 Full World Model (MDN-RNN) ✅
- [x] 实现完整 VAE + MDN-LSTM 架构 (`2_world_model_full.py`)
- [x] 实现 MDN (Mixture Density Network)
- [x] 训练并评估
- [x] 结果: 梦境 ~208 vs 真实 ~9.6 (**模型偏差更严重**)

## 3. CarRacing 复现实验 🚀
- [x] 3.1 实现论文完整架构 (`3_car_racing_world_model.py`)
- [x] 3.2 快速测试模式运行 (50 rollouts)
- [x] 3.3 结果: 梦境 ~106 vs 真实 ~12.8
- [x] 3.4 添加 checkpoint 功能支持中断恢复
- [x] 3.5 添加进度日志和 ETA 估算
- [ ] 3.6 **进行中**: 完整论文设置训练 (PID: 93750, 启动于 2025-12-10 10:56)
  - 10000 rollouts 数据收集
  - VAE 10 epochs
  - MDN-RNN 20 epochs
  - CMA-ES 300 generations
- [ ] 3.7 **待完成**: 达到论文报告的 ~900 分性能

## 4. 结果分析与文档 ⏳
- [x] 4.1 记录训练曲线和指标
- [x] 4.2 创建 PROGRESS.md 进度报告
- [ ] 4.3 **待完成**: 分析 Dream-Reality Gap 的根本原因
- [ ] 4.4 **待完成**: 撰写实验总结和教训

## 5. 下一步计划 (待决策)

### 方案 A: 修复 CartPole 实验
- [ ] 改进数据收集策略 (用 ε-greedy 替代纯随机)
- [ ] 增大模型容量 (hidden_size: 64 → 256)
- [ ] 调整损失权重

### 方案 B: 专注 CarRacing 完整复现
- [ ] 运行完整 10000 rollouts 数据收集
- [ ] 完整 VAE + MDN-RNN 训练
- [ ] 300+ 代 CMA-ES 优化

### 方案 C: 转向 Dreamer 系列
- [ ] 跳过经典 World Models，直接实现 Dreamer-v1/v2
- [ ] 使用现代 RSSM 架构

---

## 关键问题记录

### 问题 1: 为什么 CartPole 上 World Model 失败？
**分析**:
1. 数据质量差：随机策略平均仅 ~22 分
2. 状态空间虽小(4维)，但动态对初始条件敏感
3. LSTM 难以准确建模 CartPole 的精确物理动态
4. 论文原本针对高维视觉输入设计，低维状态反而不适合

### 问题 2: Dream vs Reality Gap 如何解决？
**潜在方案**:
1. 提高世界模型精度（更多数据、更大模型）
2. 使用 ensemble 世界模型
3. Dyna-style: 混合真实和梦境训练
4. 正则化：限制梦境 rollout 长度

---

**最后更新**: 2025-12-10
