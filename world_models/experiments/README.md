# World Models 实验对比

基于 CartPole-v1 的三种方法对比实验，运行在 MacBook MPS 上。

## 📁 文件结构

```
experiments/
├── 1_baseline_dqn.py          # Baseline: DQN (Model-Free RL)
├── 2_simple_world_model.py    # Simple World Model (2018)
├── 3_mini_dreamer.py           # Mini Dreamer (2020)
├── compare_results.py          # 对比分析脚本
└── README.md                   # 本文档
```

## 🎯 实验目标

对比三种方法在 CartPole-v1 上的：
1. **样本效率**：达到相同性能所需的环境步数
2. **训练时间**：墙上时钟时间
3. **最终性能**：平均回报
4. **稳定性**：方差分析

## 🔬 实验方案

### 方法 1: DQN (Baseline)

**特点**：
- Model-Free RL
- 端到端学习 Q 函数
- ε-greedy 探索

**预期**：
- 样本效率：1× (基线)
- 收敛步数：~50k steps
- 训练时间：~30 分钟

**运行**：
```bash
python 1_baseline_dqn.py
```

### 方法 2: Simple World Model

**特点**：
- 三阶段训练：数据收集 → 世界模型 → 策略进化
- LSTM 动态模型
- CMA-ES 进化算法
- 在"梦境"中训练策略

**预期**：
- 样本效率：~3× DQN
- 收敛步数：~15k steps
- 训练时间：~40 分钟

**运行**：
```bash
python 2_simple_world_model.py
```

### 方法 3: Mini Dreamer

**特点**：
- RSSM 动态模型（确定性 + 随机性）
- Actor-Critic 在想象中学习
- 在线学习（持续改进）

**预期**：
- 样本效率：~5× DQN
- 收敛步数：~10k steps
- 训练时间：~45 分钟

**运行**：
```bash
python 3_mini_dreamer.py
```

## 📊 对比分析

运行所有实验后，使用对比脚本生成分析报告：

```bash
python compare_results.py
```

**输出**：
- `comparison_report.png`：样本效率对比图
- `comparison_metrics.json`：定量指标对比
- `comparison_table.md`：结果表格

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch gymnasium numpy matplotlib
```

### 2. 运行完整实验

```bash
# 按顺序运行三个实验
python 1_baseline_dqn.py
python 2_simple_world_model.py
python 3_mini_dreamer.py

# 生成对比报告
python compare_results.py
```

### 3. 查看结果

```
results_dqn/
├── dqn_model.pt
├── training_data.json
└── training_curves.png

results_simple_wm/
├── world_model.pt
├── training_history.json
└── training_curves.png

results_mini_dreamer/
├── models.pt
├── training_data.json
└── training_curves.png
```

## 🔍 关键对比维度

| 维度 | DQN | Simple WM | Mini Dreamer |
|:---|:---|:---|:---|
| **架构** | Q-Network | VAE + LSTM + Linear | RSSM + Actor-Critic |
| **训练方式** | 端到端 | 三阶段解耦 | 在线联合训练 |
| **策略优化** | Q-learning | CMA-ES | Policy Gradient |
| **样本效率** | 1× | ~3× | ~5× |
| **计算复杂度** | 低 | 中 | 高 |
| **可解释性** | 低 | 中（可视化梦境） | 高（潜在空间） |

## 🎓 学习路径

### 初学者
1. 运行 DQN baseline，理解基本 RL
2. 对比 Simple WM，理解"梦境学习"概念
3. 阅读代码注释，理解模块化设计

### 进阶者
1. 修改超参数，观察影响
2. 可视化世界模型预测质量
3. 实现 Pendulum-v1（连续控制）

### 高级者
1. 实现 DreamerV2 的离散潜在空间
2. 扩展到 Atari 游戏（视觉输入）
3. 复现论文中的消融实验

## 📝 实验笔记

### DQN
- [x] ε-greedy 探索
- [x] Target Network
- [x] Experience Replay
- [ ] Double DQN (可选)
- [ ] Dueling DQN (可选)

### Simple World Model
- [x] LSTM 世界模型
- [x] CMA-ES 策略优化
- [x] 梦境评估
- [ ] MDN 多模态预测 (可选)

### Mini Dreamer
- [x] RSSM 双路径设计
- [x] Actor-Critic 在想象中学习
- [x] GAE 优势估计
- [ ] 离散潜在变量 (DreamerV2)
- [ ] Symlog 预测 (DreamerV3)

## 🐛 常见问题

### Q1: MPS 设备不可用？
```python
# 回退到 CPU
device = torch.device("cpu")
```

### Q2: 训练不稳定？
- 调低学习率：`learning_rate = 1e-4`
- 增加批次大小：`batch_size = 64`
- 梯度裁剪：已在代码中实现

### Q3: 收敛太慢？
- 减少 episodes：`num_episodes = 200`
- 调整想象视野：`imagination_horizon = 10`

### Q4: 内存不足？
- 减小缓冲区：`buffer_size = 1000`
- 减小批次：`batch_size = 8`

## 📚 参考资料

### 论文
1. **DQN**: Mnih et al. (2015) - "Human-level control through deep RL"
2. **World Models**: Ha & Schmidhuber (2018) - arXiv:1803.10122
3. **Dreamer**: Hafner et al. (2020) - arXiv:1912.01603

### 代码
- [官方 Dreamer 实现](https://github.com/danijar/dreamer)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

## 💡 扩展方向

### 环境扩展
- [ ] Pendulum-v1 (连续控制)
- [ ] MountainCar-v0 (稀疏奖励)
- [ ] LunarLander-v2 (复杂任务)

### 算法改进
- [ ] Prioritized Experience Replay
- [ ] DreamerV2 离散潜在空间
- [ ] Model Ensemble (不确定性估计)

### 分析工具
- [ ] 潜在空间可视化 (t-SNE)
- [ ] 世界模型预测质量分析
- [ ] 样本效率曲线置信区间

---

**最后更新**: 2025-12-08  
**环境**: Python 3.10+, PyTorch 2.0+, MacBook MPS
