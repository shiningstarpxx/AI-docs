# Project Context

## Purpose

AI学习知识库 (tech_blog/AI-docs)，用于记录和整理AI领域的深度学习研究。

**主要目标：**
- **知识沉淀**：系统性记录AI重要概念、理论和技术细节
- **深度思考**：对AI技术发展脉络进行分析和思辨
- **实践总结**：通过对比实验验证理论分析
- **知识分享**：为AI学习者提供参考资料和技术分享演示

## Tech Stack

### 文档与演示
- **Markdown**：主要文档格式，包含数学公式 (LaTeX)
- **Marp**：基于Markdown的演示文稿生成工具 (`marp-cli`)

### 实验代码
- **Python 3.x**：实验脚本语言
- **PyTorch**：深度学习框架，支持MPS (Apple Silicon)
- **Gymnasium**：强化学习环境 (OpenAI Gym继任者)
- **NumPy**：数值计算
- **Matplotlib**：数据可视化

### 开发环境
- **平台**：macOS (Apple Silicon M系列芯片)
- **GPU加速**：PyTorch MPS backend
- **编辑器**：VS Code

## Project Conventions

### Code Style

**Python代码规范：**
- 遵循PEP 8风格指南
- 使用描述性变量名和函数名
- 添加docstring说明函数用途和参数
- 实验脚本以数字前缀命名以表示执行顺序 (如 `1_baseline_dqn.py`)

**文档命名规范：**
- 研究主题目录使用英文小写加下划线 (如 `world_models/`, `rnn_transformer_mamba/`)
- 文档文件可使用数字前缀表示阅读顺序 (如 `01_concept.md`, `02_math.md`)
- 研究计划文件统一命名为 `research_plan.md` 或 `learning_plan.md`

**语言规范：**
- 文档主体使用中文撰写
- 技术术语保留英文原文 (如 Transformer, Attention, MoE)
- 代码注释使用英文

### Architecture Patterns

**目录结构模式：**
```
topic_name/
├── research_plan.md      # 研究计划与学习路线
├── 01_concept.md         # 基础概念
├── 02_math.md            # 数学推导
├── ...                   # 按主题细分的文档
├── presentation.md       # Marp格式的分享演示
└── experiments/          # 实验代码目录
    ├── 1_baseline.py     # 基线方法
    ├── 2_improved.py     # 改进方法
    └── compare_results.py # 对比分析
```

**研究方法论：**
- **历史演进视角**：追溯技术发展脉络，理解Why而非仅仅How
- **对比实验设计**：baseline vs improved方法，关注sample efficiency等关键指标
- **理论实践结合**：数学推导 + 代码实现 + 实验验证

### Testing Strategy

**实验验证方式：**
- 在标准环境 (如CartPole-v1) 上进行对比实验
- 记录关键指标：sample efficiency、reward曲线、训练稳定性
- 使用matplotlib生成可视化对比图
- 实验结果保存为图片和日志文件

### Git Workflow

**分支策略：**
- `main`：主分支，保存稳定的文档和代码
- 功能分支：用于开发新的研究主题或实验

**提交规范：**
- 使用中文提交信息，简洁描述变更内容
- 示例格式：`添加MoE历史演进研究计划`、`完善World Models数学推导`

## Domain Context

**当前研究主题：**

1. **序列建模演进** (`rnn_transformer_mamba/`)
   - RNN → LSTM/GRU → Transformer → Linear Attention → Mamba/SSM
   - 核心关注：注意力机制、并行化、长序列建模效率

2. **Mixture of Experts** (`MoE/`)
   - 历史演进：Jacobs 1991 → Shazeer 2017 → Switch 2021 → Mixtral 2024
   - 核心关注：稀疏路由、负载均衡、训练稳定性、推理效率

3. **World Models** (`world_models/`)
   - 模型基础强化学习 (Model-Based RL)
   - 核心架构：VAE + RNN-MDN、Dreamer系列
   - 核心关注：环境建模、想象力训练、sample efficiency

**关键技术术语：**
- SSM: State Space Model
- MoE: Mixture of Experts
- MDN: Mixture Density Network
- VAE: Variational Autoencoder
- RSSM: Recurrent State Space Model

## Important Constraints

- **硬件限制**：MacBook (Apple Silicon)，无NVIDIA GPU，使用MPS加速
- **实验规模**：以教学和理解为目的的小规模实验，非生产级训练
- **文档优先**：重点在于知识整理和理解，代码实现为辅助验证手段

## External Dependencies

### Python包
```bash
pip install torch gymnasium numpy matplotlib
```

### 演示工具
```bash
npm install -g @marp-team/marp-cli
```

### 参考资源
- 原始论文 (存放于 `papers/` 目录)
- GitHub开源实现 (用于学习参考)
- 相关技术博客和教程
