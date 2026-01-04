# Scaling Law V2.0 快速参考卡

## 🚀 一键启动

```bash
cd /Users/peixingxin/code/tech_blog/scaling_law
./run_experiments.sh
```

选择 **2) Quick V2 (2-3 小时)** - 推荐首次运行

---

## 📊 三种运行方式

### 1️⃣ 快速演示 (1 分钟) - 模拟数据

```bash
python quick_scaling_demo.py
open scaling_demo/scaling_laws_with_theory.png
```

✅ 理论完美  
❌ 非真实训练

---

### 2️⃣ Quick V2 (2-3 小时) - 真实训练 🆕 推荐

```bash
nohup python run_scaling_experiments_enhanced.py --mode quick \
  > experiment_v2.log 2>&1 &

tail -f experiment_v2.log
open scaling_results_quick_v2/scaling_laws_complete.png
```

✅ 3000 步充分收敛  
✅ 4 子图完整展示  
✅ 外推到 GPT-4

---

### 3️⃣ 完整对比 (3-4 小时)

```bash
# Step 1: 快速演示
python quick_scaling_demo.py

# Step 2: 真实训练  
python run_scaling_experiments_enhanced.py --mode quick

# Step 3: 对比分析
python compare_quick_vs_real.py
open comparison_results/quick_vs_real_comparison.png
```

---

## 🎨 可视化说明

### 2x2 完整布局

```
┌─────────────────────────┬─────────────────────────┐
│ 参数 Scaling + 外推      │ 数据 Scaling + 外推      │
│ • 实验点 + 拟合          │ • 实验点 + 拟合          │
│ • 外推到 GPT-4 (1.8T)    │ • 外推到 Llama 3 (15T)   │
│ • Kaplan 理论对比        │ • Hestness 理论对比      │
└─────────────────────────┴─────────────────────────┘
┌─────────────────────────┬─────────────────────────┐
│ 训练曲线 (参数维度)       │ 训练曲线 (数据维度)       │
│ • 5M, 20M, 50M 对比      │ • 10M, 50M, 100M 对比    │
│ • 观察收敛速度            │ • 验证数据 Scaling       │
└─────────────────────────┴─────────────────────────┘
```

---

## 📈 预期结果

### V1 vs V2

| 指标 | V1 (1000步) | V2 (3000步) |
|------|-----------|-----------|
| 5M Loss | 9.22 → 9.22 ❌ | 9.22 → 3.8 ✅ |
| 20M Loss | 9.22 → 9.22 ❌ | 9.22 → 3.0 ✅ |
| 50M Loss | 9.22 → 9.22 ❌ | 9.22 → 2.5 ✅ |
| R² | N/A ❌ | > 0.95 ✅ |

---

## 🛠️ 故障排除

### Loss 没有下降?

```bash
# 增加步数
python run_scaling_experiments_enhanced.py --mode quick --max-steps 5000
```

### 拟合质量差?

```bash
# 更多数据点
python run_scaling_experiments_enhanced.py --mode standard
```

### 训练太慢?

```bash
# 减少实验点
# 修改脚本: n_params_list=[5e6, 50e6]
```

---

## 📚 必读文档

| 文档 | 用途 |
|------|------|
| `README_V2.md` | 项目总览 |
| `EXPERIMENT_GUIDE_V2.md` | 实验指南 |
| `UPDATE_LOG.md` | 更新记录 |
| `V2_COMPLETION_SUMMARY.md` | 完成总结 |

---

## ✅ 快速检查

开始前:
```bash
# 检查 Python
python3 --version  # 需要 3.10+

# 检查 MPS
python3 -c "import torch; print(torch.backends.mps.is_available())"

# 检查磁盘
df -h .  # 需要 > 5GB
```

---

## 🎯 核心命令

### 后台运行
```bash
nohup python run_scaling_experiments_enhanced.py --mode quick \
  > experiment_v2.log 2>&1 &
```

### 监控进度
```bash
tail -f experiment_v2.log
```

### 查看进程
```bash
ps aux | grep run_scaling_experiments_enhanced
```

### 终止进程
```bash
kill <PID>
```

---

## 📊 实验模式

| 模式 | 步数 | 点数 | 时间 | 适用 |
|------|-----|-----|------|------|
| Quick | 3000 | 6 | 2-3h | 验证 |
| Standard | 5000 | 10 | 6-8h | 研究 |
| Full | 8000 | 13 | 1.5-2d | 论文 |

---

## 🔥 推荐工作流

### Day 1: 快速验证 (3h)
```bash
# 1. 快速演示 (1分钟)
python quick_scaling_demo.py

# 2. Quick V2 (2-3小时)
./run_experiments.sh  # 选择 2

# 3. 查看结果
open scaling_results_quick_v2/scaling_laws_complete.png
```

### Day 2: 深入分析 (2h)
```bash
# 1. 对比分析
python compare_quick_vs_real.py

# 2. 阅读 Kaplan/Chinchilla 论文

# 3. 撰写学习总结
```

### Day 3: 完整实验 (8h)
```bash
# Standard 模式 (更精确)
python run_scaling_experiments_enhanced.py --mode standard
```

---

## 🎓 学习目标

- [ ] 理解幂律关系: `L ∝ N^(-α)`
- [ ] 验证 Scaling Law 存在
- [ ] 掌握外推预测方法
- [ ] 理解 Chinchilla 最优配比

---

## 🌟 V2.0 核心亮点

✨ 训练充分收敛 (Loss 9.22 → 2-3)  
📊 2x2 完整可视化  
🚀 外推预测到 GPT-4  
🔥 Warmup + Cosine Decay  
✅ R² > 0.95 高质量拟合

---

## 📞 快速链接

- **启动**: `./run_experiments.sh`
- **文档**: `EXPERIMENT_GUIDE_V2.md`
- **问题**: 查看 `UPDATE_LOG.md` 问题修复部分

---

**版本**: V2.0  
**状态**: ✅ 就绪  
**推荐**: Quick V2 (选项 2)

**立即开始**: `./run_experiments.sh`
