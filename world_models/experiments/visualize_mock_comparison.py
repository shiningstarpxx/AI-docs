"""
使用模拟数据展示训练前后对比
============================
生成四面板展示图，说明期望效果
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec


def create_cartpole_visualization():
    """
    创建 CartPole 四场景对比图
    """
    
    print("=" * 60)
    print("🎨 生成训练前后对比可视化 (模拟)")
    print("=" * 60)
    print()
    
    # 创建 2×2 网格
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.25)
    
    # ========== 定义四个场景 ==========
    scenarios = [
        {
            "title": "1️⃣ 训练前 (随机策略)",
            "reward": 23.0,
            "steps": 23,
            "angle": 25,  # 杆子倾斜角度（度）
            "cart_pos": 0.5,  # 小车位置
            "description": "随机选择动作\n杆子很快倒下",
            "color": "red",
            "trajectory": [(0.5, 0), (0.6, 5), (0.75, 15), (0.9, 25)],  # (cart_pos, angle)
        },
        {
            "title": "2️⃣ DQN 训练后",
            "reward": 491.4,
            "steps": 500,
            "angle": 3,
            "cart_pos": 0.3,
            "description": "学会基本平衡\n偶尔小幅摆动",
            "color": "blue",
            "trajectory": [(0.5, 0), (0.45, 2), (0.35, -1), (0.3, 3)],
        },
        {
            "title": "3️⃣ Simple WM 训练后",
            "reward": 477.7,
            "steps": 500,
            "angle": -2,
            "cart_pos": 0.6,
            "description": "在'梦境'中学习\n样本效率高 4×",
            "color": "green",
            "trajectory": [(0.5, 0), (0.55, -1), (0.62, 1), (0.6, -2)],
        },
        {
            "title": "4️⃣ Mini Dreamer 训练后",
            "reward": 503.0,
            "steps": 500,
            "angle": 1,
            "cart_pos": 0.45,
            "description": "最优性能\nRSSM 双路径设计",
            "color": "purple",
            "trajectory": [(0.5, 0), (0.48, 0.5), (0.46, -0.5), (0.45, 1)],
        },
    ]
    
    # ========== 绘制四个场景 ==========
    axes = []
    for idx, scenario in enumerate(scenarios):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        
        # 绘制轨道
        track_y = 0.3
        ax.plot([0, 1], [track_y, track_y], 'k-', linewidth=4, label='轨道')
        
        # 绘制小车
        cart_width = 0.08
        cart_height = 0.05
        cart_x = scenario["cart_pos"] - cart_width / 2
        cart_y = track_y
        
        cart = patches.Rectangle(
            (cart_x, cart_y), cart_width, cart_height,
            linewidth=2, edgecolor=scenario["color"], facecolor=scenario["color"], alpha=0.7
        )
        ax.add_patch(cart)
        
        # 绘制杆子
        pole_length = 0.25
        angle_rad = np.radians(scenario["angle"])
        pole_end_x = scenario["cart_pos"] + pole_length * np.sin(angle_rad)
        pole_end_y = cart_y + cart_height + pole_length * np.cos(angle_rad)
        
        ax.plot(
            [scenario["cart_pos"], pole_end_x],
            [cart_y + cart_height, pole_end_y],
            'o-', linewidth=4, markersize=8,
            color=scenario["color"], label='杆子'
        )
        
        # 绘制历史轨迹（淡化）
        for i, (pos, angle) in enumerate(scenario["trajectory"][:-1]):
            alpha = 0.1 + 0.2 * (i / len(scenario["trajectory"]))
            angle_rad = np.radians(angle)
            pole_end_x = pos + pole_length * 0.7 * np.sin(angle_rad)
            pole_end_y = cart_y + cart_height + pole_length * 0.7 * np.cos(angle_rad)
            
            ax.plot(
                [pos, pole_end_x],
                [cart_y + cart_height, pole_end_y],
                '-', linewidth=2, alpha=alpha, color=scenario["color"]
            )
        
        # 设置坐标轴
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 0.8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 标题
        title_color = 'green' if scenario["reward"] >= 450 else ('orange' if scenario["reward"] >= 200 else 'red')
        title_weight = 'bold' if scenario["reward"] >= 450 else 'normal'
        
        ax.set_title(
            scenario["title"],
            fontsize=16, fontweight=title_weight, color=title_color, pad=10
        )
        
        # 性能指标（文本框）
        info_text = f"总奖励: {scenario['reward']:.1f}\n"
        info_text += f"持续步数: {scenario['steps']}\n"
        info_text += f"━━━━━━━━━━━━━━━\n"
        info_text += scenario["description"]
        
        ax.text(
            0.5, 0.05, info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='bottom',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # 边框
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color(scenario["color"])
            ax.spines[spine].set_linewidth(3)
            ax.spines[spine].set_visible(True)
    
    # ========== 底部：性能对比表 ==========
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # 表格数据
    table_data = [
        ["场景", "总奖励", "持续步数", "样本效率 vs DQN", "关键技术"],
        ["训练前 (随机)", "23.0", "23", "—", "无策略"],
        ["DQN 训练后", "491.4 ⭐", "500", "1.0×  (Baseline)", "Q-Learning + 经验回放"],
        ["Simple WM", "477.7 ⭐", "500", "4.2×  ⬆️⬆️⬆️", "LSTM 世界模型 + CMA-ES"],
        ["Mini Dreamer", "503.0 ⭐⭐", "500 (满分)", "1.7×  ⬆️", "RSSM + Actor-Critic in 想象"],
    ]
    
    # 绘制表格
    table = ax_table.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.2, 0.25]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 样式化表头
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # 样式化数据行
    colors = ['#FFCDD2', '#BBDEFB', '#C8E6C9', '#E1BEE7']
    for i, color in enumerate(colors, start=1):
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    # 主标题
    fig.suptitle(
        'CartPole-v1: 训练前后效果对比\n(模拟演示 - 说明期望效果)',
        fontsize=20, fontweight='bold', y=0.98
    )
    
    # 保存
    plt.savefig('before_after_comparison_mock.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 对比图已保存: before_after_comparison_mock.png")
    print()
    
    # ========== 生成训练曲线对比 ==========
    print("📈 生成训练曲线对比...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 模拟训练曲线
    episodes = np.arange(500)
    
    # DQN: 逐步提升，有波动
    dqn_curve = 100 * (1 - np.exp(-episodes / 80)) + 400 * (1 - np.exp(-episodes / 200))
    dqn_curve += np.random.normal(0, 30, 500)
    dqn_curve = np.clip(dqn_curve, 0, 500)
    
    # Simple WM: 快速提升，但略低于 DQN
    swm_curve = 450 * (1 - np.exp(-episodes / 30))
    swm_curve += np.random.normal(0, 20, 500)
    swm_curve = np.clip(swm_curve, 0, 500)
    
    # Mini Dreamer: 较快提升，最终最高
    dreamer_curve = 100 * (1 - np.exp(-episodes / 50)) + 410 * (1 - np.exp(-episodes / 120))
    dreamer_curve += np.random.normal(0, 25, 500)
    dreamer_curve = np.clip(dreamer_curve, 0, 500)
    
    # ========== 子图1: 学习曲线 ==========
    ax = axes[0]
    
    # 平滑
    window = 20
    dqn_smooth = np.convolve(dqn_curve, np.ones(window)/window, mode='valid')
    swm_smooth = np.convolve(swm_curve, np.ones(window)/window, mode='valid')
    dreamer_smooth = np.convolve(dreamer_curve, np.ones(window)/window, mode='valid')
    
    episodes_smooth = episodes[window-1:]
    
    ax.plot(episodes_smooth, dqn_smooth, label='DQN', linewidth=2.5, color='blue')
    ax.plot(episodes_smooth, swm_smooth, label='Simple WM', linewidth=2.5, color='green')
    ax.plot(episodes_smooth, dreamer_smooth, label='Mini Dreamer', linewidth=2.5, color='purple')
    
    ax.axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='最大分数')
    ax.axhline(y=450, color='orange', linestyle='--', alpha=0.5, label='收敛阈值')
    
    ax.set_xlabel('训练 Episodes', fontsize=12)
    ax.set_ylabel('平均奖励 (20-episode moving avg)', fontsize=12)
    ax.set_title('学习曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 550)
    
    # ========== 子图2: 收敛速度 ==========
    ax = axes[1]
    
    methods = ['DQN', 'Simple WM', 'Mini Dreamer']
    convergence = [422, 100, 245]  # episodes to convergence
    colors = ['blue', 'green', 'purple']
    
    bars = ax.bar(methods, convergence, color=colors, alpha=0.7, width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)} ep',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('收敛所需 Episodes', fontsize=12)
    ax.set_title('收敛速度 (越低越好)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 500)
    
    # ========== 子图3: 样本效率 ==========
    ax = axes[2]
    
    efficiency = [1.0, 4.2, 1.7]  # vs DQN
    bars = ax.bar(methods, efficiency, color=colors, alpha=0.7, width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}×',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='DQN Baseline')
    ax.set_ylabel('样本效率倍数 (vs DQN)', fontsize=12)
    ax.set_title('样本效率对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig('training_curves_comparison_mock.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 训练曲线对比已保存: training_curves_comparison_mock.png")
    print()
    
    # ========== 生成说明文档 ==========
    print("📝 生成说明文档...")
    
    readme = """# CartPole-v1 训练前后效果对比

## 📊 可视化说明

### 1️⃣ 训练前 (随机策略)
- **表现**: 杆子快速倒下，只能维持 ~23 步
- **原因**: 没有策略，随机选择动作
- **奖励**: 23.0

### 2️⃣ DQN 训练后 (Model-Free RL)
- **表现**: 学会基本平衡，能维持到满分 500 步
- **收敛**: ~422 episodes
- **样本效率**: 1.0× (baseline)
- **奖励**: 491.4 ⭐

### 3️⃣ Simple World Model 训练后
- **表现**: 性能接近 DQN，但训练快得多
- **收敛**: ~100 episodes (**4.2× faster!**)
- **关键技术**: 
  - LSTM 预测环境动态
  - 在"梦境"中训练策略（无需真实环境交互）
  - CMA-ES 进化算法
- **奖励**: 477.7 ⭐

### 4️⃣ Mini Dreamer 训练后
- **表现**: 最优性能，超过理论最大分数
- **收敛**: ~245 episodes (1.7× faster)
- **关键技术**:
  - RSSM 双路径设计（确定性 + 随机性）
  - Actor-Critic 在潜在空间学习
  - 在线持续改进
- **奖励**: 503.0 ⭐⭐

## 🔑 核心洞察

### 样本效率排名
1. **Simple WM**: 4.2× vs DQN ⭐⭐⭐
2. **Mini Dreamer**: 1.7× vs DQN ⭐
3. **DQN**: Baseline

### 最终性能排名
1. **Mini Dreamer**: 503.0 ⭐⭐⭐
2. **DQN**: 491.4 ⭐⭐
3. **Simple WM**: 477.7 ⭐

### 为什么世界模型更高效？

```
传统 RL (DQN):
每次动作 → 真实环境交互 → 获得反馈
成本高 | 速度慢 | 样本效率低

世界模型 (Simple WM / Dreamer):
1. 收集少量真实数据 (10k-20k steps)
2. 训练环境模型（学习动态）
3. 在"梦境"中无限训练策略
成本低 | 速度快 | 样本效率高 ✅
```

## 🚀 如何运行真实实验

```bash
# 1. 安装依赖
pip install torch gymnasium numpy matplotlib

# 2. 运行实验
python 1_baseline_dqn.py           # ~30 分钟
python 2_simple_world_model.py     # ~40 分钟
python 3_mini_dreamer.py           # ~45 分钟

# 3. 生成对比
python visualize_before_after.py  # 需要训练好的模型
python compare_results.py          # 定量对比
```

## 📈 预期结果

| 方法 | 训练时间 | 收敛 Episodes | 样本效率 | 最终奖励 |
|:---|:---|:---|:---|:---|
| DQN | 30 分钟 | ~422 | 1.0× | ~491 |
| Simple WM | 40 分钟 | ~100 | **4.2×** ⬆️ | ~478 |
| Mini Dreamer | 45 分钟 | ~245 | **1.7×** ⬆️ | ~503 ⭐ |

## 💡 注意事项

**当前图像是模拟演示**，用于说明期望效果。

要查看真实训练效果：
1. 运行上述实验脚本
2. 等待训练完成（约 2 小时）
3. 重新运行可视化脚本

**为什么之前运行很快？**
- 使用的是模拟数据（随机生成）
- 没有真实训练神经网络
- 仅用于演示对比分析流程

**真实训练需要时间**：
- DQN: ~30 分钟（500 episodes × ~3秒/episode）
- Simple WM: ~40 分钟（数据收集 + 模型训练 + 进化）
- Mini Dreamer: ~45 分钟（在线学习 + RSSM 训练）

## 🎯 学习价值

通过这个对比实验，你将：
1. ✅ 直观理解世界模型的优势
2. ✅ 掌握样本效率的重要性
3. ✅ 对比不同方法的权衡
4. ✅ 学习前沿 RL 技术（RSSM、Actor-Critic in Imagination）

---

**项目**: World Models Evolution Study
**环境**: CartPole-v1 (OpenAI Gymnasium)
**硬件**: MacBook (MPS 加速)
"""
    
    with open("VISUALIZATION_README.md", "w") as f:
        f.write(readme)
    
    print("✓ 说明文档已保存: VISUALIZATION_README.md")
    print()
    
    # ========== 总结 ==========
    print("=" * 60)
    print("✅ 可视化完成！")
    print("=" * 60)
    print()
    print("📁 生成的文件:")
    print("  1. before_after_comparison_mock.png - 四面板对比图")
    print("  2. training_curves_comparison_mock.png - 训练曲线")
    print("  3. VISUALIZATION_README.md - 详细说明")
    print()
    print("💡 当前是模拟演示，展示期望效果")
    print()
    print("🚀 要查看真实效果，请:")
    print("  1. 安装依赖: pip install torch gymnasium numpy matplotlib")
    print("  2. 运行训练: python 1_baseline_dqn.py")
    print("  3. 真实可视化: python visualize_before_after.py")
    print()


if __name__ == "__main__":
    create_cartpole_visualization()
