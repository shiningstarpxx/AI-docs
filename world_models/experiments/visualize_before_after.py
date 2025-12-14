"""
可视化训练前后的效果对比
========================
展示四个部分：
1. 训练前的随机策略 (Before Training)
2. DQN 训练后 (After DQN)
3. Simple WM 训练后 (After Simple WM)
4. Mini Dreamer 训练后 (After Mini Dreamer)
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import torch
import json


# ========== 渲染 Episode 到图像 ==========
def render_episode(env, policy=None, max_steps=500, title="Episode"):
    """
    运行一个 episode 并收集帧
    
    Args:
        env: Gym 环境
        policy: 策略函数 (state -> action)，None 表示随机策略
        max_steps: 最大步数
        title: 显示标题
    
    Returns:
        frames: 帧列表
        total_reward: 总奖励
        steps: 步数
    """
    frames = []
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # 渲染当前帧
        frame = env.render()
        frames.append(frame)
        
        # 选择动作
        if policy is None:
            action = env.action_space.sample()  # 随机策略
        else:
            action = policy(state)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    return frames, total_reward, step + 1


# ========== 加载训练好的模型 ==========
def load_dqn_policy(model_path):
    """加载 DQN 策略"""
    if not Path(model_path).exists():
        print(f"❌ 未找到模型: {model_path}")
        return None
    
    # 简化版 DQN 网络
    class DQN(torch.nn.Module):
        def __init__(self, state_dim=4, action_dim=2, hidden_size=128):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, action_dim)
            )
        
        def forward(self, x):
            return self.network(x)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    def policy(state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            return q_values.argmax(1).item()
    
    return policy


def load_simple_wm_policy(model_path):
    """加载 Simple WM 策略（线性）"""
    if not Path(model_path).exists():
        print(f"❌ 未找到模型: {model_path}")
        return None
    
    # 线性策略
    weights = np.load(model_path)
    
    def policy(state):
        # state: [4,] -> action: 0 or 1
        action_scores = state @ weights  # [2,]
        return int(np.argmax(action_scores))
    
    return policy


def load_mini_dreamer_policy(model_path):
    """加载 Mini Dreamer Actor"""
    if not Path(model_path).exists():
        print(f"❌ 未找到模型: {model_path}")
        return None
    
    # 简化版 Actor 网络
    class Actor(torch.nn.Module):
        def __init__(self, state_dim=4, action_dim=2, hidden_size=64):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, action_dim)
            )
        
        def forward(self, x):
            return self.network(x)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Actor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    def policy(state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits = model(state_tensor)
            action = torch.distributions.Categorical(logits=logits).sample()
            return action.item()
    
    return policy


# ========== 四面板可视化 ==========
def visualize_comparison(save_path="before_after_comparison.png"):
    """
    生成 2×2 网格对比图
    """
    print("=" * 60)
    print("🎬 生成训练前后对比可视化")
    print("=" * 60)
    print()
    
    # 创建环境（RGB 渲染模式）
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # ========== 1. 收集四个场景的数据 ==========
    scenarios = []
    
    # (1) 训练前：随机策略
    print("📹 场景 1/4: 训练前 (随机策略)...")
    frames_before, reward_before, steps_before = render_episode(
        env, policy=None, max_steps=500, title="Before Training"
    )
    scenarios.append({
        "title": "训练前 (随机策略)",
        "frames": frames_before,
        "reward": reward_before,
        "steps": steps_before,
        "color": "red"
    })
    print(f"  ✓ 奖励: {reward_before:.1f}, 步数: {steps_before}")
    print()
    
    # (2) DQN 训练后
    print("📹 场景 2/4: DQN 训练后...")
    dqn_policy = load_dqn_policy("./results_dqn/model_final.pth")
    if dqn_policy:
        frames_dqn, reward_dqn, steps_dqn = render_episode(
            env, policy=dqn_policy, max_steps=500, title="After DQN"
        )
        scenarios.append({
            "title": "DQN 训练后",
            "frames": frames_dqn,
            "reward": reward_dqn,
            "steps": steps_dqn,
            "color": "blue"
        })
        print(f"  ✓ 奖励: {reward_dqn:.1f}, 步数: {steps_dqn}")
    else:
        print("  ⚠️ 模型未找到，使用随机策略")
        scenarios.append(scenarios[0])  # 复用随机策略
    print()
    
    # (3) Simple WM 训练后
    print("📹 场景 3/4: Simple WM 训练后...")
    swm_policy = load_simple_wm_policy("./results_simple_wm/controller_best.npy")
    if swm_policy:
        frames_swm, reward_swm, steps_swm = render_episode(
            env, policy=swm_policy, max_steps=500, title="After Simple WM"
        )
        scenarios.append({
            "title": "Simple WM 训练后",
            "frames": frames_swm,
            "reward": reward_swm,
            "steps": steps_swm,
            "color": "green"
        })
        print(f"  ✓ 奖励: {reward_swm:.1f}, 步数: {steps_swm}")
    else:
        print("  ⚠️ 模型未找到，使用随机策略")
        scenarios.append(scenarios[0])
    print()
    
    # (4) Mini Dreamer 训练后
    print("📹 场景 4/4: Mini Dreamer 训练后...")
    dreamer_policy = load_mini_dreamer_policy("./results_mini_dreamer/actor_final.pth")
    if dreamer_policy:
        frames_dreamer, reward_dreamer, steps_dreamer = render_episode(
            env, policy=dreamer_policy, max_steps=500, title="After Mini Dreamer"
        )
        scenarios.append({
            "title": "Mini Dreamer 训练后",
            "frames": frames_dreamer,
            "reward": reward_dreamer,
            "steps": steps_dreamer,
            "color": "purple"
        })
        print(f"  ✓ 奖励: {reward_dreamer:.1f}, 步数: {steps_dreamer}")
    else:
        print("  ⚠️ 模型未找到，使用随机策略")
        scenarios.append(scenarios[0])
    print()
    
    env.close()
    
    # ========== 2. 创建 2×2 网格图 ==========
    print("🎨 生成对比图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 找到所有场景中的最大帧数（用于统一时间轴）
    max_frames = max(len(s["frames"]) for s in scenarios)
    
    for idx, (ax, scenario) in enumerate(zip(axes, scenarios)):
        # 显示中间帧（约一半位置）
        frame_idx = len(scenario["frames"]) // 2
        if frame_idx >= len(scenario["frames"]):
            frame_idx = len(scenario["frames"]) - 1
        
        frame = scenario["frames"][frame_idx]
        
        # 显示帧
        ax.imshow(frame)
        ax.axis('off')
        
        # 标题（包含性能指标）
        title_text = f"{scenario['title']}\n"
        title_text += f"总奖励: {scenario['reward']:.1f} | 持续步数: {scenario['steps']}"
        
        # 根据性能着色标题
        if scenario['reward'] >= 450:
            title_color = 'green'
            title_weight = 'bold'
        elif scenario['reward'] >= 200:
            title_color = 'orange'
            title_weight = 'normal'
        else:
            title_color = 'red'
            title_weight = 'normal'
        
        ax.set_title(title_text, fontsize=14, fontweight=title_weight, 
                     color=title_color, pad=10)
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_edgecolor(scenario['color'])
            spine.set_linewidth(3)
    
    plt.suptitle('CartPole-v1: 训练前后效果对比', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 对比图已保存: {save_path}")
    print()
    
    # ========== 3. 生成性能总结 ==========
    print("=" * 60)
    print("📊 性能总结")
    print("=" * 60)
    print()
    
    summary = "| 场景 | 总奖励 | 持续步数 | vs 随机策略 |\n"
    summary += "|:---|---:|---:|:---|\n"
    
    baseline_reward = scenarios[0]["reward"]
    
    for scenario in scenarios:
        reward = scenario["reward"]
        steps = scenario["steps"]
        improvement = ((reward - baseline_reward) / baseline_reward * 100) if baseline_reward > 0 else 0
        
        summary += f"| {scenario['title']} | {reward:.1f} | {steps} | "
        if improvement > 0:
            summary += f"+{improvement:.0f}% ✅ |\n"
        else:
            summary += "Baseline |\n"
    
    print(summary)
    
    # 保存总结到文件
    with open("performance_summary.md", "w") as f:
        f.write("# CartPole-v1 训练前后性能对比\n\n")
        f.write(summary)
        f.write("\n## 可视化\n\n")
        f.write(f"![对比图]({save_path})\n")
    
    print("✓ 总结已保存: performance_summary.md")
    print()
    
    return scenarios


# ========== 生成动画（可选）==========
def create_animation(scenarios, save_path="comparison_animation.gif", fps=30):
    """
    生成四面板动画 GIF
    """
    print("🎬 生成动画（这可能需要几分钟）...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 初始化图像
    images = []
    for ax, scenario in zip(axes, scenarios):
        img = ax.imshow(scenario["frames"][0])
        ax.axis('off')
        
        title_text = f"{scenario['title']}\n"
        title_text += f"总奖励: {scenario['reward']:.1f} | 步数: {scenario['steps']}"
        ax.set_title(title_text, fontsize=14, pad=10)
        
        images.append(img)
    
    plt.suptitle('CartPole-v1: 训练前后效果对比', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 找到最大帧数
    max_frames = max(len(s["frames"]) for s in scenarios)
    
    def update(frame):
        for img, scenario in zip(images, scenarios):
            # 如果该场景已结束，显示最后一帧
            idx = min(frame, len(scenario["frames"]) - 1)
            img.set_data(scenario["frames"][idx])
        return images
    
    anim = FuncAnimation(fig, update, frames=max_frames, 
                         interval=1000/fps, blit=True)
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"✓ 动画已保存: {save_path}")


# ========== 主函数 ==========
def main():
    """主函数"""
    
    # 生成静态对比图
    scenarios = visualize_comparison(save_path="before_after_comparison.png")
    
    # 询问是否生成动画（可选）
    print()
    print("💡 提示: 可以生成动画 GIF 查看完整过程")
    print("   但这需要较长时间（~5-10分钟）")
    print()
    
    # 如果需要动画，取消下面的注释
    # create_animation(scenarios, save_path="comparison_animation.gif", fps=30)
    
    print("=" * 60)
    print("✅ 可视化完成！")
    print("=" * 60)
    print()
    print("输出文件:")
    print("  - before_after_comparison.png (静态对比图)")
    print("  - performance_summary.md (性能总结)")
    print()
    print("💡 下一步:")
    print("  1. 查看对比图，了解训练效果")
    print("  2. 运行真实训练: python 1_baseline_dqn.py")
    print("  3. 重新运行此脚本查看真实效果")


if __name__ == "__main__":
    main()
