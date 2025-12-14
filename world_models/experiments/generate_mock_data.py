"""
生成模拟实验数据用于演示对比
================================
"""

import json
import numpy as np
import os

# 设置随机种子
np.random.seed(42)

def generate_dqn_data():
    """生成 DQN 模拟数据"""
    episodes = 500
    
    # DQN: 较慢收敛，样本效率低
    episode_rewards = []
    episode_lengths = []
    
    for i in range(episodes):
        # 逐步提升，加入噪声
        base_reward = min(500, 50 + i * 0.9)
        noise = np.random.normal(0, 50)
        reward = max(10, base_reward + noise)
        episode_rewards.append(reward)
        
        # Episode 长度
        length = int(min(500, reward))
        episode_lengths.append(length)
    
    loss_history = np.random.exponential(0.1, 10000).tolist()
    
    data = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "loss_history": loss_history,
        "total_steps": sum(episode_lengths)
    }
    
    return data


def generate_simple_wm_data():
    """生成 Simple World Model 模拟数据"""
    
    # 数据收集阶段（随机策略）
    data_collection_rewards = []
    for i in range(100):
        reward = np.random.uniform(20, 150)
        data_collection_rewards.append(reward)
    
    # 世界模型训练损失
    world_model_losses = []
    for i in range(50):
        loss = 2.0 * np.exp(-i * 0.05) + np.random.normal(0, 0.1)
        world_model_losses.append(max(0.01, loss))
    
    # 策略进化适应度（在梦境中）
    policy_fitness = []
    for i in range(100):
        fitness = min(500, 100 + i * 4.0 + np.random.normal(0, 30))
        policy_fitness.append(fitness)
    
    # 最终真实环境评估
    evaluation_rewards = [480]
    
    data = {
        "data_collection_rewards": data_collection_rewards,
        "world_model_losses": world_model_losses,
        "policy_fitness": policy_fitness,
        "evaluation_rewards": evaluation_rewards
    }
    
    return data


def generate_mini_dreamer_data():
    """生成 Mini Dreamer 模拟数据"""
    episodes = 300
    
    # Mini Dreamer: 快速收敛，样本效率高
    episode_rewards = []
    
    for i in range(episodes):
        # 快速提升
        base_reward = min(500, 100 + i * 1.5)
        noise = np.random.normal(0, 30)
        reward = max(20, base_reward + noise)
        episode_rewards.append(reward)
    
    # 世界模型损失
    world_model_losses = []
    for i in range(1500):
        loss = 1.5 * np.exp(-i * 0.003) + np.random.normal(0, 0.05)
        world_model_losses.append(max(0.01, loss))
    
    # Actor 损失
    actor_losses = []
    for i in range(1500):
        loss = -5.0 + i * 0.01 + np.random.normal(0, 1.0)
        actor_losses.append(loss)
    
    # Critic 损失
    critic_losses = []
    for i in range(1500):
        loss = 10.0 * np.exp(-i * 0.002) + np.random.normal(0, 0.5)
        critic_losses.append(max(0.1, loss))
    
    data = {
        "episode_rewards": episode_rewards,
        "world_model_losses": world_model_losses,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses
    }
    
    return data


def main():
    """生成所有模拟数据"""
    print("🎲 生成模拟实验数据...")
    print("-" * 50)
    
    # 创建目录
    os.makedirs("./results_dqn", exist_ok=True)
    os.makedirs("./results_simple_wm", exist_ok=True)
    os.makedirs("./results_mini_dreamer", exist_ok=True)
    
    # 生成 DQN 数据
    print("📊 生成 DQN 数据...")
    dqn_data = generate_dqn_data()
    with open("./results_dqn/training_data.json", "w") as f:
        json.dump(dqn_data, f, indent=2)
    print(f"   - Episodes: {len(dqn_data['episode_rewards'])}")
    print(f"   - Total Steps: {dqn_data['total_steps']:,}")
    print(f"   - Final Reward: {np.mean(dqn_data['episode_rewards'][-10:]):.1f}")
    
    # 生成 Simple WM 数据
    print("\n📊 生成 Simple World Model 数据...")
    swm_data = generate_simple_wm_data()
    with open("./results_simple_wm/training_history.json", "w") as f:
        json.dump(swm_data, f, indent=2)
    print(f"   - Data Collection: {len(swm_data['data_collection_rewards'])} episodes")
    print(f"   - Policy Generations: {len(swm_data['policy_fitness'])}")
    print(f"   - Final Eval: {swm_data['evaluation_rewards'][0]:.1f}")
    
    # 生成 Mini Dreamer 数据
    print("\n📊 生成 Mini Dreamer 数据...")
    dreamer_data = generate_mini_dreamer_data()
    with open("./results_mini_dreamer/training_data.json", "w") as f:
        json.dump(dreamer_data, f, indent=2)
    print(f"   - Episodes: {len(dreamer_data['episode_rewards'])}")
    print(f"   - Final Reward: {np.mean(dreamer_data['episode_rewards'][-10:]):.1f}")
    
    print("\n✅ 模拟数据生成完成！")
    print("\n现在可以运行: python3 compare_results.py")


if __name__ == "__main__":
    main()
