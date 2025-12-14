"""
评估训练好的模型
================
在无探索模式下测试真实性能
"""

import torch
import gymnasium as gym
import numpy as np
import json
from pathlib import Path


def evaluate_dqn(model_path, num_episodes=10):
    """评估 DQN 模型"""
    import torch.nn as nn
    
    class DQN(nn.Module):
        def __init__(self, state_dim=4, action_dim=2, hidden_size=256):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim)
            )
        
        def forward(self, state):
            return self.network(state)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    env = gym.make("CartPole-v1")
    rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(500):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax(1).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        print(f"  Episode {ep+1}/{ num_episodes}: {episode_reward:.1f}")
    
    env.close()
    
    return {
        "mean": np.mean(rewards),
        "std": np.std(rewards),
        "min": np.min(rewards),
        "max": np.max(rewards),
        "all": rewards
    }


def main():
    """主函数"""
    print("=" * 60)
    print("🎯 评估训练好的模型（无探索模式）")
    print("=" * 60)
    print()
    
    results = {}
    
    # 评估 DQN
    dqn_model_path = Path("./results_dqn/model_final.pth")
    if dqn_model_path.exists():
        print("📊 评估 DQN...")
        dqn_results = evaluate_dqn(dqn_model_path, num_episodes=20)
        results["dqn"] = dqn_results
        print(f"  ✓ DQN 平均奖励: {dqn_results['mean']:.1f} ± {dqn_results['std']:.1f}")
        print(f"    范围: [{dqn_results['min']:.0f}, {dqn_results['max']:.0f}]")
        print()
    else:
        print("  ⚠️ DQN 模型未找到")
        print()
    
    # 保存评估结果
    if results:
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("=" * 60)
        print("✅ 评估完成！")
        print("=" * 60)
        print()
        print(f"📁 结果已保存: evaluation_results.json")
        print()
        
        # 总结
        print("📊 性能总结:")
        for method, res in results.items():
            print(f"  {method.upper()}: {res['mean']:.1f} ± {res['std']:.1f}")
        print()


if __name__ == "__main__":
    main()
