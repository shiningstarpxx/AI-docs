"""
CartPole-v1 Baseline: DQN (Deep Q-Network)
==========================================
Model-Free RL baseline for comparison

实验目标：
- 建立样本效率基线
- 记录训练曲线
- 测试 MPS 加速效果
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime
import json

# ========== 配置 ==========
class Config:
    # 环境
    env_name = "CartPole-v1"
    
    # 训练
    episodes = 600  # 增加训练 episodes
    max_steps = 500
    batch_size = 64
    gamma = 0.99
    
    # 探索（优化：更慢的衰减）
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.998  # 从 0.995 改为 0.998，更慢衰减
    
    # 网络
    hidden_size = 128
    learning_rate = 3e-4  # 降低学习率，更稳定
    
    # 经验回放
    buffer_size = 10000
    target_update_freq = 10
    
    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 日志
    log_interval = 10
    save_dir = "./results_dqn"


# ========== DQN 网络 ==========
class DQN(nn.Module):
    """简单的全连接 Q 网络"""
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)


# ========== 经验回放 ==========
class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    
    def __len__(self):
        return len(self.buffer)


# ========== DQN Agent ==========
class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_name)
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        # Q 网络和目标网络
        self.q_network = DQN(state_dim, action_dim, config.hidden_size).to(config.device)
        self.target_network = DQN(state_dim, action_dim, config.hidden_size).to(config.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.buffer_size)
        
        self.epsilon = config.epsilon_start
        self.total_steps = 0
        
        # 记录
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
    
    def select_action(self, state, training=True):
        """ε-greedy 策略"""
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def update(self):
        """DQN 更新"""
        if len(self.buffer) < self.config.batch_size:
            return None
        
        # 采样批次
        state, action, reward, next_state, done = self.buffer.sample(self.config.batch_size)
        
        state = torch.FloatTensor(state).to(self.config.device)
        action = torch.LongTensor(action).to(self.config.device)
        reward = torch.FloatTensor(reward).to(self.config.device)
        next_state = torch.FloatTensor(next_state).to(self.config.device)
        done = torch.FloatTensor(done).to(self.config.device)
        
        # 当前 Q 值
        current_q = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        # 目标 Q 值
        with torch.no_grad():
            next_q = self.target_network(next_state).max(1)[0]
            target_q = reward + (1 - done) * self.config.gamma * next_q
        
        # 损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """训练循环"""
        print(f"🚀 开始训练 DQN on {self.config.device}")
        print(f"环境: {self.config.env_name}")
        print("-" * 50)
        
        for episode in range(self.config.episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.config.max_steps):
                # 选择动作
                action = self.select_action(state)
                
                # 环境交互
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 存储经验
                self.buffer.push(state, action, reward, next_state, done)
                
                # 更新网络
                loss = self.update()
                if loss is not None:
                    self.loss_history.append(loss)
                
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1
                
                state = next_state
                
                if done:
                    break
            
            # 更新目标网络
            if episode % self.config.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # 衰减 epsilon
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
            
            # 记录
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # 日志
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.config.log_interval:])
                print(f"Episode {episode+1}/{self.config.episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Steps: {self.total_steps}")
        
        print("\n✅ 训练完成！")
        self.save_results()
    
    def save_results(self):
        """保存结果"""
        import os
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # 保存模型
        torch.save(self.q_network.state_dict(), 
                   f"{self.config.save_dir}/dqn_model.pt")
        
        # 保存训练数据
        results = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "loss_history": self.loss_history,
            "total_steps": self.total_steps,
            "config": vars(self.config)
        }
        
        with open(f"{self.config.save_dir}/training_data.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # 绘图
        self.plot_results()
        
        print(f"📊 结果已保存到: {self.config.save_dir}")
    
    def plot_results(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 奖励曲线
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        window = 50
        if len(self.episode_rewards) >= window:
            smoothed = np.convolve(self.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.episode_rewards)), 
                           smoothed, label=f'Smoothed ({window})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode 长度
        axes[0, 1].plot(self.episode_lengths, alpha=0.3)
        if len(self.episode_lengths) >= window:
            smoothed = np.convolve(self.episode_lengths, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(self.episode_lengths)), smoothed)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True)
        
        # 损失曲线
        if self.loss_history:
            axes[1, 0].plot(self.loss_history, alpha=0.5)
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # 样本效率（累积奖励 vs 总步数）
        cumulative_rewards = np.cumsum(self.episode_rewards)
        cumulative_steps = np.cumsum(self.episode_lengths)
        axes[1, 1].plot(cumulative_steps, cumulative_rewards)
        axes[1, 1].set_xlabel('Total Environment Steps')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].set_title('Sample Efficiency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/training_curves.png", dpi=150)
        plt.close()


# ========== 主函数 ==========
def main():
    config = Config()
    agent = DQNAgent(config)
    agent.train()


if __name__ == "__main__":
    main()
