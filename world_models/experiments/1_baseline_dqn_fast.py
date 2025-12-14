"""
CartPole-v1 DQN - 快速测试版
===========================
优化参数确保快速收敛到 450+
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
import json
from pathlib import Path

# ========== 配置 ==========
class Config:
    # 环境
    env_name = "CartPole-v1"
    
    # 训练（优化版）
    episodes = 800  # 增加训练轮数
    max_steps = 500
    batch_size = 64
    gamma = 0.99
    
    # 探索（关键优化）
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.9985  # 更慢的衰减，保证充分探索
    
    # 网络
    hidden_size = 256  # 增大网络容量
    learning_rate = 1e-4  # 更小的学习率
    
    # 经验回放
    buffer_size = 20000  # 增大 buffer
    min_buffer_size = 1000  # 最小 buffer 再开始训练
    target_update_freq = 5  # 更频繁更新目标网络
    
    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 日志
    log_interval = 10
    save_dir = "./results_dqn"
    
    # 早停（收敛后停止）
    early_stop_threshold = 475  # 连续 N 次达到此分数
    early_stop_window = 20


# ========== DQN 网络 ==========
class DQN(nn.Module):
    """Q 网络"""
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # 额外一层
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
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# ========== DQN Agent ==========
class DQNAgent:
    """DQN 智能体"""
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.action_dim = action_dim
        
        # Q 网络和目标网络
        self.q_network = DQN(state_dim, action_dim, config.hidden_size).to(config.device)
        self.target_network = DQN(state_dim, action_dim, config.hidden_size).to(config.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # 探索参数
        self.epsilon = config.epsilon_start
        
    def select_action(self, state, training=True):
        """选择动作（ε-greedy）"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()
    
    def update(self):
        """更新 Q 网络"""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return 0.0
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # 转换为 tensor
        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).to(self.config.device)
        
        # 当前 Q 值
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标 Q 值（Double DQN）
        with torch.no_grad():
            # 用当前网络选择动作
            next_actions = self.q_network(next_states).argmax(1)
            # 用目标网络计算 Q 值
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_value = rewards + (1 - dones) * self.config.gamma * next_q_value
        
        # 计算损失
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)


# ========== 训练函数 ==========
def train():
    """训练 DQN"""
    config = Config()
    
    # 创建环境
    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建智能体
    agent = DQNAgent(state_dim, action_dim, config)
    
    # 记录
    episode_rewards = []
    episode_lengths = []
    moving_avg_rewards = []
    
    print(f"🚀 开始训练 DQN（优化版）on {config.device}")
    print(f"环境: {config.env_name}")
    print("=" * 50)
    
    total_steps = 0
    best_reward = 0
    convergence_count = 0
    
    for episode in range(1, config.episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(config.max_steps):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新网络
            if len(agent.replay_buffer) >= config.min_buffer_size:
                agent.update()
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            state = next_state
            
            if done:
                break
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 更新目标网络
        if episode % config.target_update_freq == 0:
            agent.update_target_network()
        
        # 记录
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 计算移动平均
        window = 10
        if len(episode_rewards) >= window:
            moving_avg = np.mean(episode_rewards[-window:])
            moving_avg_rewards.append(moving_avg)
        else:
            moving_avg_rewards.append(np.mean(episode_rewards))
        
        # 早停检查
        if episode_reward >= config.early_stop_threshold:
            convergence_count += 1
            if convergence_count >= config.early_stop_window:
                print(f"\n✅ 提前收敛！连续 {config.early_stop_window} 次达到 {config.early_stop_threshold}+")
                break
        else:
            convergence_count = 0
        
        # 记录最佳
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # 日志
        if episode % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval:])
            avg_length = np.mean(episode_lengths[-config.log_interval:])
            print(f"Episode {episode}/{config.episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {total_steps} | "
                  f"Best: {best_reward:.0f}")
    
    env.close()
    
    # 保存结果
    Path(config.save_dir).mkdir(exist_ok=True)
    
    # 保存模型
    torch.save(agent.q_network.state_dict(), f"{config.save_dir}/model_final.pth")
    
    # 保存训练数据
    training_data = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "moving_avg_rewards": moving_avg_rewards,
        "config": {
            "episodes": episode,
            "epsilon_decay": config.epsilon_decay,
            "learning_rate": config.learning_rate,
            "hidden_size": config.hidden_size,
        }
    }
    
    with open(f"{config.save_dir}/training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n✅ 训练完成！")
    print(f"📊 最终统计:")
    print(f"   - 总 Episodes: {episode}")
    print(f"   - 总步数: {total_steps}")
    print(f"   - 最佳奖励: {best_reward:.1f}")
    print(f"   - 最后10轮平均: {np.mean(episode_rewards[-10:]):.1f}")
    print(f"📊 结果已保存到: {config.save_dir}")
    
    return episode_rewards, episode_lengths


if __name__ == "__main__":
    train()
