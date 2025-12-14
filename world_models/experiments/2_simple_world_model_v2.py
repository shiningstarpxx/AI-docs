"""
CartPole-v1: Simple World Model v2 (改进版)
================================
改进点：
1. 数据收集：用 ε-greedy DQN 代替纯随机
2. 世界模型：更大容量 (256 hidden, 2层LSTM)
3. 控制器：神经网络 + 梯度优化代替线性+CMA-ES
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import json

# ========== 配置 ==========
class Config:
    # 环境
    env_name = "CartPole-v1"
    
    # 阶段 1: 改进的数据收集
    pretrain_episodes = 100  # DQN 预训练
    data_collection_episodes = 200  # 用训练好的策略收集
    epsilon_start = 0.5  # ε-greedy
    epsilon_end = 0.1
    
    # 阶段 2: 更强的世界模型
    world_model_epochs = 200  # 更多训练
    batch_size = 64
    sequence_length = 20
    hidden_size = 256  # 64 → 256
    num_lstm_layers = 2  # 1 → 2
    learning_rate = 3e-4
    
    # 阶段 3: 神经网络控制器 + 梯度优化
    dream_training_steps = 5000  # 梯度优化步数
    dream_batch_size = 32
    dream_horizon = 50  # 想象长度
    controller_lr = 1e-3
    
    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 日志
    save_dir = "./results_simple_wm_v2"


# ========== 简单 DQN (用于数据收集) ==========
class SimpleDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def get_action(self, state, epsilon=0.0, device='cpu'):
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        with torch.no_grad():
            q_values = self.forward(torch.FloatTensor(state).to(device))
            return q_values.argmax().item()


# ========== 改进的世界模型 ==========
class ImprovedWorldModel(nn.Module):
    """
    更强的 LSTM 世界模型
    - 2层 LSTM
    - 256 hidden
    - Residual connections
    """
    def __init__(self, state_dim, action_dim, hidden_size=256, num_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入编码
        self.input_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 多层 LSTM
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # 输出解码
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
        self.reward_predictor = nn.Linear(hidden_size, 1)
        self.done_predictor = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action, hidden=None):
        # 编码
        x = torch.cat([state, action], dim=-1)
        x = self.input_encoder(x)
        
        # LSTM
        x, hidden = self.lstm(x, hidden)
        
        # 预测
        next_state = self.state_predictor(x)
        reward = self.reward_predictor(x)
        done = torch.sigmoid(self.done_predictor(x))
        
        return next_state, reward, done, hidden
    
    def imagine_step(self, state, action, hidden=None):
        """单步想象"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            
            state = state.unsqueeze(1)
            action = action.unsqueeze(1)
            
            next_state, reward, done, hidden = self.forward(state, action, hidden)
            return next_state.squeeze(1), reward.squeeze(1), done.squeeze(1), hidden


# ========== 神经网络控制器 ==========
class NeuralController(nn.Module):
    """
    神经网络策略（比线性强得多）
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
    
    def get_action(self, state, device='cpu', deterministic=False):
        with torch.no_grad():
            logits = self.forward(torch.FloatTensor(state).to(device))
            if deterministic:
                return logits.argmax().item()
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()


# ========== 改进的 Agent ==========
class ImprovedWorldModelAgent:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_name)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # DQN (用于数据收集)
        self.dqn = SimpleDQN(self.state_dim, self.action_dim).to(config.device)
        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), lr=1e-3)
        self.dqn_memory = deque(maxlen=10000)
        
        # 世界模型
        self.world_model = ImprovedWorldModel(
            self.state_dim,
            self.action_dim,
            config.hidden_size,
            config.num_lstm_layers
        ).to(config.device)
        self.wm_optimizer = optim.Adam(
            self.world_model.parameters(),
            lr=config.learning_rate
        )
        
        # 控制器
        self.controller = NeuralController(
            self.state_dim,
            self.action_dim
        ).to(config.device)
        self.controller_optimizer = optim.Adam(
            self.controller.parameters(),
            lr=config.controller_lr
        )
        
        # 数据
        self.trajectories = []
        
        # 记录
        self.training_history = {
            "dqn_pretrain_rewards": [],
            "data_collection_rewards": [],
            "world_model_losses": [],
            "controller_dream_rewards": [],
            "evaluation_rewards": []
        }
    
    def pretrain_dqn(self):
        """预训练 DQN 用于数据收集"""
        print("🎯 阶段 0: 预训练 DQN (用于数据收集)")
        print("-" * 50)
        
        epsilon = 1.0
        epsilon_decay = 0.995
        
        for episode in range(self.config.pretrain_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # ε-greedy
                action = self.dqn.get_action(state, epsilon, self.config.device)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 存储经验
                self.dqn_memory.append((state, action, reward, next_state, done))
                episode_reward += reward
                state = next_state
                
                # 训练 DQN
                if len(self.dqn_memory) > 64:
                    batch = np.random.choice(len(self.dqn_memory), 64, replace=False)
                    states = torch.FloatTensor([self.dqn_memory[i][0] for i in batch]).to(self.config.device)
                    actions = torch.LongTensor([self.dqn_memory[i][1] for i in batch]).to(self.config.device)
                    rewards = torch.FloatTensor([self.dqn_memory[i][2] for i in batch]).to(self.config.device)
                    next_states = torch.FloatTensor([self.dqn_memory[i][3] for i in batch]).to(self.config.device)
                    dones = torch.FloatTensor([self.dqn_memory[i][4] for i in batch]).to(self.config.device)
                    
                    current_q = self.dqn(states).gather(1, actions.unsqueeze(1))
                    next_q = self.dqn(next_states).max(1)[0].detach()
                    target_q = rewards + 0.99 * next_q * (1 - dones)
                    
                    loss = nn.MSELoss()(current_q.squeeze(), target_q)
                    self.dqn_optimizer.zero_grad()
                    loss.backward()
                    self.dqn_optimizer.step()
            
            epsilon = max(0.1, epsilon * epsilon_decay)
            self.training_history["dqn_pretrain_rewards"].append(episode_reward)
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(self.training_history["dqn_pretrain_rewards"][-20:])
                print(f"Episode {episode+1}/{self.config.pretrain_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | ε: {epsilon:.3f}")
        
        final_avg = np.mean(self.training_history["dqn_pretrain_rewards"][-20:])
        print(f"✅ DQN 预训练完成 | 最终平均: {final_avg:.2f}")
    
    def collect_data_with_policy(self):
        """用训练好的 DQN 收集高质量数据"""
        print("\n📦 阶段 1: 用策略收集数据")
        print("-" * 50)
        
        epsilon_schedule = np.linspace(
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.config.data_collection_episodes
        )
        
        for episode in range(self.config.data_collection_episodes):
            trajectory = {
                "states": [],
                "actions": [],
                "rewards": [],
                "dones": []
            }
            
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            epsilon = epsilon_schedule[episode]
            
            while not done:
                action = self.dqn.get_action(state, epsilon, self.config.device)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # One-hot encode
                action_onehot = np.zeros(self.action_dim)
                action_onehot[action] = 1
                
                trajectory["states"].append(state)
                trajectory["actions"].append(action_onehot)
                trajectory["rewards"].append(reward)
                trajectory["dones"].append(float(done))
                
                episode_reward += reward
                state = next_state
            
            self.trajectories.append(trajectory)
            self.training_history["data_collection_rewards"].append(episode_reward)
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(self.training_history["data_collection_rewards"][-20:])
                print(f"Episode {episode+1}/{self.config.data_collection_episodes} | "
                      f"Avg Reward: {avg_reward:.2f}")
        
        avg_reward = np.mean(self.training_history["data_collection_rewards"])
        print(f"✅ 收集了 {len(self.trajectories)} 条轨迹 | 平均奖励: {avg_reward:.2f}")
    
    def train_world_model(self):
        """训练世界模型"""
        print("\n🌍 阶段 2: 训练世界模型")
        print("-" * 50)
        
        for epoch in range(self.config.world_model_epochs):
            epoch_losses = []
            
            for _ in range(100):  # 每轮更多更新
                traj = np.random.choice(self.trajectories)
                
                max_seq_len = min(len(traj["states"]) - 1, self.config.sequence_length)
                if max_seq_len < 2:
                    continue
                
                start_idx = np.random.randint(0, len(traj["states"]) - max_seq_len)
                
                states = torch.FloatTensor(
                    np.array(traj["states"][start_idx:start_idx+max_seq_len])
                ).unsqueeze(0).to(self.config.device)
                
                actions = torch.FloatTensor(
                    np.array(traj["actions"][start_idx:start_idx+max_seq_len])
                ).unsqueeze(0).to(self.config.device)
                
                next_states = torch.FloatTensor(
                    np.array(traj["states"][start_idx+1:start_idx+max_seq_len+1])
                ).unsqueeze(0).to(self.config.device)
                
                rewards = torch.FloatTensor(
                    np.array(traj["rewards"][start_idx:start_idx+max_seq_len])
                ).unsqueeze(0).unsqueeze(-1).to(self.config.device)
                
                dones = torch.FloatTensor(
                    np.array(traj["dones"][start_idx:start_idx+max_seq_len])
                ).unsqueeze(0).unsqueeze(-1).to(self.config.device)
                
                # 前向
                pred_states, pred_rewards, pred_dones, _ = self.world_model(states, actions)
                
                # 损失
                state_loss = nn.MSELoss()(pred_states, next_states)
                reward_loss = nn.MSELoss()(pred_rewards, rewards)
                done_loss = nn.BCELoss()(pred_dones, dones)
                
                loss = state_loss + reward_loss * 10.0 + done_loss * 5.0
                
                # 优化
                self.wm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
                self.wm_optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            self.training_history["world_model_losses"].append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.config.world_model_epochs} | Loss: {avg_loss:.4f}")
        
        print("✅ 世界模型训练完成")
    
    def train_controller_in_dream(self):
        """在梦境中用梯度训练控制器"""
        print("\n💭 阶段 3: 梦境训练控制器")
        print("-" * 50)
        
        for step in range(self.config.dream_training_steps):
            # 采样起始状态
            batch_trajs = np.random.choice(self.trajectories, self.config.dream_batch_size)
            start_states = []
            for traj in batch_trajs:
                idx = np.random.randint(0, len(traj["states"]))
                start_states.append(traj["states"][idx])
            
            start_states = torch.FloatTensor(np.array(start_states)).to(self.config.device)
            
            # 想象 rollout
            total_reward = 0
            states = start_states
            hidden = None
            
            for t in range(self.config.dream_horizon):
                # 策略选择动作
                logits = self.controller(states)
                action_probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(action_probs, 1).squeeze(-1)
                
                # One-hot
                action_onehot = torch.zeros(self.config.dream_batch_size, self.action_dim).to(self.config.device)
                action_onehot.scatter_(1, actions.unsqueeze(1), 1)
                
                # 世界模型预测（需要保留梯度）
                states_input = states.unsqueeze(1)
                actions_input = action_onehot.unsqueeze(1)
                
                # 临时启用梯度
                self.world_model.eval()
                with torch.enable_grad():
                    next_states, rewards, dones, hidden = self.world_model(
                        states_input, 
                        actions_input, 
                        hidden
                    )
                self.world_model.train()
                
                next_states = next_states.squeeze(1)
                rewards = rewards.squeeze(1)
                
                total_reward = total_reward + rewards
                states = next_states.detach()  # 断开梯度（只优化策略）
                
                # 提前终止
                if dones.mean() > 0.5:
                    break
            
            # 优化策略（最大化累积奖励）
            loss = -total_reward.mean()
            
            self.controller_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
            self.controller_optimizer.step()
            
            if (step + 1) % 500 == 0:
                avg_dream_reward = -loss.item()
                self.training_history["controller_dream_rewards"].append(avg_dream_reward)
                print(f"Step {step+1}/{self.config.dream_training_steps} | "
                      f"Dream Reward: {avg_dream_reward:.2f}")
        
        print("✅ 控制器训练完成")
    
    def evaluate(self, num_episodes=50):
        """在真实环境评估"""
        print("\n📊 真实环境评估...")
        rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.controller.get_action(state, self.config.device, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        self.training_history["evaluation_rewards"].append(mean_reward)
        
        print(f"✅ 评估结果: {mean_reward:.2f} ± {std_reward:.2f}")
        return mean_reward, std_reward
    
    def train(self):
        """完整训练流程"""
        print("🚀 开始训练 Improved Simple World Model")
        print(f"设备: {self.config.device}")
        print("=" * 50)
        
        # 阶段 0: 预训练 DQN
        self.pretrain_dqn()
        
        # 阶段 1: 用策略收集数据
        self.collect_data_with_policy()
        
        # 阶段 2: 训练世界模型
        self.train_world_model()
        
        # 阶段 3: 在梦境中训练控制器
        self.train_controller_in_dream()
        
        # 评估
        self.evaluate()
        
        # 保存
        self.save_results()
    
    def save_results(self):
        """保存结果"""
        import os
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # 保存模型
        torch.save(self.world_model.state_dict(),
                   f"{self.config.save_dir}/world_model.pt")
        torch.save(self.controller.state_dict(),
                   f"{self.config.save_dir}/controller.pt")
        
        # 保存训练历史
        with open(f"{self.config.save_dir}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        # 绘图
        self.plot_results()
        print(f"📊 结果已保存到: {self.config.save_dir}")
    
    def plot_results(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # DQN 预训练
        axes[0, 0].plot(self.training_history["dqn_pretrain_rewards"], alpha=0.5)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('DQN Pretraining')
        axes[0, 0].grid(True)
        
        # 数据收集
        axes[0, 1].plot(self.training_history["data_collection_rewards"], alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Data Collection')
        axes[0, 1].grid(True)
        
        # 世界模型损失
        axes[0, 2].plot(self.training_history["world_model_losses"])
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('World Model Training')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True)
        
        # 梦境奖励
        if self.training_history["controller_dream_rewards"]:
            axes[1, 0].plot(self.training_history["controller_dream_rewards"])
            axes[1, 0].set_xlabel('Step (x500)')
            axes[1, 0].set_ylabel('Dream Reward')
            axes[1, 0].set_title('Dream Training')
            axes[1, 0].grid(True)
        
        # 最终评估
        if self.training_history["evaluation_rewards"]:
            axes[1, 1].bar(['Evaluation'], self.training_history["evaluation_rewards"])
            axes[1, 1].set_ylabel('Mean Reward')
            axes[1, 1].set_title('Real Environment')
            axes[1, 1].grid(True)
        
        # 对比
        axes[1, 2].text(0.5, 0.5, 
                       f"Data Collection Avg:\n{np.mean(self.training_history['data_collection_rewards']):.2f}\n\n"
                       f"Final Evaluation:\n{self.training_history['evaluation_rewards'][0]:.2f}",
                       ha='center', va='center', fontsize=12)
        axes[1, 2].set_title('Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/training_curves.png", dpi=150)
        plt.close()


# ========== 主函数 ==========
def main():
    config = Config()
    agent = ImprovedWorldModelAgent(config)
    agent.train()


if __name__ == "__main__":
    main()
