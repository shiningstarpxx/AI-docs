"""
CartPole-v1: Mini Dreamer
==========================
简化的 Dreamer (2020) 实现

架构：
- RSSM: 确定性 RNN + 随机潜在变量
- Actor-Critic: 在想象轨迹中学习策略
- 在线学习: 持续改进世界模型和策略

关键对比点：
- 比 Simple WM 更快收敛（~5× 样本效率）
- 策略网络替代进化算法
- 支持在线学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import json

# ========== 配置 ==========
class Config:
    # 环境
    env_name = "CartPole-v1"
    
    # 训练
    num_episodes = 300
    seed_episodes = 5  # 初始随机收集
    
    # RSSM
    state_dim = 4  # CartPole 状态维度
    action_dim = 2
    hidden_size = 128  # 确定性隐藏状态
    stochastic_size = 32  # 随机潜在状态
    
    # 网络
    learning_rate = 3e-4
    batch_size = 16
    sequence_length = 50
    imagination_horizon = 15  # 想象视野
    
    # Actor-Critic
    gamma = 0.99
    lambda_gae = 0.95  # GAE 参数
    
    # 缓冲
    buffer_size = 5000
    
    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 日志
    log_interval = 10
    save_dir = "./results_mini_dreamer"


# ========== RSSM (Recurrent State Space Model) ==========
class RSSM(nn.Module):
    """
    简化的 RSSM
    h_t: 确定性路径 (RNN)
    s_t: 随机路径 (Gaussian)
    """
    def __init__(self, state_dim, action_dim, hidden_size, stochastic_size):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.stochastic_size = stochastic_size
        
        # Encoder: observation → embedding
        self.obs_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Deterministic path: RNN
        self.rnn = nn.GRUCell(hidden_size + action_dim, hidden_size)
        
        # Prior: p(s_t | h_t)
        self.prior_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(hidden_size, stochastic_size)
        self.prior_std = nn.Linear(hidden_size, stochastic_size)
        
        # Posterior: q(s_t | h_t, o_t)
        self.posterior_fc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU()
        )
        self.posterior_mean = nn.Linear(hidden_size, stochastic_size)
        self.posterior_std = nn.Linear(hidden_size, stochastic_size)
        
        # Decoder: (h_t, s_t) → o_t
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + stochastic_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_size + stochastic_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def get_prior(self, h):
        """Prior: p(s_t | h_t)"""
        x = self.prior_fc(h)
        mean = self.prior_mean(x)
        std = nn.functional.softplus(self.prior_std(x)) + 0.1
        return mean, std
    
    def get_posterior(self, h, obs_embed):
        """Posterior: q(s_t | h_t, o_t)"""
        x = self.posterior_fc(torch.cat([h, obs_embed], dim=-1))
        mean = self.posterior_mean(x)
        std = nn.functional.softplus(self.posterior_std(x)) + 0.1
        return mean, std
    
    def imagine_step(self, h, s, action):
        """想象一步（用于策略学习）"""
        # 更新确定性状态
        x = torch.cat([s, action], dim=-1)
        h_next = self.rnn(x, h)
        
        # Prior 采样
        mean, std = self.get_prior(h_next)
        s_next = mean + std * torch.randn_like(std)
        
        # 预测奖励
        reward = self.reward_predictor(torch.cat([h_next, s_next], dim=-1))
        
        return h_next, s_next, reward


# ========== Actor (策略网络) ==========
class Actor(nn.Module):
    """策略网络：(h, s) → action"""
    def __init__(self, hidden_size, stochastic_size, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size + stochastic_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, h, s):
        x = torch.cat([h, s], dim=-1)
        logits = self.network(x)
        return distributions.Categorical(logits=logits)


# ========== Critic (价值网络) ==========
class Critic(nn.Module):
    """价值网络：(h, s) → V(s)"""
    def __init__(self, hidden_size, stochastic_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size + stochastic_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, h, s):
        x = torch.cat([h, s], dim=-1)
        return self.network(x)


# ========== Mini Dreamer Agent ==========
class MiniDreamerAgent:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_name)
        
        # 网络
        self.rssm = RSSM(
            config.state_dim,
            config.action_dim,
            config.hidden_size,
            config.stochastic_size
        ).to(config.device)
        
        self.actor = Actor(
            config.hidden_size,
            config.stochastic_size,
            config.action_dim
        ).to(config.device)
        
        self.critic = Critic(
            config.hidden_size,
            config.stochastic_size
        ).to(config.device)
        
        # 优化器
        self.world_model_optimizer = optim.Adam(
            self.rssm.parameters(),
            lr=config.learning_rate
        )
        
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate
        )
        
        # 经验缓冲
        self.buffer = deque(maxlen=config.buffer_size)
        
        # 记录
        self.episode_rewards = []
        self.world_model_losses = []
        self.actor_losses = []
        self.critic_losses = []
    
    def collect_episode(self, random=False):
        """收集一条轨迹"""
        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": []
        }
        
        obs, _ = self.env.reset()
        episode_reward = 0
        done = False
        
        # 初始化 RSSM 状态
        h = torch.zeros(1, self.config.hidden_size).to(self.config.device)
        s = torch.zeros(1, self.config.stochastic_size).to(self.config.device)
        
        while not done:
            if random:
                action = self.env.action_space.sample()
            else:
                # 使用 Actor
                with torch.no_grad():
                    action_dist = self.actor(h, s)
                    action = action_dist.sample().item()
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # 记录
            trajectory["observations"].append(obs)
            
            # One-hot action
            action_onehot = np.zeros(self.config.action_dim)
            action_onehot[action] = 1
            trajectory["actions"].append(action_onehot)
            
            trajectory["rewards"].append(reward)
            
            # 更新 RSSM 状态（用于下一步决策）
            if not random:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                    action_tensor = torch.FloatTensor(action_onehot).unsqueeze(0).to(self.config.device)
                    
                    obs_embed = self.rssm.obs_encoder(obs_tensor)
                    h = self.rssm.rnn(torch.cat([s, action_tensor], dim=-1), h)
                    mean, std = self.rssm.get_posterior(h, obs_embed)
                    s = mean  # 使用均值（测试时）
            
            episode_reward += reward
            obs = next_obs
        
        # 添加最后一个观测
        trajectory["observations"].append(obs)
        
        self.buffer.append(trajectory)
        return episode_reward
    
    def train_world_model(self):
        """训练世界模型（RSSM）"""
        if len(self.buffer) < self.config.batch_size:
            return
        
        # 随机采样轨迹
        batch = np.random.choice(list(self.buffer), self.config.batch_size, replace=False)
        
        total_loss = 0
        
        for traj in batch:
            seq_len = min(len(traj["observations"]) - 1, self.config.sequence_length)
            
            observations = torch.FloatTensor(
                traj["observations"][:seq_len]
            ).to(self.config.device)
            
            actions = torch.FloatTensor(
                traj["actions"][:seq_len]
            ).to(self.config.device)
            
            next_observations = torch.FloatTensor(
                traj["observations"][1:seq_len+1]
            ).to(self.config.device)
            
            rewards = torch.FloatTensor(
                traj["rewards"][:seq_len]
            ).unsqueeze(-1).to(self.config.device)
            
            # 前向传播
            h = torch.zeros(1, self.config.hidden_size).to(self.config.device)
            
            reconstruction_loss = 0
            kl_loss = 0
            reward_loss = 0
            
            for t in range(seq_len):
                obs = observations[t:t+1]
                action = actions[t:t+1]
                next_obs = next_observations[t:t+1]
                reward = rewards[t:t+1]
                
                # Encode
                obs_embed = self.rssm.obs_encoder(obs)
                
                # Prior
                prior_mean, prior_std = self.rssm.get_prior(h)
                
                # Posterior
                posterior_mean, posterior_std = self.rssm.get_posterior(h, obs_embed)
                
                # 采样
                s = posterior_mean + posterior_std * torch.randn_like(posterior_std)
                
                # Decode
                reconstructed_obs = self.rssm.decoder(torch.cat([h, s], dim=-1))
                
                # Predict reward
                predicted_reward = self.rssm.reward_predictor(torch.cat([h, s], dim=-1))
                
                # 损失
                reconstruction_loss += nn.MSELoss()(reconstructed_obs, next_obs)
                
                # KL divergence
                kl = torch.distributions.kl_divergence(
                    distributions.Normal(posterior_mean, posterior_std),
                    distributions.Normal(prior_mean, prior_std)
                ).sum(-1).mean()
                kl_loss += kl
                
                reward_loss += nn.MSELoss()(predicted_reward, reward)
                
                # 更新 h
                h = self.rssm.rnn(torch.cat([s, action], dim=-1), h)
            
            # 总损失
            loss = reconstruction_loss + kl_loss + reward_loss
            total_loss += loss.item()
            
            self.world_model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 100)
            self.world_model_optimizer.step()
        
        self.world_model_losses.append(total_loss / self.config.batch_size)
    
    def train_actor_critic(self):
        """在想象中训练 Actor-Critic"""
        if len(self.buffer) < self.config.batch_size:
            return
        
        # 从缓冲中采样起始状态
        batch = np.random.choice(list(self.buffer), self.config.batch_size, replace=False)
        
        actor_loss_total = 0
        critic_loss_total = 0
        
        for traj in batch:
            # 随机选择起始点
            start_idx = np.random.randint(0, len(traj["observations"]) - 1)
            obs = torch.FloatTensor(traj["observations"][start_idx]).unsqueeze(0).to(self.config.device)
            
            # 初始化 RSSM 状态
            obs_embed = self.rssm.obs_encoder(obs)
            h = torch.zeros(1, self.config.hidden_size).to(self.config.device)
            s, _ = self.rssm.get_prior(h)
            
            # 想象展开
            imagined_trajectory = []
            
            for _ in range(self.config.imagination_horizon):
                # Actor 采样动作
                action_dist = self.actor(h, s)
                action = action_dist.sample()
                
                # One-hot
                action_onehot = torch.zeros(self.config.action_dim).to(self.config.device)
                action_onehot[action] = 1
                
                # Critic 估计价值
                value = self.critic(h, s)
                
                # RSSM 想象下一步
                h, s, reward = self.rssm.imagine_step(h, s, action_onehot.unsqueeze(0))
                h = h.squeeze(0).unsqueeze(0)
                s = s.squeeze(0).unsqueeze(0)
                
                imagined_trajectory.append({
                    "h": h.detach(),
                    "s": s.detach(),
                    "action": action,
                    "reward": reward.squeeze(),
                    "value": value.squeeze(),
                    "log_prob": action_dist.log_prob(action)
                })
            
            # 计算 GAE 和 returns
            returns = []
            advantages = []
            
            next_value = self.critic(h, s).squeeze().detach()
            
            for t in reversed(range(len(imagined_trajectory))):
                reward = imagined_trajectory[t]["reward"]
                value = imagined_trajectory[t]["value"]
                
                # TD error
                td_error = reward + self.config.gamma * next_value - value
                
                # GAE
                if t == len(imagined_trajectory) - 1:
                    advantage = td_error
                else:
                    advantage = td_error + self.config.gamma * self.config.lambda_gae * advantages[0]
                
                advantages.insert(0, advantage)
                returns.insert(0, advantage + value)
                
                next_value = value
            
            # Actor loss (Policy Gradient)
            actor_loss = 0
            for t, step in enumerate(imagined_trajectory):
                actor_loss -= step["log_prob"] * advantages[t].detach()
            
            # Critic loss
            critic_loss = 0
            for t, step in enumerate(imagined_trajectory):
                critic_loss += (step["value"] - returns[t].detach()) ** 2
            
            # 优化
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
            self.critic_optimizer.step()
            
            actor_loss_total += actor_loss.item()
            critic_loss_total += critic_loss.item()
        
        self.actor_losses.append(actor_loss_total / self.config.batch_size)
        self.critic_losses.append(critic_loss_total / self.config.batch_size)
    
    def train(self):
        """主训练循环"""
        print("🚀 开始训练 Mini Dreamer")
        print(f"设备: {self.config.device}")
        print("-" * 50)
        
        # 初始随机收集
        print("📦 初始数据收集...")
        for _ in range(self.config.seed_episodes):
            self.collect_episode(random=True)
        
        # 主循环
        for episode in range(self.config.num_episodes):
            # 收集数据
            episode_reward = self.collect_episode(random=False)
            self.episode_rewards.append(episode_reward)
            
            # 训练世界模型
            for _ in range(5):  # 多次更新
                self.train_world_model()
            
            # 训练 Actor-Critic
            for _ in range(5):
                self.train_actor_critic()
            
            # 日志
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.log_interval:])
                print(f"Episode {episode+1}/{self.config.num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Buffer Size: {len(self.buffer)}")
        
        print("\n✅ 训练完成！")
        self.save_results()
    
    def save_results(self):
        """保存结果"""
        import os
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # 保存模型
        torch.save({
            "rssm": self.rssm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, f"{self.config.save_dir}/models.pt")
        
        # 保存训练数据
        results = {
            "episode_rewards": self.episode_rewards,
            "world_model_losses": self.world_model_losses,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses
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
        window = 30
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
        
        # World Model Loss
        if self.world_model_losses:
            axes[0, 1].plot(self.world_model_losses)
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('World Model Loss')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)
        
        # Actor Loss
        if self.actor_losses:
            axes[1, 0].plot(self.actor_losses)
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].grid(True)
        
        # Critic Loss
        if self.critic_losses:
            axes[1, 1].plot(self.critic_losses)
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/training_curves.png", dpi=150)
        plt.close()


# ========== 主函数 ==========
def main():
    config = Config()
    agent = MiniDreamerAgent(config)
    agent.train()


if __name__ == "__main__":
    main()
