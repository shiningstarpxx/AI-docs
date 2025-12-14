"""
CartPole-v1: Full World Model Implementation
=============================================
完整的 World Models (2018) 实现，包含 MDN-RNN

架构：
- Vision: 状态编码器（CartPole 状态已是低维，用简单 MLP）
- Memory: MDN-LSTM 预测下一状态的分布
- Controller: 线性策略 + CMA-ES

与简化版的区别：
- 使用 MDN (Mixture Density Network) 输出概率分布
- 更接近论文原版架构

参考论文：
- World Models (Ha & Schmidhuber, 2018)
- https://arxiv.org/abs/1803.10122
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import json
import math
import os


# ========== 配置 ==========
class Config:
    # 环境
    env_name = "CartPole-v1"

    # 阶段 1: 数据收集
    random_episodes = 100  # 随机策略收集数据

    # 阶段 2: 训练世界模型
    world_model_epochs = 100
    batch_size = 32
    sequence_length = 50  # LSTM 序列长度
    hidden_size = 64      # LSTM 隐藏层大小
    latent_size = 16      # 潜在空间维度 (状态编码)
    n_gaussians = 5       # MDN 混合高斯分量数
    learning_rate = 1e-3

    # 阶段 3: 梦境中训练策略 (CMA-ES)
    population_size = 50
    generations = 100
    dream_rollout_length = 200
    elite_frac = 0.2
    temperature = 1.0     # 采样温度

    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    # 日志
    save_dir = "./results_world_model_full"


# ========== Vision Model: 状态编码器 ==========
class VisionEncoder(nn.Module):
    """
    V: Vision Model

    对于 CartPole，状态已经是低维向量 (4维)，
    但为了完整性，我们仍然加入一个编码器。

    在图像任务中，这里应该是 VAE。
    """
    def __init__(self, state_dim, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size * 2)  # 输出 μ 和 log_σ
        )
        self.latent_size = latent_size

    def forward(self, x):
        """编码状态到潜在空间"""
        h = self.encoder(x)
        mu, log_sigma = h.chunk(2, dim=-1)
        return mu, log_sigma

    def sample(self, mu, log_sigma):
        """重参数化采样"""
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z

    def encode(self, x):
        """编码并采样"""
        mu, log_sigma = self.forward(x)
        z = self.sample(mu, log_sigma)
        return z, mu, log_sigma


class VisionDecoder(nn.Module):
    """解码器：从潜在空间重建状态"""
    def __init__(self, latent_size, state_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )

    def forward(self, z):
        return self.decoder(z)


# ========== MDN (Mixture Density Network) ==========
class MDN(nn.Module):
    """
    Mixture Density Network

    输出混合高斯分布的参数：
    - π: 混合权重 (K个)
    - μ: 均值 (K * output_dim 个)
    - σ: 标准差 (K * output_dim 个)
    """
    def __init__(self, input_size, output_size, n_gaussians):
        super().__init__()
        self.output_size = output_size
        self.n_gaussians = n_gaussians

        # 输出层
        self.pi_layer = nn.Linear(input_size, n_gaussians)
        self.mu_layer = nn.Linear(input_size, n_gaussians * output_size)
        self.sigma_layer = nn.Linear(input_size, n_gaussians * output_size)

    def forward(self, x):
        """
        Args:
            x: (batch, input_size)
        Returns:
            pi: (batch, n_gaussians) - 混合权重
            mu: (batch, n_gaussians, output_size) - 均值
            sigma: (batch, n_gaussians, output_size) - 标准差
        """
        pi = F.softmax(self.pi_layer(x), dim=-1)
        mu = self.mu_layer(x).view(-1, self.n_gaussians, self.output_size)
        sigma = torch.exp(self.sigma_layer(x)).view(-1, self.n_gaussians, self.output_size)
        # 限制 sigma 的范围，防止数值问题
        sigma = torch.clamp(sigma, min=1e-4, max=10.0)
        return pi, mu, sigma

    def sample(self, pi, mu, sigma, temperature=1.0):
        """
        从混合高斯分布中采样

        Args:
            temperature: 温度参数，>1 增加随机性，<1 减少随机性
        """
        batch_size = pi.shape[0]

        # 按权重选择高斯分量
        pi_temp = pi ** (1.0 / temperature)
        pi_temp = pi_temp / pi_temp.sum(dim=-1, keepdim=True)

        # 采样分量索引
        indices = torch.multinomial(pi_temp, 1).squeeze(-1)  # (batch,)

        # 获取对应的 μ 和 σ
        batch_idx = torch.arange(batch_size, device=mu.device)
        mu_selected = mu[batch_idx, indices]  # (batch, output_size)
        sigma_selected = sigma[batch_idx, indices] * temperature

        # 从选中的高斯分布采样
        eps = torch.randn_like(mu_selected)
        sample = mu_selected + sigma_selected * eps

        return sample

    def log_prob(self, pi, mu, sigma, target):
        """
        计算目标值的对数概率

        Args:
            target: (batch, output_size)
        Returns:
            log_prob: (batch,)
        """
        target = target.unsqueeze(1)  # (batch, 1, output_size)

        # 计算每个高斯分量的概率密度
        # N(x | μ, σ) = (1 / sqrt(2πσ²)) * exp(-(x-μ)² / 2σ²)
        var = sigma ** 2
        log_prob_per_dim = -0.5 * (
            math.log(2 * math.pi) +
            torch.log(var) +
            (target - mu) ** 2 / var
        )

        # 对所有维度求和 (假设维度独立)
        log_prob_per_gaussian = log_prob_per_dim.sum(dim=-1)  # (batch, n_gaussians)

        # 混合：log(Σ π_i * exp(log_prob_i))
        # 使用 log-sum-exp 技巧保证数值稳定性
        log_pi = torch.log(pi + 1e-8)
        log_prob = torch.logsumexp(log_pi + log_prob_per_gaussian, dim=-1)

        return log_prob


# ========== Memory Model: MDN-LSTM ==========
class MemoryModel(nn.Module):
    """
    M: Memory Model (MDN-LSTM)

    输入: (z_t, a_t, h_{t-1})
    输出: P(z_{t+1}) = Σ π_i * N(μ_i, σ_i²)

    同时预测 reward 和 done
    """
    def __init__(self, latent_size, action_dim, hidden_size, n_gaussians):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # 输入编码
        input_size = latent_size + action_dim
        self.input_encoder = nn.Linear(input_size, hidden_size)

        # LSTM 核心
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # MDN 输出下一状态分布
        self.mdn = MDN(hidden_size, latent_size, n_gaussians)

        # 奖励和终止预测
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)

    def forward(self, z, action, hidden=None):
        """
        Args:
            z: (batch, seq_len, latent_size)
            action: (batch, seq_len, action_dim)
            hidden: LSTM 隐藏状态
        Returns:
            pi, mu, sigma: MDN 参数
            reward: 预测奖励
            done: 预测终止概率
            hidden: 更新后的隐藏状态
        """
        batch_size, seq_len, _ = z.shape

        # 编码输入
        x = torch.cat([z, action], dim=-1)
        x = torch.relu(self.input_encoder(x))

        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # 重塑用于 MDN
        lstm_out_flat = lstm_out.reshape(batch_size * seq_len, -1)

        # MDN 预测下一状态分布
        pi, mu, sigma = self.mdn(lstm_out_flat)

        # 重塑回序列形式
        pi = pi.view(batch_size, seq_len, -1)
        mu = mu.view(batch_size, seq_len, self.mdn.n_gaussians, self.latent_size)
        sigma = sigma.view(batch_size, seq_len, self.mdn.n_gaussians, self.latent_size)

        # 奖励和终止预测
        reward = self.reward_head(lstm_out)
        done = torch.sigmoid(self.done_head(lstm_out))

        return pi, mu, sigma, reward, done, hidden

    def imagine_step(self, z, action, hidden=None, temperature=1.0):
        """
        单步想象（用于策略训练）

        Args:
            z: (batch, latent_size)
            action: (batch, action_dim)
            temperature: 采样温度
        """
        with torch.no_grad():
            z = z.unsqueeze(1)  # (batch, 1, latent_size)
            action = action.unsqueeze(1)  # (batch, 1, action_dim)

            pi, mu, sigma, reward, done, hidden = self.forward(z, action, hidden)

            # 从分布中采样下一状态
            pi = pi.squeeze(1)  # (batch, n_gaussians)
            mu = mu.squeeze(1)  # (batch, n_gaussians, latent_size)
            sigma = sigma.squeeze(1)

            next_z = self.mdn.sample(pi, mu, sigma, temperature)

            return next_z, reward.squeeze(1), done.squeeze(1), hidden


# ========== Controller: 线性策略 ==========
class LinearController:
    """
    C: Controller

    线性策略: action = argmax(W @ [z, h])
    参数极少，适合进化算法优化

    注意：使用潜在空间 z 和 LSTM 隐藏状态 h
    """
    def __init__(self, input_dim, action_dim):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.weights = np.random.randn(action_dim, input_dim) * 0.1
        self.bias = np.zeros(action_dim)

    def get_action(self, state):
        """选择动作"""
        logits = self.weights @ state + self.bias
        return np.argmax(logits)

    def get_params(self):
        """获取参数（扁平化）"""
        return np.concatenate([self.weights.flatten(), self.bias])

    def set_params(self, params):
        """设置参数"""
        w_size = self.action_dim * self.input_dim
        self.weights = params[:w_size].reshape(self.action_dim, self.input_dim)
        self.bias = params[w_size:]

    @property
    def num_params(self):
        return self.action_dim * self.input_dim + self.action_dim


# ========== CMA-ES 进化算法 ==========
class CMAES:
    """
    简化的 CMA-ES 实现

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    是一种无梯度优化算法，适合优化参数较少的策略
    """
    def __init__(self, dim, population_size=50, elite_frac=0.2, sigma=0.5):
        self.dim = dim
        self.population_size = population_size
        self.elite_size = int(population_size * elite_frac)

        # 初始分布参数
        self.mean = np.zeros(dim)
        self.sigma = sigma

        # 协方差矩阵（简化为对角）
        self.cov = np.ones(dim)

    def ask(self):
        """生成候选解"""
        # 从正态分布采样
        noise = np.random.randn(self.population_size, self.dim)
        population = self.mean + self.sigma * np.sqrt(self.cov) * noise
        return population

    def tell(self, population, fitness):
        """根据适应度更新分布"""
        # 选择精英（适应度最高的）
        elite_idxs = np.argsort(fitness)[-self.elite_size:]
        elite = population[elite_idxs]
        elite_fitness = fitness[elite_idxs]

        # 加权更新均值
        weights = np.exp(elite_fitness - elite_fitness.max())
        weights = weights / weights.sum()
        self.mean = (weights[:, None] * elite).sum(axis=0)

        # 更新协方差（简化版）
        diff = elite - self.mean
        self.cov = 0.8 * self.cov + 0.2 * (weights[:, None] * diff ** 2).sum(axis=0)

        # 自适应更新 sigma
        self.sigma = self.sigma * 0.95 + 0.05 * np.std(elite_fitness)


# ========== Full World Model Agent ==========
class FullWorldModelAgent:
    """
    完整的 World Model Agent

    包含：
    - V: Vision Model (状态编码器)
    - M: Memory Model (MDN-LSTM)
    - C: Controller (线性策略)
    """
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_name)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # V: Vision Model
        self.encoder = VisionEncoder(
            self.state_dim,
            config.latent_size
        ).to(config.device)

        self.decoder = VisionDecoder(
            config.latent_size,
            self.state_dim
        ).to(config.device)

        # M: Memory Model
        self.memory = MemoryModel(
            config.latent_size,
            self.action_dim,
            config.hidden_size,
            config.n_gaussians
        ).to(config.device)

        # 优化器
        self.vae_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config.learning_rate
        )
        self.memory_optimizer = optim.Adam(
            self.memory.parameters(),
            lr=config.learning_rate
        )

        # 数据缓冲
        self.trajectories = []

        # 记录
        self.training_history = {
            "data_collection_rewards": [],
            "vae_losses": [],
            "memory_losses": [],
            "policy_fitness": [],
            "evaluation_rewards": []
        }

    def collect_data(self):
        """阶段 1: 随机策略收集数据"""
        print("=" * 60)
        print("📦 阶段 1: 收集数据")
        print("=" * 60)

        for episode in range(self.config.random_episodes):
            trajectory = {
                "states": [],
                "actions": [],
                "rewards": [],
                "dones": []
            }

            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # One-hot encode action
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
                print(f"  Episode {episode+1:3d}/{self.config.random_episodes} | "
                      f"Avg Reward: {avg_reward:.1f}")

        total_steps = sum(len(t["states"]) for t in self.trajectories)
        print(f"\n✅ 收集了 {len(self.trajectories)} 条轨迹, 共 {total_steps} 步")

    def train_vae(self):
        """训练 VAE (Vision Model)"""
        print("\n" + "=" * 60)
        print("👁️  训练 VAE (Vision Model)")
        print("=" * 60)

        for epoch in range(self.config.world_model_epochs // 2):
            epoch_losses = []

            for _ in range(20):
                # 随机采样一批状态
                traj = np.random.choice(self.trajectories)
                idx = np.random.randint(0, len(traj["states"]))
                state = torch.FloatTensor(traj["states"][idx]).to(self.config.device)

                # 前向传播
                z, mu, log_sigma = self.encoder.encode(state.unsqueeze(0))
                state_recon = self.decoder(z)

                # 损失：重建 + KL
                recon_loss = F.mse_loss(state_recon, state.unsqueeze(0))
                kl_loss = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - torch.exp(2 * log_sigma))
                loss = recon_loss + 0.001 * kl_loss  # β = 0.001

                # 优化
                self.vae_optimizer.zero_grad()
                loss.backward()
                self.vae_optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            self.training_history["vae_losses"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.world_model_epochs//2} | "
                      f"Loss: {avg_loss:.4f}")

        print("✅ VAE 训练完成")

    def train_memory(self):
        """训练 MDN-LSTM (Memory Model)"""
        print("\n" + "=" * 60)
        print("🧠 训练 MDN-LSTM (Memory Model)")
        print("=" * 60)

        for epoch in range(self.config.world_model_epochs):
            epoch_losses = []

            for _ in range(20):
                traj = np.random.choice(self.trajectories)

                # 准备序列数据
                seq_len = min(len(traj["states"]) - 1, self.config.sequence_length)
                if seq_len < 5:
                    continue

                start_idx = np.random.randint(0, len(traj["states"]) - seq_len)

                # 转换为张量
                states = torch.FloatTensor(
                    traj["states"][start_idx:start_idx+seq_len]
                ).to(self.config.device)

                actions = torch.FloatTensor(
                    traj["actions"][start_idx:start_idx+seq_len]
                ).to(self.config.device)

                next_states = torch.FloatTensor(
                    traj["states"][start_idx+1:start_idx+seq_len+1]
                ).to(self.config.device)

                rewards = torch.FloatTensor(
                    traj["rewards"][start_idx:start_idx+seq_len]
                ).unsqueeze(-1).to(self.config.device)

                dones = torch.FloatTensor(
                    traj["dones"][start_idx:start_idx+seq_len]
                ).unsqueeze(-1).to(self.config.device)

                # 编码状态
                with torch.no_grad():
                    z, _, _ = self.encoder.encode(states)
                    next_z, _, _ = self.encoder.encode(next_states)

                # 添加 batch 维度
                z = z.unsqueeze(0)
                actions = actions.unsqueeze(0)
                next_z = next_z.unsqueeze(0)
                rewards = rewards.unsqueeze(0)
                dones = dones.unsqueeze(0)

                # 前向传播
                pi, mu, sigma, pred_reward, pred_done, _ = self.memory(z, actions)

                # MDN 损失（负对数似然）
                # 重塑用于计算
                batch_size, seq_len_actual, n_g, latent_s = mu.shape
                pi_flat = pi.view(batch_size * seq_len_actual, n_g)
                mu_flat = mu.view(batch_size * seq_len_actual, n_g, latent_s)
                sigma_flat = sigma.view(batch_size * seq_len_actual, n_g, latent_s)
                next_z_flat = next_z.view(batch_size * seq_len_actual, latent_s)

                log_prob = self.memory.mdn.log_prob(pi_flat, mu_flat, sigma_flat, next_z_flat)
                mdn_loss = -log_prob.mean()

                # 奖励和终止损失
                reward_loss = F.mse_loss(pred_reward, rewards)
                done_loss = F.binary_cross_entropy(pred_done, dones)

                # 总损失
                loss = mdn_loss + reward_loss + done_loss

                # 优化
                self.memory_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.memory.parameters(), 1.0)
                self.memory_optimizer.step()

                epoch_losses.append(loss.item())

            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                self.training_history["memory_losses"].append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.config.world_model_epochs} | "
                          f"Loss: {avg_loss:.4f}")

        print("✅ Memory Model 训练完成")

    def train_controller_in_dream(self):
        """阶段 3: 在梦境中训练策略"""
        print("\n" + "=" * 60)
        print("💭 阶段 3: 梦境训练策略 (CMA-ES)")
        print("=" * 60)

        # Controller 输入: z + LSTM hidden state
        controller_input_dim = self.config.latent_size + self.config.hidden_size

        # 初始化 CMA-ES
        controller = LinearController(controller_input_dim, self.action_dim)
        cmaes = CMAES(controller.num_params, self.config.population_size)

        best_controller = None
        best_fitness = -float('inf')

        for generation in range(self.config.generations):
            # 生成种群
            population = cmaes.ask()
            fitness_scores = []

            # 评估每个个体
            for params in population:
                controller.set_params(params)

                # 在梦境中评估
                fitness = self.evaluate_in_dream(controller)
                fitness_scores.append(fitness)

            fitness_scores = np.array(fitness_scores)

            # 更新 CMA-ES
            cmaes.tell(population, fitness_scores)

            # 记录最佳
            gen_best = fitness_scores.max()
            gen_mean = fitness_scores.mean()

            if gen_best > best_fitness:
                best_fitness = gen_best
                best_controller = LinearController(controller_input_dim, self.action_dim)
                best_controller.set_params(population[fitness_scores.argmax()])

            self.training_history["policy_fitness"].append(gen_best)

            if (generation + 1) % 10 == 0:
                print(f"  Generation {generation+1:3d}/{self.config.generations} | "
                      f"Best: {gen_best:.1f} | Mean: {gen_mean:.1f}")

        print(f"\n✅ 策略训练完成 | 最佳适应度: {best_fitness:.1f}")
        return best_controller

    def evaluate_in_dream(self, controller, num_rollouts=3):
        """在梦境中评估策略"""
        total_reward = 0

        for _ in range(num_rollouts):
            # 从随机轨迹采样起始状态
            traj = np.random.choice(self.trajectories)
            state = torch.FloatTensor(traj["states"][0]).to(self.config.device)

            # 编码初始状态
            z, _, _ = self.encoder.encode(state.unsqueeze(0))
            z = z.squeeze(0)

            # 初始化 LSTM 隐藏状态
            hidden = None

            episode_reward = 0

            for _ in range(self.config.dream_rollout_length):
                # 获取 LSTM 隐藏状态
                if hidden is None:
                    h = torch.zeros(self.config.hidden_size).to(self.config.device)
                else:
                    h = hidden[0].squeeze(0).squeeze(0)  # (hidden_size,)

                # 构建 controller 输入
                controller_input = torch.cat([z, h]).detach().cpu().numpy()

                # 控制器选择动作
                action_idx = controller.get_action(controller_input)
                action = torch.zeros(self.action_dim).to(self.config.device)
                action[action_idx] = 1

                # 世界模型预测
                next_z, reward, done, hidden = self.memory.imagine_step(
                    z.unsqueeze(0),
                    action.unsqueeze(0),
                    hidden,
                    temperature=self.config.temperature
                )

                episode_reward += reward.item()

                # 终止检查
                if done.item() > 0.5:
                    break

                z = next_z.squeeze(0)

            total_reward += episode_reward

        return total_reward / num_rollouts

    def evaluate_in_real_env(self, controller, num_episodes=10):
        """在真实环境中评估"""
        rewards = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            # 编码初始状态
            state_tensor = torch.FloatTensor(state).to(self.config.device)
            z, _, _ = self.encoder.encode(state_tensor.unsqueeze(0))
            z = z.squeeze(0)

            hidden = None

            while not done:
                # 获取 LSTM 隐藏状态
                if hidden is None:
                    h = torch.zeros(self.config.hidden_size).to(self.config.device)
                else:
                    h = hidden[0].squeeze(0).squeeze(0)

                # 构建 controller 输入
                controller_input = torch.cat([z, h]).detach().cpu().numpy()

                # 选择动作
                action = controller.get_action(controller_input)

                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # 编码新状态并更新 LSTM
                if not done:
                    state_tensor = torch.FloatTensor(next_state).to(self.config.device)
                    z, _, _ = self.encoder.encode(state_tensor.unsqueeze(0))
                    z = z.squeeze(0)

                    # 更新 LSTM 隐藏状态
                    action_onehot = torch.zeros(self.action_dim).to(self.config.device)
                    action_onehot[action] = 1
                    _, _, _, _, _, hidden = self.memory(
                        z.unsqueeze(0).unsqueeze(0),
                        action_onehot.unsqueeze(0).unsqueeze(0),
                        hidden
                    )

            rewards.append(episode_reward)

        return np.mean(rewards), np.std(rewards)

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("🚀 Full World Model Training")
        print("=" * 60)
        print(f"  环境: {self.config.env_name}")
        print(f"  设备: {self.config.device}")
        print(f"  潜在空间维度: {self.config.latent_size}")
        print(f"  MDN 高斯分量数: {self.config.n_gaussians}")
        print(f"  LSTM 隐藏层: {self.config.hidden_size}")
        print("=" * 60)

        # 阶段 1: 收集数据
        self.collect_data()

        # 阶段 2: 训练世界模型
        self.train_vae()
        self.train_memory()

        # 阶段 3: 在梦境中训练策略
        best_controller = self.train_controller_in_dream()

        # 评估
        print("\n" + "=" * 60)
        print("📊 在真实环境中评估")
        print("=" * 60)

        mean_reward, std_reward = self.evaluate_in_real_env(best_controller, num_episodes=50)
        self.training_history["evaluation_rewards"].append(mean_reward)

        print(f"\n✅ 最终评估结果: {mean_reward:.1f} ± {std_reward:.1f}")

        # 保存结果
        self.save_results(best_controller)

        return best_controller

    def save_results(self, controller):
        """保存结果"""
        os.makedirs(self.config.save_dir, exist_ok=True)

        # 保存模型
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'memory': self.memory.state_dict()
        }, f"{self.config.save_dir}/world_model.pt")

        # 保存 controller
        np.savez(f"{self.config.save_dir}/controller.npz",
                 weights=controller.weights,
                 bias=controller.bias)

        # 保存训练历史
        with open(f"{self.config.save_dir}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        # 绘图
        self.plot_results()

        print(f"\n📁 结果已保存到: {self.config.save_dir}")

    def plot_results(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 数据收集阶段奖励
        axes[0, 0].plot(self.training_history["data_collection_rewards"], alpha=0.7)
        axes[0, 0].axhline(y=np.mean(self.training_history["data_collection_rewards"]),
                          color='r', linestyle='--', label='Mean')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Stage 1: Data Collection (Random Policy)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # VAE 损失
        if self.training_history["vae_losses"]:
            axes[0, 1].plot(self.training_history["vae_losses"])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Stage 2a: VAE Training')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)

        # Memory 损失
        if self.training_history["memory_losses"]:
            axes[1, 0].plot(self.training_history["memory_losses"])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss (NLL)')
            axes[1, 0].set_title('Stage 2b: MDN-LSTM Training')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # 策略适应度
        if self.training_history["policy_fitness"]:
            axes[1, 1].plot(self.training_history["policy_fitness"])
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Fitness (Dream Reward)')
            axes[1, 1].set_title('Stage 3: Policy Evolution (CMA-ES)')
            axes[1, 1].grid(True, alpha=0.3)

            # 添加最终评估结果
            if self.training_history["evaluation_rewards"]:
                final_reward = self.training_history["evaluation_rewards"][-1]
                axes[1, 1].axhline(y=final_reward, color='g', linestyle='--',
                                  label=f'Real Env: {final_reward:.1f}')
                axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/training_curves.png", dpi=150)
        plt.close()

        print(f"📈 训练曲线已保存")


# ========== 主函数 ==========
def main():
    config = Config()
    agent = FullWorldModelAgent(config)
    agent.train()


if __name__ == "__main__":
    main()
