"""
CarRacing World Model: Improved Version
========================================
在论文基础上的改进版本，保持三阶段框架不变：
1. V-M-C 架构不变
2. CMA-ES 训练流程不变
3. 针对可能的问题点进行优化

改进点：
1. 更强的 VAE (更多通道、更深)
2. 非线性 Controller (MLP 替代纯线性)
3. 数据增强 (随机翻转、亮度调整)
4. 温度探索 (0.5, 1.0, 1.5)
5. 学习率调度 (Cosine Annealing)
6. 更好的数据收集策略 (混合随机+启发式)

参考论文：
- https://arxiv.org/abs/1803.10122
- https://worldmodels.github.io/
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
import pickle
import time
import logging
from datetime import datetime, timedelta
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# ========== 日志设置 ==========
def setup_logging(save_dir):
    """设置日志记录"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = f"{save_dir}/training.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ========== 配置 ==========
MODE = "improved"  # "improved" 或 "improved_quick"

class Config:
    # 环境
    env_name = "CarRacing-v3"
    img_size = 64

    # ===== 改进的 VAE 设置 =====
    latent_size = 64  # 🔧 从 32 增加到 64（更强表达能力）
    vae_channels = [64, 128, 256, 512]  # 🔧 更多通道

    # ===== 改进的 MDN-RNN 设置 =====
    hidden_size = 512  # 🔧 从 256 增加到 512
    n_gaussians = 7    # 🔧 从 5 增加到 7（更灵活的分布）

    if MODE == "improved":
        # ===== 改进的完整版本 =====
        # 数据收集（改进策略）
        random_rollouts = 10000
        max_steps_per_episode = 1000
        use_heuristic_data = True  # 🔧 混合启发式数据

        # VAE 训练（改进）
        vae_epochs = 20           # 🔧 增加到 20
        vae_batch_size = 128      # 🔧 增加 batch size
        vae_lr = 2e-4             # 🔧 稍高学习率
        vae_beta = 0.5            # 🔧 KL 权重（防止 posterior collapse）
        use_data_aug = True       # 🔧 数据增强

        # MDN-RNN 训练（改进）
        rnn_epochs = 30           # 🔧 增加到 30
        rnn_batch_size = 128
        rnn_seq_len = 999
        rnn_lr = 2e-4
        rnn_grad_clip = 5.0       # 🔧 梯度裁剪

        # CMA-ES 设置（改进）
        population_size = 128     # 🔧 增加种群
        generations = 500         # 🔧 更多代数
        dream_rollout_length = 1000
        n_rollouts_per_eval = 32  # 🔧 更多评估
        temperature_candidates = [0.5, 0.8, 1.0, 1.2, 1.5]  # 🔧 温度探索
        use_mlp_controller = True  # 🔧 非线性控制器

        save_dir = "./results_car_racing_improved"

    else:  # improved_quick
        # ===== 快速测试版本 =====
        random_rollouts = 2000
        max_steps_per_episode = 500
        use_heuristic_data = True

        vae_epochs = 15
        vae_batch_size = 128
        vae_lr = 2e-4
        vae_beta = 0.5
        use_data_aug = True

        rnn_epochs = 20
        rnn_batch_size = 128
        rnn_seq_len = 499
        rnn_lr = 2e-4
        rnn_grad_clip = 5.0

        population_size = 64
        generations = 150
        dream_rollout_length = 500
        n_rollouts_per_eval = 16
        temperature_candidates = [0.8, 1.0, 1.2]
        use_mlp_controller = True

        save_dir = "./results_car_racing_improved_quick"

    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint 设置
    checkpoint_interval = 50
    resume_from_checkpoint = True


# ========== 改进的 VAE ==========
class ImprovedConvVAE(nn.Module):
    """
    改进的 VAE：
    1. 更多通道 (64 -> 512)
    2. Batch Normalization
    3. Residual connections
    4. 可调 KL 权重
    """
    def __init__(self, latent_size=64, channels=[64, 128, 256, 512]):
        super().__init__()
        self.latent_size = latent_size
        self.channels = channels

        # Encoder: 64x64x3 -> latent_size
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),  # 🔧 Batch Norm
                nn.LeakyReLU(0.2),       # 🔧 LeakyReLU
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # 计算 feature map 大小: 64 -> 32 -> 16 -> 8 -> 4
        self.feature_size = channels[-1] * 4 * 4

        # Latent
        self.fc_mu = nn.Linear(self.feature_size, latent_size)
        self.fc_logvar = nn.Linear(self.feature_size, latent_size)

        # Decoder
        self.fc_decode = nn.Linear(latent_size, self.feature_size)

        dec_layers = []
        channels_rev = list(reversed(channels))
        for i in range(len(channels_rev) - 1):
            dec_layers.extend([
                nn.ConvTranspose2d(channels_rev[i], channels_rev[i+1], 4, stride=2, padding=1),
                nn.BatchNorm2d(channels_rev[i+1]),
                nn.LeakyReLU(0.2),
            ])
        # 最后一层
        dec_layers.append(nn.ConvTranspose2d(channels_rev[-1], 3, 4, stride=2, padding=1))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.reshape(-1, self.channels[-1], 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE loss with adjustable KL weight"""
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


# ========== 数据增强 ==========
class DataAugmentation:
    """简单的数据增强"""
    @staticmethod
    def augment(frames_batch):
        """
        frames_batch: (B, 3, 64, 64) tensor
        """
        # 随机水平翻转
        if np.random.rand() > 0.5:
            frames_batch = torch.flip(frames_batch, dims=[3])

        # 随机亮度调整
        brightness = 0.8 + np.random.rand() * 0.4  # [0.8, 1.2]
        frames_batch = torch.clamp(frames_batch * brightness, 0, 1)

        # 确保 tensor 是连续的，避免 view/reshape 错误
        return frames_batch.contiguous()


# ========== 改进的 MDN-RNN（与原版基本相同，增加梯度裁剪）==========
class MDNRNN(nn.Module):
    """MDN-RNN: 保持论文架构，但增加 hidden size"""
    def __init__(self, latent_size, action_size, hidden_size, n_gaussians):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.n_gaussians = n_gaussians

        input_size = latent_size + action_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc_pi = nn.Linear(hidden_size, n_gaussians)
        self.fc_mu = nn.Linear(hidden_size, n_gaussians * latent_size)
        self.fc_sigma = nn.Linear(hidden_size, n_gaussians * latent_size)

        self.fc_reward = nn.Linear(hidden_size, 1)
        self.fc_done = nn.Linear(hidden_size, 1)

    def forward(self, z, action, hidden=None):
        batch_size, seq_len, _ = z.shape

        x = torch.cat([z, action], dim=-1)
        lstm_out, hidden = self.lstm(x, hidden)

        out_flat = lstm_out.reshape(batch_size * seq_len, -1)

        pi = F.softmax(self.fc_pi(out_flat), dim=-1)
        mu = self.fc_mu(out_flat).view(-1, self.n_gaussians, self.latent_size)
        sigma = torch.exp(self.fc_sigma(out_flat)).view(-1, self.n_gaussians, self.latent_size)
        sigma = torch.clamp(sigma, min=1e-4, max=10.0)

        reward = self.fc_reward(out_flat)
        done = torch.sigmoid(self.fc_done(out_flat))

        pi = pi.view(batch_size, seq_len, -1)
        mu = mu.view(batch_size, seq_len, self.n_gaussians, self.latent_size)
        sigma = sigma.view(batch_size, seq_len, self.n_gaussians, self.latent_size)
        reward = reward.view(batch_size, seq_len, 1)
        done = done.view(batch_size, seq_len, 1)

        return pi, mu, sigma, reward, done, hidden

    def mdn_loss(self, pi, mu, sigma, target):
        batch_size, seq_len, _ = target.shape
        target = target.view(batch_size * seq_len, 1, -1)
        pi = pi.view(batch_size * seq_len, -1)
        mu = mu.view(batch_size * seq_len, self.n_gaussians, -1)
        sigma = sigma.view(batch_size * seq_len, self.n_gaussians, -1)

        var = sigma ** 2
        log_prob = -0.5 * (math.log(2 * math.pi) + torch.log(var) + (target - mu) ** 2 / var)
        log_prob = log_prob.sum(dim=-1)

        log_pi = torch.log(pi + 1e-8)
        log_prob_mixture = torch.logsumexp(log_pi + log_prob, dim=-1)

        return -log_prob_mixture.mean()

    def sample(self, pi, mu, sigma, temperature=1.0):
        batch_size = pi.shape[0]

        pi_temp = pi ** (1.0 / temperature)
        pi_temp = pi_temp / pi_temp.sum(dim=-1, keepdim=True)

        idx = torch.multinomial(pi_temp, 1).squeeze(-1)

        batch_idx = torch.arange(batch_size, device=mu.device)
        mu_sel = mu[batch_idx, idx]
        sigma_sel = sigma[batch_idx, idx] * temperature

        eps = torch.randn_like(mu_sel)
        return mu_sel + sigma_sel * eps


# ========== 改进的 Controller (MLP) ==========
class MLPController:
    """
    🔧 非线性 MLP Controller 替代纯线性
    Architecture: input -> hidden(64) -> ReLU -> output
    """
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 两层网络
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(action_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(action_dim)

    def get_action(self, state):
        # Layer 1
        h = np.maximum(0, self.W1 @ state + self.b1)  # ReLU

        # Layer 2
        raw = self.W2 @ h + self.b2

        # Output activation
        action = np.zeros(3)
        action[0] = np.tanh(raw[0])  # steering
        action[1] = 1.0 / (1.0 + np.exp(-raw[1]))  # gas
        action[2] = 1.0 / (1.0 + np.exp(-raw[2]))  # brake
        return action

    def get_params(self):
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2
        ])

    def set_params(self, params):
        w1_size = self.hidden_dim * self.input_dim
        b1_size = self.hidden_dim
        w2_size = self.action_dim * self.hidden_dim
        b2_size = self.action_dim

        idx = 0
        self.W1 = params[idx:idx+w1_size].reshape(self.hidden_dim, self.input_dim)
        idx += w1_size
        self.b1 = params[idx:idx+b1_size]
        idx += b1_size
        self.W2 = params[idx:idx+w2_size].reshape(self.action_dim, self.hidden_dim)
        idx += w2_size
        self.b2 = params[idx:idx+b2_size]

    @property
    def num_params(self):
        return (self.hidden_dim * self.input_dim + self.hidden_dim +
                self.action_dim * self.hidden_dim + self.action_dim)


# ========== Linear Controller (Fallback) ==========
class LinearController:
    """原始线性 controller（作为对比）"""
    def __init__(self, input_dim, action_dim):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.W = np.random.randn(action_dim, input_dim) * 0.1
        self.b = np.zeros(action_dim)

    def get_action(self, state):
        raw = self.W @ state + self.b
        action = np.zeros(3)
        action[0] = np.tanh(raw[0])
        action[1] = 1.0 / (1.0 + np.exp(-raw[1]))
        action[2] = 1.0 / (1.0 + np.exp(-raw[2]))
        return action

    def get_params(self):
        return np.concatenate([self.W.flatten(), self.b])

    def set_params(self, params):
        w_size = self.action_dim * self.input_dim
        self.W = params[:w_size].reshape(self.action_dim, self.input_dim)
        self.b = params[w_size:]

    @property
    def num_params(self):
        return self.action_dim * self.input_dim + self.action_dim


# ========== CMA-ES (保持不变) ==========
class SimpleCMAES:
    """Simplified CMA-ES"""
    def __init__(self, dim, pop_size=64, sigma=0.5):
        self.dim = dim
        self.pop_size = pop_size
        self.elite_size = pop_size // 4

        self.mean = np.zeros(dim)
        self.sigma = sigma
        self.C = np.ones(dim)

    def ask(self):
        noise = np.random.randn(self.pop_size, self.dim)
        return self.mean + self.sigma * np.sqrt(self.C) * noise

    def tell(self, population, fitness):
        elite_idx = np.argsort(fitness)[-self.elite_size:]
        elite = population[elite_idx]
        elite_fit = fitness[elite_idx]

        weights = np.exp(elite_fit - elite_fit.max())
        weights = weights / weights.sum()

        self.mean = (weights[:, None] * elite).sum(axis=0)

        diff = elite - self.mean
        self.C = 0.8 * self.C + 0.2 * (weights[:, None] * diff ** 2).sum(axis=0)


# ========== 改进的 World Model Agent ==========
class ImprovedCarRacingWorldModel:
    """
    改进版 World Model Agent
    保持三阶段框架，但每个阶段都有优化
    """
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.logger = setup_logging(config.save_dir)
        self.logger.info("=" * 60)
        self.logger.info("Improved World Models: CarRacing")
        self.logger.info("=" * 60)
        self.logger.info(f"  Latent size: {config.latent_size} (improved from 32)")
        self.logger.info(f"  Hidden size: {config.hidden_size} (improved from 256)")
        self.logger.info(f"  Controller: {'MLP' if config.use_mlp_controller else 'Linear'}")
        self.logger.info(f"  Data augmentation: {config.use_data_aug}")
        self.logger.info("=" * 60)

        # Models
        self.vae = ImprovedConvVAE(
            config.latent_size,
            config.vae_channels
        ).to(self.device)

        self.rnn = MDNRNN(
            config.latent_size, 3,
            config.hidden_size, config.n_gaussians
        ).to(self.device)

        # 数据存储（分块）
        self.chunk_size = 100
        self.current_chunk_frames = []
        self.current_chunk_actions = []
        self.current_chunk_rewards = []
        self.current_chunk_dones = []
        self.current_chunk_episode_ends = []
        self.num_chunks_saved = 0

        self.z_data = None
        self.action_data = None
        self.reward_data = None
        self.done_data = None

        # History
        self.history = {
            "vae_loss": [],
            "rnn_loss": [],
            "dream_fitness": [],
            "real_reward": [],
            "temperature_used": []  # 🔧 记录温度
        }

        # Training state
        self.training_state = {
            "stage": "init",
            "data_collection_progress": 0,
            "vae_epoch": 0,
            "rnn_epoch": 0,
            "cmaes_generation": 0,
            "start_time": None,
            "elapsed_time": 0,
            "num_chunks": 0,
            "best_temperature": 1.0  # 🔧 记录最佳温度
        }

        self.cmaes_state = None
        self.best_controller_params = None
        self.best_fitness = -float('inf')

    def get_data_dir(self):
        data_dir = f"{self.config.save_dir}/data_chunks"
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def save_current_chunk(self):
        """保存当前 chunk"""
        if len(self.current_chunk_frames) == 0:
            return

        data_dir = self.get_data_dir()
        chunk_path = f"{data_dir}/chunk_{self.num_chunks_saved:04d}.npz"

        frames_array = (np.array(self.current_chunk_frames) * 255).astype(np.uint8)
        actions_array = np.array(self.current_chunk_actions, dtype=np.float32)
        rewards_array = np.array(self.current_chunk_rewards, dtype=np.float32)
        dones_array = np.array(self.current_chunk_dones, dtype=np.float32)
        episode_ends_array = np.array(self.current_chunk_episode_ends, dtype=np.int32)

        np.savez_compressed(
            chunk_path,
            frames=frames_array,
            actions=actions_array,
            rewards=rewards_array,
            dones=dones_array,
            episode_ends=episode_ends_array
        )

        self.current_chunk_frames = []
        self.current_chunk_actions = []
        self.current_chunk_rewards = []
        self.current_chunk_dones = []
        self.current_chunk_episode_ends = []
        self.num_chunks_saved += 1
        self.training_state["num_chunks"] = self.num_chunks_saved

    def get_memory_usage(self):
        """获取当前内存使用情况"""
        import gc
        gc.collect()
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            return f"{mem_mb:.0f} MB"
        except ImportError:
            return "N/A"

    def estimate_remaining_time(self):
        """估算剩余时间"""
        if self.training_state["start_time"] is None:
            return "Unknown"

        elapsed = time.time() - self.training_state["start_time"]
        stage = self.training_state["stage"]

        # 根据当前阶段估算总进度
        progress = 0
        if stage == "data_collection":
            progress = 0.1 * (self.training_state["data_collection_progress"] / self.config.random_rollouts)
        elif stage == "vae_training":
            progress = 0.1 + 0.2 * (self.training_state["vae_epoch"] / self.config.vae_epochs)
        elif stage == "rnn_training":
            progress = 0.3 + 0.25 * (self.training_state["rnn_epoch"] / self.config.rnn_epochs)
        elif stage == "controller_training":
            progress = 0.55 + 0.45 * (self.training_state["cmaes_generation"] / self.config.generations)
        elif stage == "done":
            progress = 1.0

        if progress > 0:
            total_estimated = elapsed / progress
            remaining = total_estimated - elapsed
            return str(timedelta(seconds=int(remaining)))
        return "Calculating..."

    def get_checkpoint_path(self):
        return f"{self.config.save_dir}/checkpoint.pkl"

    def save_checkpoint(self):
        if len(self.current_chunk_frames) > 0:
            self.save_current_chunk()

        checkpoint = {
            "training_state": self.training_state,
            "history": self.history,
            "vae_state": self.vae.state_dict(),
            "rnn_state": self.rnn.state_dict(),
            "cmaes_state": self.cmaes_state,
            "best_controller_params": self.best_controller_params,
            "best_fitness": self.best_fitness,
            "config_mode": MODE,
            "num_chunks_saved": self.num_chunks_saved,
        }

        checkpoint_path = self.get_checkpoint_path()
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self):
        checkpoint_path = self.get_checkpoint_path()

        if not os.path.exists(checkpoint_path):
            self.logger.info("No checkpoint found, starting fresh")
            return False

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            if checkpoint.get("config_mode") != MODE:
                self.logger.warning(f"Checkpoint mode mismatch. Starting fresh.")
                return False

            self.training_state = checkpoint["training_state"]
            self.history = checkpoint["history"]
            self.vae.load_state_dict(checkpoint["vae_state"])
            self.rnn.load_state_dict(checkpoint["rnn_state"])
            self.cmaes_state = checkpoint.get("cmaes_state")
            self.best_controller_params = checkpoint.get("best_controller_params")
            self.best_fitness = checkpoint.get("best_fitness", -float('inf'))
            self.num_chunks_saved = checkpoint.get("num_chunks_saved", 0)

            self.logger.info(f"Checkpoint loaded successfully")
            self.logger.info(f"  Stage: {self.training_state['stage']}")
            self.logger.info(f"  CMA-ES generation: {self.training_state['cmaes_generation']}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def preprocess_frame(self, frame):
        img = Image.fromarray(frame)
        img = img.resize((self.config.img_size, self.config.img_size))
        return np.array(img) / 255.0

    def collect_data(self):
        """
        🔧 改进的数据收集：
        - 混合随机策略（保持探索）
        - 简单启发式（让车往前开）
        """
        self.training_state["stage"] = "data_collection"
        start_ep = self.training_state["data_collection_progress"]

        self.logger.info("=" * 60)
        self.logger.info("Stage 1: Improved Data Collection")
        self.logger.info(f"  Heuristic data: {self.config.use_heuristic_data}")
        self.logger.info("=" * 60)

        env = gym.make(self.config.env_name, render_mode=None)
        rollouts_in_current_chunk = start_ep % self.chunk_size

        for ep in range(start_ep, self.config.random_rollouts):
            obs, _ = env.reset()
            ep_start_idx = len(self.current_chunk_frames)

            # 🔧 50% 随机，50% 启发式（简单往前）
            use_heuristic = self.config.use_heuristic_data and (ep % 2 == 0)

            for step in range(self.config.max_steps_per_episode):
                if use_heuristic:
                    # 简单启发式：gas=0.5, steering 随机小幅度
                    action = np.array([
                        np.random.uniform(-0.3, 0.3),  # steering
                        0.5 + np.random.uniform(-0.2, 0.2),  # gas
                        0.0  # brake
                    ])
                else:
                    action = env.action_space.sample()

                frame = self.preprocess_frame(obs)
                self.current_chunk_frames.append(frame)
                self.current_chunk_actions.append(action)

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.current_chunk_rewards.append(reward)
                self.current_chunk_dones.append(float(done))

                if done:
                    break

            self.current_chunk_episode_ends.append(len(self.current_chunk_frames))
            rollouts_in_current_chunk += 1
            self.training_state["data_collection_progress"] = ep + 1

            if rollouts_in_current_chunk >= self.chunk_size:
                self.save_current_chunk()
                rollouts_in_current_chunk = 0

            if (ep + 1) % 50 == 0:
                remaining = self.estimate_remaining_time()
                mem_usage = self.get_memory_usage()
                self.logger.info(f"  Rollout {ep+1}/{self.config.random_rollouts} | "
                               f"Chunks: {self.num_chunks_saved} | "
                               f"Memory: {mem_usage} | ETA: {remaining}")

            if (ep + 1) % 500 == 0:
                self.save_checkpoint()

        if len(self.current_chunk_frames) > 0:
            self.save_current_chunk()

        env.close()
        self.logger.info(f"Data collection complete: {self.num_chunks_saved} chunks")

    def train_vae(self):
        """
        🔧 改进的 VAE 训练：
        - 数据增强
        - Cosine Annealing LR
        - KL 权重控制
        """
        self.training_state["stage"] = "vae_training"
        start_epoch = self.training_state["vae_epoch"]

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Stage 2a: Training Improved VAE")
        self.logger.info(f"  Data augmentation: {self.config.use_data_aug}")
        self.logger.info(f"  KL beta: {self.config.vae_beta}")
        self.logger.info("=" * 60)

        optimizer = optim.Adam(self.vae.parameters(), lr=self.config.vae_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.vae_epochs
        )

        data_dir = self.get_data_dir()
        chunk_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

        for epoch in range(start_epoch, self.config.vae_epochs):
            epoch_loss = 0
            n_batches = 0

            np.random.shuffle(chunk_files)

            for chunk_file in chunk_files:
                chunk_path = os.path.join(data_dir, chunk_file)
                with np.load(chunk_path) as data:
                    frames = data['frames'].astype(np.float32) / 255.0

                frames = torch.FloatTensor(frames).permute(0, 3, 1, 2).to(self.device)
                n_samples = len(frames)

                idx = np.random.permutation(n_samples)

                for i in range(0, n_samples - self.config.vae_batch_size, self.config.vae_batch_size):
                    batch_idx = idx[i:i+self.config.vae_batch_size]
                    batch = frames[batch_idx]

                    # 🔧 数据增强
                    if self.config.use_data_aug:
                        batch = DataAugmentation.augment(batch)

                    recon, mu, logvar, _ = self.vae(batch)
                    loss = vae_loss(recon, batch, mu, logvar, beta=self.config.vae_beta)
                    loss = loss / batch.size(0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                del frames

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history["vae_loss"].append(avg_loss)
            self.training_state["vae_epoch"] = epoch + 1

            scheduler.step()  # 🔧 学习率调度

            remaining = self.estimate_remaining_time()
            mem_usage = self.get_memory_usage()
            self.logger.info(f"  Epoch {epoch+1}/{self.config.vae_epochs} | "
                           f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
                           f"Memory: {mem_usage} | ETA: {remaining}")

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint()

        self.logger.info("VAE training complete")

    def train_rnn(self):
        """🔧 改进的 RNN 训练：梯度裁剪、学习率调度"""
        self.training_state["stage"] = "rnn_training"
        start_epoch = self.training_state["rnn_epoch"]

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Stage 2b: Training MDN-RNN")
        self.logger.info(f"  Gradient clipping: {self.config.rnn_grad_clip}")
        self.logger.info("=" * 60)

        optimizer = optim.Adam(self.rnn.parameters(), lr=self.config.rnn_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.rnn_epochs
        )

        data_dir = self.get_data_dir()
        chunk_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

        # 编码所有数据
        z_path = f"{self.config.save_dir}/encoded_z.npz"

        if not os.path.exists(z_path) or start_epoch == 0:
            self.logger.info("  Encoding frames with improved VAE...")
            all_z = []
            all_actions = []
            all_rewards = []
            all_dones = []

            for chunk_idx, chunk_file in enumerate(chunk_files):
                chunk_path = os.path.join(data_dir, chunk_file)
                with np.load(chunk_path) as data:
                    frames = data['frames'].astype(np.float32) / 255.0
                    actions = data['actions']
                    rewards = data['rewards']
                    dones = data['dones']

                frames_t = torch.FloatTensor(frames).permute(0, 3, 1, 2).to(self.device)
                with torch.no_grad():
                    z_chunk = []
                    for i in range(0, len(frames_t), 256):
                        batch = frames_t[i:i+256]
                        mu, _ = self.vae.encode(batch)
                        z_chunk.append(mu.cpu().numpy())
                    z_chunk = np.concatenate(z_chunk, axis=0)

                all_z.append(z_chunk)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_dones.append(dones)

                del frames_t
                if (chunk_idx + 1) % 10 == 0:
                    self.logger.info(f"    Encoded chunk {chunk_idx+1}/{len(chunk_files)}")

            all_z = np.concatenate(all_z, axis=0)
            all_actions = np.concatenate(all_actions, axis=0)
            all_rewards = np.concatenate(all_rewards, axis=0)
            all_dones = np.concatenate(all_dones, axis=0)

            np.savez_compressed(z_path, z=all_z, actions=all_actions,
                              rewards=all_rewards, dones=all_dones)
            self.logger.info(f"  Saved encoded data: {os.path.getsize(z_path)/(1024*1024):.1f} MB")
        else:
            self.logger.info(f"  Loading pre-encoded data...")
            data = np.load(z_path)
            all_z = data['z']
            all_actions = data['actions']
            all_rewards = data['rewards']
            all_dones = data['dones']

        z_all = torch.FloatTensor(all_z)
        actions = torch.FloatTensor(all_actions)
        rewards = torch.FloatTensor(all_rewards)
        dones = torch.FloatTensor(all_dones)

        n_samples = len(z_all) - self.config.rnn_seq_len - 1

        for epoch in range(start_epoch, self.config.rnn_epochs):
            epoch_loss = 0
            n_batches = 0

            for _ in range(100):
                starts = np.random.randint(0, n_samples, self.config.rnn_batch_size)

                z_seq = []
                a_seq = []
                z_next = []
                r_seq = []
                d_seq = []

                for s in starts:
                    z_seq.append(z_all[s:s+self.config.rnn_seq_len])
                    a_seq.append(actions[s:s+self.config.rnn_seq_len])
                    z_next.append(z_all[s+1:s+self.config.rnn_seq_len+1])
                    r_seq.append(rewards[s:s+self.config.rnn_seq_len])
                    d_seq.append(dones[s:s+self.config.rnn_seq_len])

                z_seq = torch.stack(z_seq).to(self.device)
                a_seq = torch.stack(a_seq).to(self.device)
                z_next = torch.stack(z_next).to(self.device)
                r_seq = torch.stack(r_seq).unsqueeze(-1).to(self.device)
                d_seq = torch.stack(d_seq).unsqueeze(-1).to(self.device)

                pi, mu, sigma, pred_r, pred_d, _ = self.rnn(z_seq, a_seq)

                mdn_loss = self.rnn.mdn_loss(pi, mu, sigma, z_next)
                reward_loss = F.mse_loss(pred_r, r_seq)
                done_loss = F.binary_cross_entropy(pred_d, d_seq)

                loss = mdn_loss + reward_loss + done_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.config.rnn_grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history["rnn_loss"].append(avg_loss)
            self.training_state["rnn_epoch"] = epoch + 1

            scheduler.step()

            remaining = self.estimate_remaining_time()
            mem_usage = self.get_memory_usage()
            self.logger.info(f"  Epoch {epoch+1}/{self.config.rnn_epochs} | "
                           f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
                           f"Memory: {mem_usage} | ETA: {remaining}")

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint()

        self.logger.info("MDN-RNN training complete")

    def dream_rollout(self, controller, temperature=1.0):
        """梦境 rollout（保持与原版相同）"""
        z_path = f"{self.config.save_dir}/encoded_z.npz"
        if not hasattr(self, '_cached_z_for_dream') or self._cached_z_for_dream is None:
            if os.path.exists(z_path):
                data = np.load(z_path)
                self._cached_z_for_dream = data['z']
                self.logger.info(f"  Loaded {len(self._cached_z_for_dream)} z vectors")
            else:
                raise RuntimeError("No encoded z data found.")

        idx = np.random.randint(0, len(self._cached_z_for_dream))
        z = torch.FloatTensor(self._cached_z_for_dream[idx]).to(self.device)

        hidden = None
        total_reward = 0

        for _ in range(self.config.dream_rollout_length):
            if hidden is None:
                h = torch.zeros(self.config.hidden_size).to(self.device)
            else:
                h = hidden[0].squeeze(0).squeeze(0)

            ctrl_input = torch.cat([z, h]).detach().cpu().numpy()
            action = controller.get_action(ctrl_input)
            action_t = torch.FloatTensor(action).to(self.device)

            z_in = z.unsqueeze(0).unsqueeze(0)
            a_in = action_t.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                pi, mu, sigma, reward, done, hidden = self.rnn(z_in, a_in, hidden)

            pi = pi.squeeze(0).squeeze(0)
            mu = mu.squeeze(0).squeeze(0)
            sigma = sigma.squeeze(0).squeeze(0)

            z = self.rnn.sample(pi.unsqueeze(0), mu.unsqueeze(0), sigma.unsqueeze(0), temperature)
            z = z.squeeze(0)

            total_reward += reward.item()

            if done.item() > 0.5:
                break

        return total_reward

    def train_controller(self):
        """
        🔧 改进的 Controller 训练：
        - MLP controller
        - 温度探索
        - 更大种群
        """
        self.training_state["stage"] = "controller_training"
        start_gen = self.training_state["cmaes_generation"]

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Stage 3: Training Improved Controller")
        self.logger.info(f"  Controller type: {'MLP' if self.config.use_mlp_controller else 'Linear'}")
        self.logger.info(f"  Temperature candidates: {self.config.temperature_candidates}")
        self.logger.info("=" * 60)

        input_dim = self.config.latent_size + self.config.hidden_size

        # 🔧 选择控制器类型
        if self.config.use_mlp_controller:
            controller = MLPController(input_dim, 3, hidden_dim=64)
        else:
            controller = LinearController(input_dim, 3)

        self.logger.info(f"  Controller params: {controller.num_params}")

        # 恢复或创建 CMA-ES
        if self.cmaes_state is not None and start_gen > 0:
            cmaes = SimpleCMAES(controller.num_params, self.config.population_size)
            cmaes.mean = self.cmaes_state["mean"]
            cmaes.sigma = self.cmaes_state["sigma"]
            cmaes.C = self.cmaes_state["C"]
            self.logger.info("  Restored CMA-ES state")
        else:
            cmaes = SimpleCMAES(controller.num_params, self.config.population_size)

        best_controller = None
        if self.best_controller_params is not None:
            if self.config.use_mlp_controller:
                best_controller = MLPController(input_dim, 3, hidden_dim=64)
            else:
                best_controller = LinearController(input_dim, 3)
            best_controller.set_params(self.best_controller_params)

        # 🔧 温度探索：每 50 代测试不同温度
        current_temp = self.training_state.get("best_temperature", 1.0)

        for gen in range(start_gen, self.config.generations):
            population = cmaes.ask()
            fitness = []

            for params in population:
                controller.set_params(params)
                rewards = [self.dream_rollout(controller, current_temp)
                          for _ in range(self.config.n_rollouts_per_eval)]
                fitness.append(np.mean(rewards))

            fitness = np.array(fitness)
            cmaes.tell(population, fitness)

            gen_best = fitness.max()
            gen_mean = fitness.mean()

            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
                if self.config.use_mlp_controller:
                    best_controller = MLPController(input_dim, 3, hidden_dim=64)
                else:
                    best_controller = LinearController(input_dim, 3)
                best_controller.set_params(population[fitness.argmax()])
                self.best_controller_params = population[fitness.argmax()].copy()

            self.history["dream_fitness"].append(gen_best)
            self.history["temperature_used"].append(current_temp)
            self.training_state["cmaes_generation"] = gen + 1

            self.cmaes_state = {
                "mean": cmaes.mean.copy(),
                "sigma": cmaes.sigma,
                "C": cmaes.C.copy()
            }

            # 🔧 温度探索
            if (gen + 1) % 50 == 0 and gen < self.config.generations - 50:
                self.logger.info(f"  === Temperature exploration at gen {gen+1} ===")
                temp_results = {}
                for temp in self.config.temperature_candidates:
                    rewards = [self.dream_rollout(best_controller, temp)
                              for _ in range(8)]
                    temp_results[temp] = np.mean(rewards)
                    self.logger.info(f"    Temp {temp}: {temp_results[temp]:.1f}")

                best_temp = max(temp_results, key=temp_results.get)
                if temp_results[best_temp] > temp_results.get(current_temp, -1e9):
                    current_temp = best_temp
                    self.training_state["best_temperature"] = current_temp
                    self.logger.info(f"  → Switching to temperature {current_temp}")

            if (gen + 1) % 10 == 0:
                self.logger.info(f"  Gen {gen+1}/{self.config.generations} | "
                               f"Best: {gen_best:.1f} | Mean: {gen_mean:.1f} | "
                               f"All-time: {self.best_fitness:.1f} | Temp: {current_temp}")

            if (gen + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

        self.logger.info(f"Controller training complete | Best: {self.best_fitness:.1f}")
        return best_controller

    def evaluate_real(self, controller, n_episodes=10):
        """真实环境评估"""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Evaluating in real environment")
        self.logger.info("=" * 60)

        env = gym.make(self.config.env_name, render_mode=None)
        rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            hidden = None
            total_reward = 0

            frame = self.preprocess_frame(obs)
            frame_t = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                mu, _ = self.vae.encode(frame_t)
                z = mu.squeeze(0)

            for step in range(self.config.max_steps_per_episode):
                if hidden is None:
                    h = torch.zeros(self.config.hidden_size).to(self.device)
                else:
                    h = hidden[0].squeeze(0).squeeze(0)

                ctrl_input = torch.cat([z, h]).detach().cpu().numpy()
                action = controller.get_action(ctrl_input)

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

                if done:
                    break

                frame = self.preprocess_frame(obs)
                frame_t = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    mu, _ = self.vae.encode(frame_t)
                    z = mu.squeeze(0)

                    action_t = torch.FloatTensor(action).to(self.device)
                    _, _, _, _, _, hidden = self.rnn(
                        z.unsqueeze(0).unsqueeze(0),
                        action_t.unsqueeze(0).unsqueeze(0),
                        hidden
                    )

            rewards.append(total_reward)
            self.logger.info(f"  Episode {ep+1}/{n_episodes}: {total_reward:.1f}")

        env.close()

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        self.history["real_reward"].append(mean_reward)

        self.logger.info(f"Real environment: {mean_reward:.1f} +/- {std_reward:.1f}")
        return mean_reward, std_reward

    def train(self):
        """完整训练流程"""
        resumed = False
        if self.config.resume_from_checkpoint:
            resumed = self.load_checkpoint()

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Improved World Models Training")
        self.logger.info("=" * 60)
        self.logger.info(f"  Mode: {MODE}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Resumed: {resumed}")
        self.logger.info("=" * 60)

        if self.training_state["start_time"] is None:
            self.training_state["start_time"] = time.time()

        stage = self.training_state["stage"]

        # Stage 1
        if stage in ["init", "data_collection"]:
            if self.training_state["data_collection_progress"] < self.config.random_rollouts:
                self.collect_data()

        # Stage 2a
        if stage in ["init", "data_collection", "vae_training"]:
            if self.training_state["vae_epoch"] < self.config.vae_epochs:
                self.train_vae()

        # Stage 2b
        if stage in ["init", "data_collection", "vae_training", "rnn_training"]:
            if self.training_state["rnn_epoch"] < self.config.rnn_epochs:
                self.train_rnn()

        # Stage 3
        if stage in ["init", "data_collection", "vae_training", "rnn_training", "controller_training"]:
            if self.training_state["cmaes_generation"] < self.config.generations:
                controller = self.train_controller()
            else:
                input_dim = self.config.latent_size + self.config.hidden_size
                if self.config.use_mlp_controller:
                    controller = MLPController(input_dim, 3, hidden_dim=64)
                else:
                    controller = LinearController(input_dim, 3)
                if self.best_controller_params is not None:
                    controller.set_params(self.best_controller_params)

        self.training_state["stage"] = "done"

        # Evaluate
        self.evaluate_real(controller, n_episodes=20)

        # Save
        self.save(controller)

        total_time = time.time() - self.training_state["start_time"]
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Training Complete!")
        self.logger.info(f"  Total time: {timedelta(seconds=int(total_time))}")
        self.logger.info(f"  Best dream fitness: {self.best_fitness:.1f}")
        self.logger.info(f"  Real env reward: {self.history['real_reward'][-1]:.1f}")
        self.logger.info("=" * 60)

        return controller

    def save(self, controller):
        """保存模型和结果"""
        os.makedirs(self.config.save_dir, exist_ok=True)

        torch.save({
            'vae': self.vae.state_dict(),
            'rnn': self.rnn.state_dict()
        }, f"{self.config.save_dir}/models.pt")

        if hasattr(controller, 'W1'):  # MLP
            np.savez(f"{self.config.save_dir}/controller.npz",
                    W1=controller.W1, b1=controller.b1,
                    W2=controller.W2, b2=controller.b2,
                    type='mlp')
        else:  # Linear
            np.savez(f"{self.config.save_dir}/controller.npz",
                    W=controller.W, b=controller.b,
                    type='linear')

        with open(f"{self.config.save_dir}/history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        self.save_checkpoint()

        self.logger.info(f"Results saved to {self.config.save_dir}")


# ========== Main ==========
def main():
    config = Config()
    agent = ImprovedCarRacingWorldModel(config)
    agent.train()


if __name__ == "__main__":
    main()
