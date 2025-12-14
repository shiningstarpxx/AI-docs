"""
CarRacing World Model: Paper Reproduction
==========================================
复现 World Models (Ha & Schmidhuber, 2018) 在 CarRacing 上的实验

论文设置：
- 环境：CarRacing-v2 (96x96 RGB -> 64x64 resize)
- V: VAE with 32-dim latent space
- M: MDN-LSTM with 256 hidden units, 5 Gaussian components
- C: Linear controller trained with CMA-ES

训练流程：
1. 收集 10,000 rollouts (随机策略)
2. 训练 VAE
3. 训练 MDN-RNN
4. 在梦境中用 CMA-ES 训练 Controller

参考：
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
# 可选: "paper" (完全论文设置, 需要几小时), "medium" (中等规模), "quick" (快速测试)
MODE = "paper"  # 修改此处选择模式

class Config:
    # 环境
    env_name = "CarRacing-v3"
    img_size = 64  # 论文用 64x64

    # VAE 设置 (论文: Table 1)
    latent_size = 32  # 论文用 32

    # MDN-RNN 设置 (论文: Table 1)
    hidden_size = 256  # 论文用 256
    n_gaussians = 5    # 论文用 5

    if MODE == "paper":
        # ===== 完全论文设置 =====
        # 数据收集 (论文: 10000 rollouts, ~3M frames)
        random_rollouts = 10000    # 论文: 10000 rollouts
        max_steps_per_episode = 1000  # 论文: 最大 1000 步

        # VAE 训练 (论文设置)
        vae_epochs = 10           # 论文: ~10 epochs
        vae_batch_size = 100      # 论文: batch size 100
        vae_lr = 1e-4             # 论文: lr 0.0001

        # MDN-RNN 训练 (论文设置)
        rnn_epochs = 20           # 论文: 20 epochs
        rnn_batch_size = 100      # 论文: batch size 100
        rnn_seq_len = 999         # 论文: 完整序列
        rnn_lr = 1e-4             # 论文: lr 0.0001

        # CMA-ES 设置 (论文设置)
        population_size = 64      # 论文: 64
        generations = 300         # 论文: 300+
        dream_rollout_length = 1000  # 论文: 1000 steps
        n_rollouts_per_eval = 16  # 论文: 16 rollouts
        temperature = 1.0         # 论文: tau = 1.0

        save_dir = "./results_car_racing_paper"

    elif MODE == "medium":
        # ===== 中等规模 (约 1-2 小时) =====
        random_rollouts = 2000
        max_steps_per_episode = 500

        vae_epochs = 10
        vae_batch_size = 100
        vae_lr = 1e-4

        rnn_epochs = 15
        rnn_batch_size = 100
        rnn_seq_len = 499
        rnn_lr = 1e-4

        population_size = 64
        generations = 100
        dream_rollout_length = 500
        n_rollouts_per_eval = 8
        temperature = 1.0

        save_dir = "./results_car_racing_medium"

    else:  # quick
        # ===== 快速测试 (约 30 分钟) =====
        random_rollouts = 500
        max_steps_per_episode = 300

        vae_epochs = 10
        vae_batch_size = 64
        vae_lr = 1e-3

        rnn_epochs = 10
        rnn_batch_size = 32
        rnn_seq_len = 100
        rnn_lr = 1e-3

        population_size = 32
        generations = 50
        dream_rollout_length = 300
        n_rollouts_per_eval = 4
        temperature = 1.0

        save_dir = "./results_car_racing_quick"

    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint 设置
    checkpoint_interval = 50  # 每 50 个 rollout/generation 保存一次
    resume_from_checkpoint = True  # 是否从 checkpoint 恢复


# ========== VAE (论文架构) ==========
class ConvVAE(nn.Module):
    """
    Convolutional VAE for 64x64 RGB images

    论文架构：
    - Encoder: 4 conv layers
    - Latent: 32-dim Gaussian
    - Decoder: 4 deconv layers
    """
    def __init__(self, latent_size=32):
        super().__init__()
        self.latent_size = latent_size

        # Encoder: 64x64x3 -> 32 dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8 -> 4
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_size)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_size)

        # Decoder
        self.fc_decode = nn.Linear(latent_size, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 32 -> 64
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)  # Use reshape instead of view
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.reshape(-1, 256, 4, 4)  # Use reshape instead of view
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE loss = Reconstruction + KL divergence"""
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


# ========== MDN-RNN (论文架构) ==========
class MDNRNN(nn.Module):
    """
    MDN-RNN: LSTM + Mixture Density Network

    输入: (z_t, a_t)
    输出: P(z_{t+1}) = mixture of Gaussians + reward + done
    """
    def __init__(self, latent_size, action_size, hidden_size, n_gaussians):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.n_gaussians = n_gaussians

        # LSTM
        input_size = latent_size + action_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # MDN outputs
        self.fc_pi = nn.Linear(hidden_size, n_gaussians)
        self.fc_mu = nn.Linear(hidden_size, n_gaussians * latent_size)
        self.fc_sigma = nn.Linear(hidden_size, n_gaussians * latent_size)

        # Reward and done prediction
        self.fc_reward = nn.Linear(hidden_size, 1)
        self.fc_done = nn.Linear(hidden_size, 1)

    def forward(self, z, action, hidden=None):
        """
        Args:
            z: (batch, seq_len, latent_size)
            action: (batch, seq_len, action_size)
            hidden: LSTM hidden state
        """
        batch_size, seq_len, _ = z.shape

        x = torch.cat([z, action], dim=-1)
        lstm_out, hidden = self.lstm(x, hidden)

        # Reshape for MDN
        out_flat = lstm_out.reshape(batch_size * seq_len, -1)

        # MDN parameters
        pi = F.softmax(self.fc_pi(out_flat), dim=-1)
        mu = self.fc_mu(out_flat).view(-1, self.n_gaussians, self.latent_size)
        sigma = torch.exp(self.fc_sigma(out_flat)).view(-1, self.n_gaussians, self.latent_size)
        sigma = torch.clamp(sigma, min=1e-4, max=10.0)

        # Reward and done
        reward = self.fc_reward(out_flat)
        done = torch.sigmoid(self.fc_done(out_flat))

        # Reshape back
        pi = pi.view(batch_size, seq_len, -1)
        mu = mu.view(batch_size, seq_len, self.n_gaussians, self.latent_size)
        sigma = sigma.view(batch_size, seq_len, self.n_gaussians, self.latent_size)
        reward = reward.view(batch_size, seq_len, 1)
        done = done.view(batch_size, seq_len, 1)

        return pi, mu, sigma, reward, done, hidden

    def mdn_loss(self, pi, mu, sigma, target):
        """Negative log-likelihood of MDN"""
        batch_size, seq_len, _ = target.shape
        target = target.view(batch_size * seq_len, 1, -1)
        pi = pi.view(batch_size * seq_len, -1)
        mu = mu.view(batch_size * seq_len, self.n_gaussians, -1)
        sigma = sigma.view(batch_size * seq_len, self.n_gaussians, -1)

        # Gaussian log probability
        var = sigma ** 2
        log_prob = -0.5 * (math.log(2 * math.pi) + torch.log(var) + (target - mu) ** 2 / var)
        log_prob = log_prob.sum(dim=-1)  # sum over latent dims

        # Mixture: logsumexp for numerical stability
        log_pi = torch.log(pi + 1e-8)
        log_prob_mixture = torch.logsumexp(log_pi + log_prob, dim=-1)

        return -log_prob_mixture.mean()

    def sample(self, pi, mu, sigma, temperature=1.0):
        """Sample from MDN"""
        batch_size = pi.shape[0]

        # Temperature scaling
        pi_temp = pi ** (1.0 / temperature)
        pi_temp = pi_temp / pi_temp.sum(dim=-1, keepdim=True)

        # Sample component
        idx = torch.multinomial(pi_temp, 1).squeeze(-1)

        # Get selected mu and sigma
        batch_idx = torch.arange(batch_size, device=mu.device)
        mu_sel = mu[batch_idx, idx]
        sigma_sel = sigma[batch_idx, idx] * temperature

        # Sample from Gaussian
        eps = torch.randn_like(mu_sel)
        return mu_sel + sigma_sel * eps


# ========== Controller (论文: 简单线性) ==========
class Controller:
    """
    Linear controller: a = W @ [z, h] + b

    论文强调用简单的 controller 防止过拟合世界模型
    """
    def __init__(self, input_dim, action_dim):
        self.input_dim = input_dim
        self.action_dim = action_dim
        # 初始化参数
        self.W = np.random.randn(action_dim, input_dim) * 0.1
        self.b = np.zeros(action_dim)

    def get_action(self, state):
        """Get continuous action"""
        raw = self.W @ state + self.b
        # CarRacing actions: [steering, gas, brake]
        # steering: tanh (-1 to 1)
        # gas, brake: sigmoid (0 to 1)
        action = np.zeros(3)
        action[0] = np.tanh(raw[0])  # steering
        action[1] = 1.0 / (1.0 + np.exp(-raw[1]))  # gas
        action[2] = 1.0 / (1.0 + np.exp(-raw[2]))  # brake
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


# ========== CMA-ES ==========
class SimpleCMAES:
    """Simplified CMA-ES"""
    def __init__(self, dim, pop_size=64, sigma=0.5):
        self.dim = dim
        self.pop_size = pop_size
        self.elite_size = pop_size // 4

        self.mean = np.zeros(dim)
        self.sigma = sigma
        self.C = np.ones(dim)  # diagonal covariance

    def ask(self):
        noise = np.random.randn(self.pop_size, self.dim)
        return self.mean + self.sigma * np.sqrt(self.C) * noise

    def tell(self, population, fitness):
        # Select elite
        elite_idx = np.argsort(fitness)[-self.elite_size:]
        elite = population[elite_idx]
        elite_fit = fitness[elite_idx]

        # Weighted update
        weights = np.exp(elite_fit - elite_fit.max())
        weights = weights / weights.sum()

        self.mean = (weights[:, None] * elite).sum(axis=0)

        # Update covariance
        diff = elite - self.mean
        self.C = 0.8 * self.C + 0.2 * (weights[:, None] * diff ** 2).sum(axis=0)


# ========== World Model Agent ==========
class CarRacingWorldModel:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 设置日志
        self.logger = setup_logging(config.save_dir)
        self.logger.info("=" * 60)
        self.logger.info("World Models: CarRacing - Initializing")
        self.logger.info("=" * 60)

        # Models
        self.vae = ConvVAE(config.latent_size).to(self.device)
        self.rnn = MDNRNN(
            config.latent_size, 3,  # action_size = 3
            config.hidden_size, config.n_gaussians
        ).to(self.device)

        # ===== 内存优化：分块存储 =====
        # 不再存储原始帧，直接存储 z (latent)
        # 每个 chunk 存储 100 个 rollout 的数据（约 100MB 压缩后）
        self.chunk_size = 100  # rollouts per chunk - 减小以控制内存峰值
        self.current_chunk_frames = []
        self.current_chunk_actions = []
        self.current_chunk_rewards = []
        self.current_chunk_dones = []
        self.current_chunk_episode_ends = []  # 记录每个 episode 结束位置
        self.num_chunks_saved = 0

        # 内存中只保留用于训练的 z 序列（VAE 编码后）
        self.z_data = None  # 延迟加载
        self.action_data = None
        self.reward_data = None
        self.done_data = None

        # History
        self.history = {
            "vae_loss": [],
            "rnn_loss": [],
            "dream_fitness": [],
            "real_reward": []
        }

        # Training state for checkpoint
        self.training_state = {
            "stage": "init",  # init, data_collection, vae_training, rnn_training, controller_training, done
            "data_collection_progress": 0,
            "vae_epoch": 0,
            "rnn_epoch": 0,
            "cmaes_generation": 0,
            "start_time": None,
            "elapsed_time": 0,
            "num_chunks": 0
        }

        # CMA-ES state (for resume)
        self.cmaes_state = None
        self.best_controller_params = None
        self.best_fitness = -float('inf')

    def get_data_dir(self):
        """获取数据存储目录"""
        data_dir = f"{self.config.save_dir}/data_chunks"
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def save_current_chunk(self):
        """保存当前 chunk 到磁盘并清空内存"""
        if len(self.current_chunk_frames) == 0:
            return

        data_dir = self.get_data_dir()
        chunk_path = f"{data_dir}/chunk_{self.num_chunks_saved:04d}.npz"

        # 转换为 numpy 数组并压缩保存
        # frames: 使用 uint8 节省空间 (0-255)
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

        chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        self.logger.info(f"  Saved chunk {self.num_chunks_saved}: {len(self.current_chunk_frames)} frames, {chunk_size_mb:.1f} MB")

        # 清空内存
        self.current_chunk_frames = []
        self.current_chunk_actions = []
        self.current_chunk_rewards = []
        self.current_chunk_dones = []
        self.current_chunk_episode_ends = []
        self.num_chunks_saved += 1
        self.training_state["num_chunks"] = self.num_chunks_saved

    def load_chunk(self, chunk_idx):
        """加载指定 chunk 的数据"""
        data_dir = self.get_data_dir()
        chunk_path = f"{data_dir}/chunk_{chunk_idx:04d}.npz"

        if not os.path.exists(chunk_path):
            return None

        data = np.load(chunk_path)
        return {
            "frames": data["frames"].astype(np.float32) / 255.0,
            "actions": data["actions"],
            "rewards": data["rewards"],
            "dones": data["dones"],
            "episode_ends": data["episode_ends"]
        }

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

    def get_checkpoint_path(self):
        """获取 checkpoint 文件路径"""
        return f"{self.config.save_dir}/checkpoint.pkl"

    def save_checkpoint(self, extra_data=None):
        """保存 checkpoint（内存优化版）"""
        # 先保存当前 chunk（如果有数据）
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

        if extra_data:
            checkpoint.update(extra_data)

        checkpoint_path = self.get_checkpoint_path()
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        mem_usage = self.get_memory_usage()
        self.logger.info(f"Checkpoint saved: {checkpoint_path} | Memory: {mem_usage}")

    def load_checkpoint(self):
        """加载 checkpoint（内存优化版）"""
        checkpoint_path = self.get_checkpoint_path()

        if not os.path.exists(checkpoint_path):
            self.logger.info("No checkpoint found, starting fresh")
            return False

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            # 检查配置是否匹配
            if checkpoint.get("config_mode") != MODE:
                self.logger.warning(f"Checkpoint mode ({checkpoint.get('config_mode')}) "
                                   f"differs from current mode ({MODE}). Starting fresh.")
                return False

            self.training_state = checkpoint["training_state"]
            self.history = checkpoint["history"]
            self.vae.load_state_dict(checkpoint["vae_state"])
            self.rnn.load_state_dict(checkpoint["rnn_state"])
            self.cmaes_state = checkpoint.get("cmaes_state")
            self.best_controller_params = checkpoint.get("best_controller_params")
            self.best_fitness = checkpoint.get("best_fitness", -float('inf'))
            self.num_chunks_saved = checkpoint.get("num_chunks_saved", 0)

            # 统计已保存的数据
            data_dir = self.get_data_dir()
            total_frames = 0
            if os.path.exists(data_dir):
                for chunk_file in sorted(os.listdir(data_dir)):
                    if chunk_file.endswith('.npz'):
                        chunk_path = os.path.join(data_dir, chunk_file)
                        with np.load(chunk_path) as data:
                            total_frames += len(data['frames'])

            self.logger.info(f"Checkpoint loaded successfully")
            self.logger.info(f"  Stage: {self.training_state['stage']}")
            self.logger.info(f"  Data chunks: {self.num_chunks_saved}")
            self.logger.info(f"  Total frames: {total_frames}")
            self.logger.info(f"  VAE epochs: {self.training_state['vae_epoch']}")
            self.logger.info(f"  RNN epochs: {self.training_state['rnn_epoch']}")
            self.logger.info(f"  CMA-ES generations: {self.training_state['cmaes_generation']}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

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

    def preprocess_frame(self, frame):
        """Preprocess: resize to 64x64, normalize to [0,1]"""
        # frame is 96x96x3
        img = Image.fromarray(frame)
        img = img.resize((self.config.img_size, self.config.img_size))
        return np.array(img) / 255.0

    def collect_data(self):
        """Stage 1: Collect random rollouts（内存优化版）"""
        self.training_state["stage"] = "data_collection"
        start_ep = self.training_state["data_collection_progress"]

        self.logger.info("=" * 60)
        self.logger.info("Stage 1: Collecting random rollouts (Memory Optimized)")
        self.logger.info(f"  Starting from episode {start_ep}")
        self.logger.info(f"  Chunk size: {self.chunk_size} rollouts")
        self.logger.info("=" * 60)

        env = gym.make(self.config.env_name, render_mode=None)
        total_frames = 0
        rollouts_in_current_chunk = start_ep % self.chunk_size

        for ep in range(start_ep, self.config.random_rollouts):
            obs, _ = env.reset()
            ep_start_idx = len(self.current_chunk_frames)

            for step in range(self.config.max_steps_per_episode):
                # Random action
                action = env.action_space.sample()

                # Store frame (预处理后)
                frame = self.preprocess_frame(obs)
                self.current_chunk_frames.append(frame)
                self.current_chunk_actions.append(action)

                # Step
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.current_chunk_rewards.append(reward)
                self.current_chunk_dones.append(float(done))

                if done:
                    break

            # 记录 episode 结束位置
            self.current_chunk_episode_ends.append(len(self.current_chunk_frames))
            rollouts_in_current_chunk += 1
            total_frames += len(self.current_chunk_frames) - ep_start_idx

            self.training_state["data_collection_progress"] = ep + 1

            # 每 chunk_size 个 rollout 保存一次
            if rollouts_in_current_chunk >= self.chunk_size:
                self.save_current_chunk()
                rollouts_in_current_chunk = 0

            if (ep + 1) % 20 == 0:
                remaining = self.estimate_remaining_time()
                mem_usage = self.get_memory_usage()
                chunk_frames = len(self.current_chunk_frames)
                self.logger.info(f"  Rollout {ep+1}/{self.config.random_rollouts} | "
                               f"Chunk frames: {chunk_frames} | "
                               f"Chunks saved: {self.num_chunks_saved} | "
                               f"Memory: {mem_usage} | ETA: {remaining}")

            # Checkpoint（每 500 个 rollout）
            if (ep + 1) % 500 == 0:
                self.save_checkpoint()

        # 保存最后一个不完整的 chunk
        if len(self.current_chunk_frames) > 0:
            self.save_current_chunk()

        env.close()

        # 统计总帧数
        total_frames = 0
        data_dir = self.get_data_dir()
        for chunk_file in os.listdir(data_dir):
            if chunk_file.endswith('.npz'):
                with np.load(os.path.join(data_dir, chunk_file)) as data:
                    total_frames += len(data['frames'])

        self.logger.info(f"Data collection complete: {total_frames} frames in "
                        f"{self.num_chunks_saved} chunks")

    def train_vae(self):
        """Stage 2a: Train VAE（从 chunk 文件加载数据）"""
        self.training_state["stage"] = "vae_training"
        start_epoch = self.training_state["vae_epoch"]

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Stage 2a: Training VAE (Chunk-based loading)")
        self.logger.info(f"  Starting from epoch {start_epoch}")
        self.logger.info("=" * 60)

        optimizer = optim.Adam(self.vae.parameters(), lr=self.config.vae_lr)

        # 获取所有 chunk 文件
        data_dir = self.get_data_dir()
        chunk_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.logger.info(f"  Found {len(chunk_files)} data chunks")

        for epoch in range(start_epoch, self.config.vae_epochs):
            epoch_loss = 0
            n_batches = 0

            # 随机打乱 chunk 顺序
            np.random.shuffle(chunk_files)

            for chunk_file in chunk_files:
                # 加载单个 chunk
                chunk_path = os.path.join(data_dir, chunk_file)
                with np.load(chunk_path) as data:
                    frames = data['frames'].astype(np.float32) / 255.0

                # 转换为 tensor: (N, 64, 64, 3) -> (N, 3, 64, 64)
                frames = torch.FloatTensor(frames).permute(0, 3, 1, 2).to(self.device)
                n_samples = len(frames)

                # 在这个 chunk 内随机采样
                idx = np.random.permutation(n_samples)

                for i in range(0, n_samples - self.config.vae_batch_size, self.config.vae_batch_size):
                    batch_idx = idx[i:i+self.config.vae_batch_size]
                    batch = frames[batch_idx]

                    recon, mu, logvar, _ = self.vae(batch)
                    loss = vae_loss(recon, batch, mu, logvar, beta=1.0)
                    loss = loss / batch.size(0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                # 释放这个 chunk 的内存
                del frames

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history["vae_loss"].append(avg_loss)
            self.training_state["vae_epoch"] = epoch + 1

            remaining = self.estimate_remaining_time()
            mem_usage = self.get_memory_usage()
            self.logger.info(f"  Epoch {epoch+1}/{self.config.vae_epochs} | "
                           f"Loss: {avg_loss:.4f} | Memory: {mem_usage} | ETA: {remaining}")

            # Checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint()

        self.logger.info("VAE training complete")

    def train_rnn(self):
        """Stage 2b: Train MDN-RNN（从 chunk 文件加载并编码）"""
        self.training_state["stage"] = "rnn_training"
        start_epoch = self.training_state["rnn_epoch"]

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Stage 2b: Training MDN-RNN (Chunk-based)")
        self.logger.info(f"  Starting from epoch {start_epoch}")
        self.logger.info("=" * 60)

        optimizer = optim.Adam(self.rnn.parameters(), lr=self.config.rnn_lr)

        # 获取所有 chunk 文件
        data_dir = self.get_data_dir()
        chunk_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.logger.info(f"  Found {len(chunk_files)} data chunks")

        # 预先编码所有帧为 z（分 chunk 处理节省内存）
        self.logger.info("  Encoding frames with VAE (chunk by chunk)...")
        z_path = f"{self.config.save_dir}/encoded_z.npz"

        if not os.path.exists(z_path) or start_epoch == 0:
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

                # 编码
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
                self.logger.info(f"    Encoded chunk {chunk_idx+1}/{len(chunk_files)}")

            # 合并并保存
            all_z = np.concatenate(all_z, axis=0)
            all_actions = np.concatenate(all_actions, axis=0)
            all_rewards = np.concatenate(all_rewards, axis=0)
            all_dones = np.concatenate(all_dones, axis=0)

            np.savez_compressed(z_path, z=all_z, actions=all_actions,
                              rewards=all_rewards, dones=all_dones)
            self.logger.info(f"  Saved encoded data: {os.path.getsize(z_path)/(1024*1024):.1f} MB")
        else:
            self.logger.info(f"  Loading pre-encoded data from {z_path}")
            data = np.load(z_path)
            all_z = data['z']
            all_actions = data['actions']
            all_rewards = data['rewards']
            all_dones = data['dones']

        # 转换为 tensor
        z_all = torch.FloatTensor(all_z)
        actions = torch.FloatTensor(all_actions)
        rewards = torch.FloatTensor(all_rewards)
        dones = torch.FloatTensor(all_dones)

        n_samples = len(z_all) - self.config.rnn_seq_len - 1
        self.logger.info(f"  Total samples: {n_samples}, seq_len: {self.config.rnn_seq_len}")

        for epoch in range(start_epoch, self.config.rnn_epochs):
            epoch_loss = 0
            n_batches = 0

            for _ in range(100):  # 100 batches per epoch
                # Random starting points
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

                # Forward
                pi, mu, sigma, pred_r, pred_d, _ = self.rnn(z_seq, a_seq)

                # Losses
                mdn_loss = self.rnn.mdn_loss(pi, mu, sigma, z_next)
                reward_loss = F.mse_loss(pred_r, r_seq)
                done_loss = F.binary_cross_entropy(pred_d, d_seq)

                loss = mdn_loss + reward_loss + done_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history["rnn_loss"].append(avg_loss)
            self.training_state["rnn_epoch"] = epoch + 1

            remaining = self.estimate_remaining_time()
            mem_usage = self.get_memory_usage()
            self.logger.info(f"  Epoch {epoch+1}/{self.config.rnn_epochs} | "
                           f"Loss: {avg_loss:.4f} | Memory: {mem_usage} | ETA: {remaining}")

            # Checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint()

        self.logger.info("MDN-RNN training complete")

    def dream_rollout(self, controller, temperature=1.0):
        """Run a rollout in the dream (world model)"""
        # 从预编码的 z 数据中采样初始状态
        z_path = f"{self.config.save_dir}/encoded_z.npz"
        if not hasattr(self, '_cached_z_for_dream') or self._cached_z_for_dream is None:
            if os.path.exists(z_path):
                data = np.load(z_path)
                self._cached_z_for_dream = data['z']
                self.logger.info(f"  Loaded {len(self._cached_z_for_dream)} z vectors for dream rollouts")
            else:
                raise RuntimeError("No encoded z data found. Run train_rnn first.")

        idx = np.random.randint(0, len(self._cached_z_for_dream))
        z = torch.FloatTensor(self._cached_z_for_dream[idx]).to(self.device)

        hidden = None
        total_reward = 0

        for _ in range(self.config.dream_rollout_length):
            # Get LSTM hidden for controller input
            if hidden is None:
                h = torch.zeros(self.config.hidden_size).to(self.device)
            else:
                h = hidden[0].squeeze(0).squeeze(0)

            # Controller input: [z, h]
            ctrl_input = torch.cat([z, h]).detach().cpu().numpy()
            action = controller.get_action(ctrl_input)
            action_t = torch.FloatTensor(action).to(self.device)

            # RNN step
            z_in = z.unsqueeze(0).unsqueeze(0)
            a_in = action_t.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                pi, mu, sigma, reward, done, hidden = self.rnn(z_in, a_in, hidden)

            # Sample next z
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
        """Stage 3: Train controller with CMA-ES in dreams"""
        self.training_state["stage"] = "controller_training"
        start_gen = self.training_state["cmaes_generation"]

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Stage 3: Training Controller (CMA-ES in dreams)")
        self.logger.info(f"  Starting from generation {start_gen}")
        self.logger.info("=" * 60)

        # Controller input: z + hidden
        input_dim = self.config.latent_size + self.config.hidden_size
        controller = Controller(input_dim, 3)  # 3 actions

        # 恢复或创建 CMA-ES
        if self.cmaes_state is not None and start_gen > 0:
            cmaes = SimpleCMAES(controller.num_params, self.config.population_size)
            cmaes.mean = self.cmaes_state["mean"]
            cmaes.sigma = self.cmaes_state["sigma"]
            cmaes.C = self.cmaes_state["C"]
            self.logger.info("  Restored CMA-ES state from checkpoint")
        else:
            cmaes = SimpleCMAES(controller.num_params, self.config.population_size)

        best_controller = None
        if self.best_controller_params is not None:
            best_controller = Controller(input_dim, 3)
            best_controller.set_params(self.best_controller_params)

        for gen in range(start_gen, self.config.generations):
            population = cmaes.ask()
            fitness = []

            for params in population:
                controller.set_params(params)
                # Average over multiple dreams (论文用 16 rollouts)
                rewards = [self.dream_rollout(controller, self.config.temperature)
                          for _ in range(self.config.n_rollouts_per_eval)]
                fitness.append(np.mean(rewards))

            fitness = np.array(fitness)
            cmaes.tell(population, fitness)

            gen_best = fitness.max()
            gen_mean = fitness.mean()

            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
                best_controller = Controller(input_dim, 3)
                best_controller.set_params(population[fitness.argmax()])
                self.best_controller_params = population[fitness.argmax()].copy()

            self.history["dream_fitness"].append(gen_best)
            self.training_state["cmaes_generation"] = gen + 1

            # 保存 CMA-ES 状态
            self.cmaes_state = {
                "mean": cmaes.mean.copy(),
                "sigma": cmaes.sigma,
                "C": cmaes.C.copy()
            }

            if (gen + 1) % 10 == 0:
                remaining = self.estimate_remaining_time()
                self.logger.info(f"  Generation {gen+1}/{self.config.generations} | "
                               f"Best: {gen_best:.1f} | Mean: {gen_mean:.1f} | "
                               f"All-time best: {self.best_fitness:.1f} | ETA: {remaining}")

            # Checkpoint
            if (gen + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

        self.logger.info(f"Controller training complete | Best dream fitness: {self.best_fitness:.1f}")
        return best_controller

    def evaluate_real(self, controller, n_episodes=10):
        """Evaluate in real environment"""
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

            # Encode initial frame
            frame = self.preprocess_frame(obs)
            frame_t = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                mu, _ = self.vae.encode(frame_t)
                z = mu.squeeze(0)

            for step in range(self.config.max_steps_per_episode):
                # Controller
                if hidden is None:
                    h = torch.zeros(self.config.hidden_size).to(self.device)
                else:
                    h = hidden[0].squeeze(0).squeeze(0)

                ctrl_input = torch.cat([z, h]).detach().cpu().numpy()
                action = controller.get_action(ctrl_input)

                # Step
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

                if done:
                    break

                # Encode new frame and update RNN
                frame = self.preprocess_frame(obs)
                frame_t = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    mu, _ = self.vae.encode(frame_t)
                    z = mu.squeeze(0)

                    # Update hidden state
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
        """Full training pipeline with checkpoint support"""
        # 尝试从 checkpoint 恢复
        resumed = False
        if self.config.resume_from_checkpoint:
            resumed = self.load_checkpoint()

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("World Models: CarRacing Reproduction")
        self.logger.info("=" * 60)
        self.logger.info(f"  Mode: {MODE}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Latent size: {self.config.latent_size}")
        self.logger.info(f"  Hidden size: {self.config.hidden_size}")
        self.logger.info(f"  Rollouts: {self.config.random_rollouts}")
        self.logger.info(f"  Resumed: {resumed}")
        self.logger.info("=" * 60)

        # 记录开始时间
        if self.training_state["start_time"] is None:
            self.training_state["start_time"] = time.time()

        stage = self.training_state["stage"]

        # Stage 1: 数据收集
        if stage in ["init", "data_collection"]:
            if self.training_state["data_collection_progress"] < self.config.random_rollouts:
                self.collect_data()

        # Stage 2a: VAE 训练
        if stage in ["init", "data_collection", "vae_training"]:
            if self.training_state["vae_epoch"] < self.config.vae_epochs:
                self.train_vae()

        # Stage 2b: RNN 训练
        if stage in ["init", "data_collection", "vae_training", "rnn_training"]:
            if self.training_state["rnn_epoch"] < self.config.rnn_epochs:
                self.train_rnn()

        # Stage 3: Controller 训练
        if stage in ["init", "data_collection", "vae_training", "rnn_training", "controller_training"]:
            if self.training_state["cmaes_generation"] < self.config.generations:
                controller = self.train_controller()
            else:
                # 从保存的参数恢复 controller
                input_dim = self.config.latent_size + self.config.hidden_size
                controller = Controller(input_dim, 3)
                if self.best_controller_params is not None:
                    controller.set_params(self.best_controller_params)

        # 标记完成
        self.training_state["stage"] = "done"

        # Evaluate
        self.evaluate_real(controller, n_episodes=20)

        # Save final results
        self.save(controller)

        # 计算总耗时
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
        """Save models and results"""
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Models
        torch.save({
            'vae': self.vae.state_dict(),
            'rnn': self.rnn.state_dict()
        }, f"{self.config.save_dir}/models.pt")

        np.savez(f"{self.config.save_dir}/controller.npz",
                W=controller.W, b=controller.b)

        # History
        with open(f"{self.config.save_dir}/history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # Plot
        self.plot_results()

        # 保存最终 checkpoint
        self.save_checkpoint()

        self.logger.info(f"Results saved to {self.config.save_dir}")

    def plot_results(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        if self.history["vae_loss"]:
            axes[0, 0].plot(self.history["vae_loss"])
            axes[0, 0].set_title("VAE Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_yscale("log")
            axes[0, 0].grid(True, alpha=0.3)

        if self.history["rnn_loss"]:
            axes[0, 1].plot(self.history["rnn_loss"])
            axes[0, 1].set_title("MDN-RNN Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_yscale("log")
            axes[0, 1].grid(True, alpha=0.3)

        if self.history["dream_fitness"]:
            axes[1, 0].plot(self.history["dream_fitness"])
            axes[1, 0].set_title("Dream Fitness (CMA-ES)")
            axes[1, 0].set_xlabel("Generation")
            axes[1, 0].grid(True, alpha=0.3)

        # Summary
        axes[1, 1].axis('off')

        # 统计总帧数
        total_frames = 0
        data_dir = self.get_data_dir()
        if os.path.exists(data_dir):
            for chunk_file in os.listdir(data_dir):
                if chunk_file.endswith('.npz'):
                    with np.load(os.path.join(data_dir, chunk_file)) as data:
                        total_frames += len(data['frames'])

        dream_fitness = self.history['dream_fitness'][-1] if self.history['dream_fitness'] else 0
        real_reward = self.history['real_reward'][-1] if self.history['real_reward'] else 'N/A'

        summary = f"""
        World Models: CarRacing
        =======================

        Rollouts collected: {self.config.random_rollouts}
        Total frames: {total_frames}
        Data chunks: {self.num_chunks_saved}

        VAE latent dim: {self.config.latent_size}
        LSTM hidden: {self.config.hidden_size}
        MDN components: {self.config.n_gaussians}

        Final dream fitness: {dream_fitness:.1f}
        Real env reward: {real_reward}
        """
        axes[1, 1].text(0.1, 0.5, summary, fontsize=12, family='monospace',
                       verticalalignment='center')

        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/training_curves.png", dpi=150)
        plt.close()


# ========== Main ==========
def main():
    config = Config()
    agent = CarRacingWorldModel(config)
    agent.train()


if __name__ == "__main__":
    main()
