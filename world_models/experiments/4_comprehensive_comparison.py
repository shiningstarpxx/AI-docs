"""
CartPole-v1 综合对比实验
========================

对比三种方法的样本效率：
1. DQN (Baseline)
2. Simple World Model
3. Mini Dreamer

输出：
- 样本效率对比图
- 训练时间对比
- 最终性能对比表
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import gymnasium as gym
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import json
import time
import os
from datetime import datetime

# ========== 设备配置 ==========
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ========== 通用配置 ==========
class BaseConfig:
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2
    gamma = 0.99
    max_steps = 500
    log_interval = 10
    device = DEVICE


# ========== 1. DQN Agent ==========
class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_name)

        self.q_net = DQNNet(config.state_dim, config.action_dim).to(config.device)
        self.target_net = DQNNet(config.state_dim, config.action_dim).to(config.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=3e-4)
        self.buffer = deque(maxlen=10000)

        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.998

        self.total_env_steps = 0
        self.history = {"env_steps": [], "rewards": []}

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            return self.q_net(state_t).argmax(1).item()

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        batch = random.sample(list(self.buffer), batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.FloatTensor(np.array(s)).to(self.config.device)
        a = torch.LongTensor(a).to(self.config.device)
        r = torch.FloatTensor(r).to(self.config.device)
        s2 = torch.FloatTensor(np.array(s2)).to(self.config.device)
        d = torch.FloatTensor(d).to(self.config.device)

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q2 = self.target_net(s2).max(1)[0]
            target = r + (1 - d) * self.config.gamma * q2

        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes=400):
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            ep_reward = 0

            for step in range(self.config.max_steps):
                action = self.select_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc

                self.buffer.append((state, action, reward, next_state, float(done)))
                self.update()

                ep_reward += reward
                self.total_env_steps += 1
                state = next_state

                if done:
                    break

            if ep % 10 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            self.history["env_steps"].append(self.total_env_steps)
            self.history["rewards"].append(ep_reward)

            if (ep + 1) % self.config.log_interval == 0:
                avg = np.mean(self.history["rewards"][-10:])
                print(f"DQN | Ep {ep+1} | Steps {self.total_env_steps} | Avg Reward: {avg:.1f}")

        return self.history


# ========== 2. Simple World Model Agent ==========
class SimpleWorldModelAgent:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_name)

        hidden = 64
        latent = 16

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent)
        ).to(config.device)

        # Dynamics model (LSTM-like)
        self.dynamics = nn.Sequential(
            nn.Linear(latent + config.action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent)
        ).to(config.device)

        # Reward predictor
        self.reward_pred = nn.Linear(latent, 1).to(config.device)

        # Policy
        self.policy = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.action_dim)
        ).to(config.device)

        # Decoder (for reconstruction loss)
        self.decoder = nn.Linear(latent, config.state_dim).to(config.device)

        # Optimizers
        wm_params = list(self.encoder.parameters()) + list(self.dynamics.parameters()) + \
                    list(self.reward_pred.parameters()) + list(self.decoder.parameters())
        self.wm_optimizer = optim.Adam(wm_params, lr=1e-3)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.buffer = deque(maxlen=5000)
        self.total_env_steps = 0
        self.history = {"env_steps": [], "rewards": []}

    def collect_episode(self, random_policy=False):
        trajectory = []
        state, _ = self.env.reset()
        ep_reward = 0

        for _ in range(self.config.max_steps):
            if random_policy:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    z = self.encoder(torch.FloatTensor(state).to(self.config.device))
                    logits = self.policy(z)
                    action = distributions.Categorical(logits=logits).sample().item()

            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            action_oh = np.zeros(self.config.action_dim)
            action_oh[action] = 1
            trajectory.append((state, action_oh, reward, next_state, done))

            ep_reward += reward
            self.total_env_steps += 1
            state = next_state

            if done:
                break

        self.buffer.append(trajectory)
        return ep_reward

    def train_world_model(self, updates=5):
        if len(self.buffer) < 5:
            return

        for _ in range(updates):
            traj = random.choice(list(self.buffer))
            if len(traj) < 2:
                continue

            states = torch.FloatTensor([t[0] for t in traj]).to(self.config.device)
            actions = torch.FloatTensor([t[1] for t in traj]).to(self.config.device)
            rewards = torch.FloatTensor([t[2] for t in traj]).unsqueeze(1).to(self.config.device)
            next_states = torch.FloatTensor([t[3] for t in traj]).to(self.config.device)

            z = self.encoder(states)
            z_next_pred = self.dynamics(torch.cat([z, actions], dim=-1))
            reward_pred = self.reward_pred(z)
            state_recon = self.decoder(z)

            z_next_true = self.encoder(next_states).detach()

            loss = nn.MSELoss()(z_next_pred, z_next_true) + \
                   nn.MSELoss()(reward_pred, rewards) + \
                   0.1 * nn.MSELoss()(state_recon, states)

            self.wm_optimizer.zero_grad()
            loss.backward()
            self.wm_optimizer.step()

    def train_policy_in_imagination(self, imagination_steps=50):
        if len(self.buffer) < 5:
            return

        # Sample starting state
        traj = random.choice(list(self.buffer))
        start_state = torch.FloatTensor(traj[0][0]).to(self.config.device)

        z = self.encoder(start_state)
        total_reward = 0

        for _ in range(imagination_steps):
            logits = self.policy(z)
            dist = distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action_oh = torch.zeros(self.config.action_dim).to(self.config.device)
            action_oh[action] = 1

            z = self.dynamics(torch.cat([z, action_oh], dim=-1))
            reward = self.reward_pred(z)

            total_reward = total_reward + reward

        # Policy gradient (maximize reward)
        policy_loss = -total_reward

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def train(self, num_episodes=200):
        # Seed episodes
        for _ in range(5):
            self.collect_episode(random_policy=True)

        for ep in range(num_episodes):
            ep_reward = self.collect_episode(random_policy=False)

            self.train_world_model(updates=10)

            for _ in range(5):
                self.train_policy_in_imagination()

            self.history["env_steps"].append(self.total_env_steps)
            self.history["rewards"].append(ep_reward)

            if (ep + 1) % self.config.log_interval == 0:
                avg = np.mean(self.history["rewards"][-10:])
                print(f"SimpleWM | Ep {ep+1} | Steps {self.total_env_steps} | Avg Reward: {avg:.1f}")

        return self.history


# ========== 3. Mini Dreamer Agent ==========
class MiniDreamerAgent:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_name)

        hidden = 64
        stoch = 16

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        ).to(config.device)

        # RNN (deterministic path)
        self.rnn = nn.GRUCell(stoch + config.action_dim, hidden).to(config.device)

        # Prior/Posterior
        self.prior = nn.Linear(hidden, stoch * 2).to(config.device)
        self.posterior = nn.Linear(hidden * 2, stoch * 2).to(config.device)

        # Decoder
        self.decoder = nn.Linear(hidden + stoch, config.state_dim).to(config.device)
        self.reward_pred = nn.Linear(hidden + stoch, 1).to(config.device)

        # Actor-Critic
        self.actor = nn.Sequential(
            nn.Linear(hidden + stoch, 64),
            nn.ReLU(),
            nn.Linear(64, config.action_dim)
        ).to(config.device)

        self.critic = nn.Sequential(
            nn.Linear(hidden + stoch, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(config.device)

        # Optimizers
        wm_params = list(self.encoder.parameters()) + list(self.rnn.parameters()) + \
                    list(self.prior.parameters()) + list(self.posterior.parameters()) + \
                    list(self.decoder.parameters()) + list(self.reward_pred.parameters())
        self.wm_opt = optim.Adam(wm_params, lr=1e-3)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.buffer = deque(maxlen=5000)
        self.total_env_steps = 0
        self.history = {"env_steps": [], "rewards": []}

        self.hidden = hidden
        self.stoch = stoch

    def get_dist(self, params):
        mean, logstd = params.chunk(2, dim=-1)
        std = torch.exp(logstd.clamp(-5, 2)) + 0.1
        return mean, std

    def collect_episode(self, random_policy=False):
        trajectory = []
        state, _ = self.env.reset()
        ep_reward = 0

        h = torch.zeros(1, self.hidden).to(self.config.device)
        z = torch.zeros(1, self.stoch).to(self.config.device)

        for _ in range(self.config.max_steps):
            if random_policy:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    feat = torch.cat([h, z], dim=-1)
                    logits = self.actor(feat)
                    action = distributions.Categorical(logits=logits).sample().item()

            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            action_oh = np.zeros(self.config.action_dim)
            action_oh[action] = 1
            trajectory.append((state, action_oh, reward, next_state, done))

            # Update RSSM state
            if not random_policy:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
                    action_t = torch.FloatTensor(action_oh).unsqueeze(0).to(self.config.device)

                    obs_embed = self.encoder(state_t)
                    inp = torch.cat([z, action_t], dim=-1)
                    h = self.rnn(inp, h)

                    post_params = self.posterior(torch.cat([h, obs_embed], dim=-1))
                    mean, _ = self.get_dist(post_params)
                    z = mean

            ep_reward += reward
            self.total_env_steps += 1
            state = next_state

            if done:
                break

        self.buffer.append(trajectory)
        return ep_reward

    def train_world_model(self, updates=5):
        if len(self.buffer) < 5:
            return

        for _ in range(updates):
            traj = random.choice(list(self.buffer))
            if len(traj) < 2:
                continue

            h = torch.zeros(1, self.hidden).to(self.config.device)
            total_loss = 0

            for t in range(len(traj) - 1):
                state = torch.FloatTensor(traj[t][0]).unsqueeze(0).to(self.config.device)
                action = torch.FloatTensor(traj[t][1]).unsqueeze(0).to(self.config.device)
                reward = torch.FloatTensor([traj[t][2]]).unsqueeze(0).to(self.config.device)
                next_state = torch.FloatTensor(traj[t][3]).unsqueeze(0).to(self.config.device)

                obs_embed = self.encoder(state)

                # Prior
                prior_params = self.prior(h)
                prior_mean, prior_std = self.get_dist(prior_params)

                # Posterior
                post_params = self.posterior(torch.cat([h, obs_embed], dim=-1))
                post_mean, post_std = self.get_dist(post_params)

                # Sample
                z = post_mean + post_std * torch.randn_like(post_std)

                # Predictions
                feat = torch.cat([h, z], dim=-1)
                state_pred = self.decoder(feat)
                reward_pred = self.reward_pred(feat)

                # Losses
                recon_loss = nn.MSELoss()(state_pred, next_state)
                reward_loss = nn.MSELoss()(reward_pred, reward)

                kl_loss = torch.distributions.kl_divergence(
                    distributions.Normal(post_mean, post_std),
                    distributions.Normal(prior_mean, prior_std)
                ).sum(-1).mean()

                total_loss = total_loss + recon_loss + reward_loss + 0.1 * kl_loss

                # Update h
                h = self.rnn(torch.cat([z, action], dim=-1), h)

            self.wm_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 10)
            self.wm_opt.step()

    def train_actor_critic(self, imagination_horizon=15):
        if len(self.buffer) < 5:
            return

        traj = random.choice(list(self.buffer))
        start_idx = random.randint(0, len(traj) - 1)

        state = torch.FloatTensor(traj[start_idx][0]).unsqueeze(0).to(self.config.device)
        obs_embed = self.encoder(state)

        h = torch.zeros(1, self.hidden).to(self.config.device)
        prior_params = self.prior(h)
        z, _ = self.get_dist(prior_params)

        # Imagination
        imagined = []
        for _ in range(imagination_horizon):
            feat = torch.cat([h, z], dim=-1)

            logits = self.actor(feat)
            dist = distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action_oh = torch.zeros(self.config.action_dim).to(self.config.device)
            action_oh[action] = 1

            value = self.critic(feat)
            reward = self.reward_pred(feat)

            # Next state
            h = self.rnn(torch.cat([z, action_oh.unsqueeze(0)], dim=-1), h)
            prior_params = self.prior(h)
            z, _ = self.get_dist(prior_params)

            imagined.append({
                "log_prob": log_prob,
                "value": value.squeeze(),
                "reward": reward.squeeze()
            })

        # Compute returns and advantages
        returns = []
        advantages = []
        next_value = self.critic(torch.cat([h, z], dim=-1)).squeeze().detach()

        for t in reversed(range(len(imagined))):
            r = imagined[t]["reward"]
            v = imagined[t]["value"]

            td_error = r + self.config.gamma * next_value - v

            if t == len(imagined) - 1:
                adv = td_error
            else:
                adv = td_error + self.config.gamma * 0.95 * advantages[0]

            advantages.insert(0, adv)
            returns.insert(0, adv + v)
            next_value = v

        # Actor loss
        actor_loss = 0
        for t, step in enumerate(imagined):
            actor_loss = actor_loss - step["log_prob"] * advantages[t].detach()

        self.actor_opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_opt.step()

        # Critic loss
        critic_loss = 0
        for t, step in enumerate(imagined):
            critic_loss = critic_loss + (step["value"] - returns[t].detach()) ** 2

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

    def train(self, num_episodes=150):
        # Seed episodes
        for _ in range(5):
            self.collect_episode(random_policy=True)

        for ep in range(num_episodes):
            ep_reward = self.collect_episode(random_policy=False)

            for _ in range(10):
                self.train_world_model()

            for _ in range(5):
                self.train_actor_critic()

            self.history["env_steps"].append(self.total_env_steps)
            self.history["rewards"].append(ep_reward)

            if (ep + 1) % self.config.log_interval == 0:
                avg = np.mean(self.history["rewards"][-10:])
                print(f"MiniDreamer | Ep {ep+1} | Steps {self.total_env_steps} | Avg Reward: {avg:.1f}")

        return self.history


# ========== 对比实验 ==========
def run_comparison():
    print("=" * 60)
    print("CartPole-v1 样本效率对比实验")
    print("=" * 60)

    config = BaseConfig()
    results = {}

    # 1. DQN
    print("\n" + "=" * 40)
    print("Training DQN (Baseline)")
    print("=" * 40)
    start = time.time()
    dqn = DQNAgent(config)
    results["DQN"] = dqn.train(num_episodes=400)
    results["DQN"]["time"] = time.time() - start
    print(f"DQN completed in {results['DQN']['time']:.1f}s")

    # 2. Simple World Model
    print("\n" + "=" * 40)
    print("Training Simple World Model")
    print("=" * 40)
    start = time.time()
    swm = SimpleWorldModelAgent(config)
    results["SimpleWM"] = swm.train(num_episodes=200)
    results["SimpleWM"]["time"] = time.time() - start
    print(f"SimpleWM completed in {results['SimpleWM']['time']:.1f}s")

    # 3. Mini Dreamer
    print("\n" + "=" * 40)
    print("Training Mini Dreamer")
    print("=" * 40)
    start = time.time()
    dreamer = MiniDreamerAgent(config)
    results["MiniDreamer"] = dreamer.train(num_episodes=150)
    results["MiniDreamer"]["time"] = time.time() - start
    print(f"MiniDreamer completed in {results['MiniDreamer']['time']:.1f}s")

    return results


def plot_comparison(results, save_dir="./results_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"DQN": "blue", "SimpleWM": "green", "MiniDreamer": "red"}

    # Plot 1: Reward vs Environment Steps (Sample Efficiency)
    ax1 = axes[0]
    for name, data in results.items():
        steps = data["env_steps"]
        rewards = data["rewards"]

        # Smooth
        window = 20
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            ax1.plot(smooth_steps, smoothed, label=name, color=colors[name], linewidth=2)
        else:
            ax1.plot(steps, rewards, label=name, color=colors[name], linewidth=2)

    ax1.axhline(y=475, color='black', linestyle='--', alpha=0.5, label='Solved (475)')
    ax1.set_xlabel("Environment Steps", fontsize=12)
    ax1.set_ylabel("Average Reward", fontsize=12)
    ax1.set_title("Sample Efficiency Comparison", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Summary Bar Chart
    ax2 = axes[1]
    names = list(results.keys())

    # Calculate metrics
    final_rewards = [np.mean(results[n]["rewards"][-20:]) for n in names]
    total_steps = [results[n]["env_steps"][-1] for n in names]
    times = [results[n]["time"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    bars1 = ax2.bar(x - width, [r/500*100 for r in final_rewards], width, label='Final Reward (%)', color='steelblue')
    bars2 = ax2.bar(x, [s/max(total_steps)*100 for s in total_steps], width, label='Env Steps (rel.%)', color='seagreen')
    bars3 = ax2.bar(x + width, [t/max(times)*100 for t in times], width, label='Time (rel.%)', color='coral')

    ax2.set_ylabel("Percentage", fontsize=12)
    ax2.set_title("Performance Summary", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison.png", dpi=150)
    plt.close()

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<15} {'Final Reward':<15} {'Env Steps':<15} {'Time (s)':<15}")
    print("-" * 60)
    for name in names:
        fr = np.mean(results[name]["rewards"][-20:])
        ts = results[name]["env_steps"][-1]
        tm = results[name]["time"]
        print(f"{name:<15} {fr:<15.1f} {ts:<15} {tm:<15.1f}")
    print("=" * 60)

    # Save JSON
    save_data = {}
    for name, data in results.items():
        save_data[name] = {
            "env_steps": data["env_steps"],
            "rewards": data["rewards"],
            "time": data["time"],
            "final_reward_mean": float(np.mean(data["rewards"][-20:])),
            "total_env_steps": data["env_steps"][-1]
        }

    with open(f"{save_dir}/results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {save_dir}/")


# ========== Main ==========
if __name__ == "__main__":
    results = run_comparison()
    plot_comparison(results)
