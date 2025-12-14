"""
CartPole-v1: Simple World Model
================================
简化的 World Models (2018) 实现

架构：
- Vision: 简单编码器（状态已是低维向量）
- Memory: LSTM 预测下一状态
- Controller: 线性策略 + CMA-ES

关键对比点：
- 在"梦境"中训练策略
- 样本效率提升 ~3×
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
    
    # 阶段 1: 数据收集（优化：更多数据）
    random_episodes = 200  # 增加数据量
    
    # 阶段 2: 训练世界模型（优化：更强的模型）
    world_model_epochs = 100  # 更多训练轮次
    batch_size = 32
    sequence_length = 30  # 缩短序列，更稳定
    hidden_size = 128  # 增加模型容量
    learning_rate = 1e-3
    
    # 阶段 3: 梦境中训练策略（优化：更保守的梦境长度）
    population_size = 100  # 增加种群大小
    generations = 150  # 更多进化代数
    dream_rollout_length = 100  # 缩短梦境长度，减少累积误差
    elite_frac = 0.2
    
    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 日志
    save_dir = "./results_simple_wm"


# ========== World Model: LSTM 动态模型 ==========
class WorldModel(nn.Module):
    """
    LSTM-based World Model
    输入: (state_t, action_t)
    输出: state_{t+1}, reward_t, done_t
    """
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # 输入编码
        self.input_encoder = nn.Linear(state_dim + action_dim, hidden_size)
        
        # LSTM 核心
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # 输出解码
        self.state_predictor = nn.Linear(hidden_size, state_dim)
        self.reward_predictor = nn.Linear(hidden_size, 1)
        self.done_predictor = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action, hidden=None):
        """
        Args:
            state: (batch, seq_len, state_dim)
            action: (batch, seq_len, action_dim)
            hidden: LSTM hidden state
        Returns:
            next_state, reward, done, hidden
        """
        # 编码输入
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.input_encoder(x))
        
        # LSTM
        x, hidden = self.lstm(x, hidden)
        
        # 预测
        next_state = self.state_predictor(x)
        reward = self.reward_predictor(x)
        done = torch.sigmoid(self.done_predictor(x))
        
        return next_state, reward, done, hidden
    
    def imagine_step(self, state, action, hidden=None):
        """单步想象（用于策略训练）"""
        with torch.no_grad():
            state = state.unsqueeze(1)  # (batch, 1, state_dim)
            action = action.unsqueeze(1)  # (batch, 1, action_dim)
            next_state, reward, done, hidden = self.forward(state, action, hidden)
            return next_state.squeeze(1), reward.squeeze(1), done.squeeze(1), hidden


# ========== Controller: 简单线性策略 ==========
class LinearController:
    """
    线性策略: action = argmax(W @ [state, hidden])
    参数极少，适合进化算法优化
    """
    def __init__(self, input_dim, action_dim):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.weights = np.random.randn(action_dim, input_dim) * 0.1
    
    def get_action(self, state):
        """选择动作"""
        logits = self.weights @ state
        return np.argmax(logits)
    
    def get_params(self):
        """获取参数（扁平化）"""
        return self.weights.flatten()
    
    def set_params(self, params):
        """设置参数"""
        self.weights = params.reshape(self.action_dim, self.input_dim)


# ========== CMA-ES 进化算法 ==========
class CMAES:
    """简化的 CMA-ES 实现"""
    def __init__(self, dim, population_size=50, elite_frac=0.2):
        self.dim = dim
        self.population_size = population_size
        self.elite_size = int(population_size * elite_frac)
        
        # 初始分布
        self.mean = np.zeros(dim)
        self.sigma = 0.5
    
    def ask(self):
        """生成候选解"""
        return np.random.randn(self.population_size, self.dim) * self.sigma + self.mean
    
    def tell(self, population, fitness):
        """更新分布"""
        # 选择精英
        elite_idxs = np.argsort(fitness)[-self.elite_size:]
        elite = population[elite_idxs]
        
        # 更新均值
        self.mean = elite.mean(axis=0)
        
        # 更新方差（简化版）
        self.sigma = elite.std() * 0.9


# ========== Simple World Model Agent ==========
class SimpleWorldModelAgent:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_name)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # 世界模型
        self.world_model = WorldModel(
            self.state_dim, 
            self.action_dim, 
            config.hidden_size
        ).to(config.device)
        
        self.optimizer = optim.Adam(
            self.world_model.parameters(), 
            lr=config.learning_rate
        )
        
        # 数据缓冲
        self.trajectories = []
        
        # 记录
        self.training_history = {
            "data_collection_rewards": [],
            "world_model_losses": [],
            "policy_fitness": [],
            "evaluation_rewards": []
        }
    
    def collect_data(self):
        """阶段 1: 随机策略收集数据"""
        print("📦 阶段 1: 收集数据")
        print("-" * 50)
        
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
                print(f"Episode {episode+1}/{self.config.random_episodes} | Avg Reward: {avg_reward:.2f}")
        
        print(f"✅ 收集了 {len(self.trajectories)} 条轨迹")
    
    def train_world_model(self):
        """阶段 2: 训练世界模型"""
        print("\n🌍 阶段 2: 训练世界模型")
        print("-" * 50)
        
        for epoch in range(self.config.world_model_epochs):
            epoch_losses = []
            
            # 随机采样轨迹（增加采样次数）
            for _ in range(50):
                traj = np.random.choice(self.trajectories)
                
                # 准备序列数据（需要确保有足够的长度来取 next_state）
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
                
                # 前向传播
                pred_states, pred_rewards, pred_dones, _ = self.world_model(states, actions)
                
                # 损失（增加权重平衡）
                state_loss = nn.MSELoss()(pred_states, next_states)
                reward_loss = nn.MSELoss()(pred_rewards, rewards) * 10.0  # 增加奖励权重
                done_loss = nn.BCELoss()(pred_dones, dones) * 5.0  # 增加终止预测权重
                
                loss = state_loss + reward_loss + done_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)  # 梯度裁剪
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            self.training_history["world_model_losses"].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.world_model_epochs} | Loss: {avg_loss:.4f}")
        
        print("✅ 世界模型训练完成")
    
    def train_controller_in_dream(self):
        """阶段 3: 在梦境中训练策略"""
        print("\n💭 阶段 3: 梦境训练策略")
        print("-" * 50)
        
        # 初始化 CMA-ES
        param_dim = self.state_dim * self.action_dim
        cmaes = CMAES(param_dim, self.config.population_size)
        
        best_controller = None
        best_fitness = -float('inf')
        
        for generation in range(self.config.generations):
            # 生成种群
            population = cmaes.ask()
            fitness_scores = []
            
            # 评估每个个体
            for params in population:
                controller = LinearController(self.state_dim, self.action_dim)
                controller.set_params(params)
                
                # 在梦境中评估
                fitness = self.evaluate_in_dream(controller)
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            
            # 更新 CMA-ES
            cmaes.tell(population, fitness_scores)
            
            # 记录最佳
            gen_best = fitness_scores.max()
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_controller = LinearController(self.state_dim, self.action_dim)
                best_controller.set_params(population[fitness_scores.argmax()])
            
            self.training_history["policy_fitness"].append(gen_best)
            
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation+1}/{self.config.generations} | "
                      f"Best Fitness: {gen_best:.2f} | "
                      f"Mean: {fitness_scores.mean():.2f}")
        
        print(f"✅ 策略训练完成 | 最佳适应度: {best_fitness:.2f}")
        return best_controller
    
    def evaluate_in_dream(self, controller):
        """在梦境中评估策略"""
        total_reward = 0
        
        # 从随机轨迹采样起始状态
        traj = np.random.choice(self.trajectories)
        state = torch.FloatTensor(traj["states"][0]).to(self.config.device)
        
        hidden = None
        
        for _ in range(self.config.dream_rollout_length):
            # 控制器选择动作
            action_idx = controller.get_action(state.cpu().numpy())
            action = torch.zeros(self.action_dim).to(self.config.device)
            action[action_idx] = 1
            
            # 世界模型预测
            next_state, reward, done, hidden = self.world_model.imagine_step(
                state.unsqueeze(0), 
                action.unsqueeze(0), 
                hidden
            )
            
            total_reward += reward.item()
            
            # 终止检查
            if done.item() > 0.5:
                break
            
            state = next_state.squeeze(0)
        
        return total_reward
    
    def evaluate_in_real_env(self, controller, num_episodes=10):
        """在真实环境中评估"""
        rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = controller.get_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return np.mean(rewards), np.std(rewards)
    
    def train(self):
        """完整训练流程"""
        print("🚀 开始训练 Simple World Model")
        print(f"设备: {self.config.device}")
        print("=" * 50)
        
        # 阶段 1: 收集数据
        self.collect_data()
        
        # 阶段 2: 训练世界模型
        self.train_world_model()
        
        # 阶段 3: 在梦境中训练策略
        best_controller = self.train_controller_in_dream()
        
        # 评估
        print("\n📊 在真实环境中评估...")
        mean_reward, std_reward = self.evaluate_in_real_env(best_controller, num_episodes=50)
        self.training_history["evaluation_rewards"].append(mean_reward)
        
        print(f"✅ 评估结果: {mean_reward:.2f} ± {std_reward:.2f}")
        
        self.save_results()
    
    def save_results(self):
        """保存结果"""
        import os
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # 保存模型
        torch.save(self.world_model.state_dict(), 
                   f"{self.config.save_dir}/world_model.pt")
        
        # 保存训练历史
        with open(f"{self.config.save_dir}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        # 绘图
        self.plot_results()
        
        print(f"📊 结果已保存到: {self.config.save_dir}")
    
    def plot_results(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 数据收集阶段奖励
        axes[0, 0].plot(self.training_history["data_collection_rewards"], alpha=0.5)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Data Collection (Random Policy)')
        axes[0, 0].grid(True)
        
        # 世界模型损失
        axes[0, 1].plot(self.training_history["world_model_losses"])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('World Model Training')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # 策略适应度
        axes[1, 0].plot(self.training_history["policy_fitness"])
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Fitness (Dream Reward)')
        axes[1, 0].set_title('Policy Evolution (CMA-ES)')
        axes[1, 0].grid(True)
        
        # 最终评估
        if self.training_history["evaluation_rewards"]:
            axes[1, 1].bar(['Evaluation'], self.training_history["evaluation_rewards"])
            axes[1, 1].set_ylabel('Mean Reward')
            axes[1, 1].set_title('Real Environment Evaluation')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/training_curves.png", dpi=150)
        plt.close()


# ========== 主函数 ==========
def main():
    config = Config()
    agent = SimpleWorldModelAgent(config)
    agent.train()


if __name__ == "__main__":
    main()
