"""
Dyna-Q Algorithm Implementation

Dyna-Q 是 Model-Based RL 的经典算法，由 Sutton (1990) 提出。
核心思想：结合 direct RL (从真实经验学习) 和 planning (从模型生成的经验学习)

架构：
1. Q-learning: 从真实交互更新 Q 值
2. Model Learning: 学习环境模型 (s, a) -> (s', r)
3. Planning: 用模型生成虚拟经验，额外更新 Q 值

对比实验：
- Q-learning (无 planning)
- Dyna-Q (n=5, 10, 50 planning steps)

环境：GridWorld (简单离散环境)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from typing import Tuple, Dict, List, Optional


# =============================================================================
# GridWorld 环境
# =============================================================================

class GridWorld:
    """
    简单的 GridWorld 环境

    地图说明：
    - 0: 空地 (可通行)
    - 1: 墙壁 (不可通行)
    - 2: 起点
    - 3: 终点 (奖励 +1)
    - 4: 陷阱 (奖励 -1, 可选)
    """

    def __init__(self, grid_size: int = 6, seed: int = 42):
        self.grid_size = grid_size
        self.rng = np.random.RandomState(seed)

        # 动作空间: 上、下、左、右
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        self.n_actions = 4

        # 创建地图
        self._create_map()

        # 状态
        self.state = None
        self.steps = 0
        self.max_steps = 100

    def _create_map(self):
        """创建地图"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # 起点 (左上角)
        self.start = (0, 0)

        # 终点 (右下角)
        self.goal = (self.grid_size - 1, self.grid_size - 1)

        # 添加一些墙壁 (简单迷宫)
        if self.grid_size >= 6:
            # 中间横墙
            for i in range(1, self.grid_size - 2):
                self.grid[self.grid_size // 2, i] = 1
            # 留一个缺口
            self.grid[self.grid_size // 2, self.grid_size - 2] = 0

    def reset(self) -> Tuple[int, int]:
        """重置环境"""
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行动作

        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否结束
        """
        self.steps += 1

        # 计算下一位置
        dx, dy = self.actions[action]
        next_x = self.state[0] + dx
        next_y = self.state[1] + dy

        # 边界检查
        if (0 <= next_x < self.grid_size and
            0 <= next_y < self.grid_size and
            self.grid[next_x, next_y] != 1):  # 不是墙
            next_state = (next_x, next_y)
        else:
            next_state = self.state  # 撞墙，原地不动

        self.state = next_state

        # 奖励
        if self.state == self.goal:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            reward = 0.0
            done = True
        else:
            reward = -0.01  # 小惩罚，鼓励快速到达
            done = False

        return self.state, reward, done

    def render(self, q_table: Optional[Dict] = None):
        """可视化"""
        display = np.zeros((self.grid_size, self.grid_size), dtype=str)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 1:
                    display[i, j] = '#'
                elif (i, j) == self.goal:
                    display[i, j] = 'G'
                elif (i, j) == self.start:
                    display[i, j] = 'S'
                elif self.state and (i, j) == self.state:
                    display[i, j] = 'A'
                else:
                    display[i, j] = '.'

        print('\n'.join([''.join(row) for row in display]))
        print()


# =============================================================================
# Q-Learning (Baseline)
# =============================================================================

class QLearning:
    """标准 Q-Learning"""

    def __init__(self, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率

        # Q 表
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions)
        )

    def select_action(self, state: Tuple, greedy: bool = False) -> int:
        """epsilon-greedy 策略"""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state: Tuple, action: int,
               reward: float, next_state: Tuple, done: bool):
        """Q-Learning 更新"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (
            target - self.q_table[state][action]
        )


# =============================================================================
# Dyna-Q
# =============================================================================

class DynaQ(QLearning):
    """
    Dyna-Q: Q-Learning + Model-Based Planning

    关键区别：
    1. 学习环境模型
    2. 每步真实交互后，进行 n 步 planning
    """

    def __init__(self, n_actions: int, planning_steps: int = 10,
                 alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        super().__init__(n_actions, alpha, gamma, epsilon)

        self.planning_steps = planning_steps

        # 环境模型: model[(s, a)] = (s', r)
        # 确定性模型 (简化版)
        self.model: Dict[Tuple, Tuple] = {}

        # 记录访问过的 (state, action) 对
        self.visited_sa: List[Tuple] = []

    def update_model(self, state: Tuple, action: int,
                     reward: float, next_state: Tuple):
        """更新环境模型"""
        sa = (state, action)
        self.model[sa] = (next_state, reward)

        if sa not in self.visited_sa:
            self.visited_sa.append(sa)

    def planning(self):
        """
        Planning: 从模型中采样经验，更新 Q 值

        这是 Dyna 的核心：用模型生成的虚拟经验
        来额外更新 Q 表，提高样本效率
        """
        if len(self.visited_sa) == 0:
            return

        for _ in range(self.planning_steps):
            # 随机选择一个访问过的 (s, a)
            idx = np.random.randint(len(self.visited_sa))
            state, action = self.visited_sa[idx]

            # 从模型获取转移
            next_state, reward = self.model[(state, action)]

            # 假设 planning 的经验不会是 terminal
            # (简化：真实实现需要存储 done 信息)
            done = (next_state == (5, 5))  # 硬编码终点

            # Q-Learning 更新
            self.update(state, action, reward, next_state, done)


# =============================================================================
# Prioritized Sweeping (进阶版)
# =============================================================================

class PrioritizedSweeping(DynaQ):
    """
    Prioritized Sweeping: 优先更新那些 Q 值变化大的状态

    比 Dyna-Q 更高效：
    - 使用优先队列
    - 优先处理 TD error 大的状态
    - 反向传播更新
    """

    def __init__(self, n_actions: int, planning_steps: int = 10,
                 alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.1, theta: float = 0.0001):
        super().__init__(n_actions, planning_steps, alpha, gamma, epsilon)

        self.theta = theta  # 优先级阈值

        # 优先队列: {(s, a): priority}
        self.priority_queue: Dict[Tuple, float] = {}

        # 前驱模型: predecessors[s'] = [(s, a, r), ...]
        self.predecessors: Dict[Tuple, List] = defaultdict(list)

    def update_model(self, state: Tuple, action: int,
                     reward: float, next_state: Tuple):
        """更新模型和前驱关系"""
        super().update_model(state, action, reward, next_state)

        # 记录前驱
        pred = (state, action, reward)
        if pred not in self.predecessors[next_state]:
            self.predecessors[next_state].append(pred)

    def compute_priority(self, state: Tuple, action: int,
                        reward: float, next_state: Tuple, done: bool) -> float:
        """计算 TD error 作为优先级"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        return abs(target - self.q_table[state][action])

    def planning(self):
        """优先级扫描"""
        for _ in range(self.planning_steps):
            if not self.priority_queue:
                break

            # 选择优先级最高的 (s, a)
            sa = max(self.priority_queue, key=self.priority_queue.get)
            del self.priority_queue[sa]

            state, action = sa
            next_state, reward = self.model[sa]
            done = (next_state == (5, 5))

            # 更新 Q 值
            self.update(state, action, reward, next_state, done)

            # 反向传播：更新所有前驱状态的优先级
            for pred_state, pred_action, pred_reward in self.predecessors[state]:
                priority = self.compute_priority(
                    pred_state, pred_action, pred_reward, state, False
                )
                if priority > self.theta:
                    self.priority_queue[(pred_state, pred_action)] = priority


# =============================================================================
# 训练函数
# =============================================================================

def train_agent(env: GridWorld, agent, n_episodes: int = 200,
                use_planning: bool = False) -> Dict:
    """
    训练智能体

    Returns:
        results: 包含奖励、步数等统计信息
    """
    episode_rewards = []
    episode_steps = []
    cumulative_steps = []
    total_steps = 0

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done = env.step(action)

            # 更新 Q 值 (Direct RL)
            agent.update(state, action, reward, next_state, done)

            # Dyna: 更新模型 + Planning
            if use_planning and hasattr(agent, 'update_model'):
                agent.update_model(state, action, reward, next_state)

                # Prioritized Sweeping 需要先计算优先级
                if hasattr(agent, 'priority_queue'):
                    priority = agent.compute_priority(
                        state, action, reward, next_state, done
                    )
                    if priority > agent.theta:
                        agent.priority_queue[(state, action)] = priority

                agent.planning()

            episode_reward += reward
            steps += 1
            total_steps += 1
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        cumulative_steps.append(total_steps)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_steps = np.mean(episode_steps[-50:])
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Steps: {avg_steps:5.1f} | "
                  f"Total Steps: {total_steps}")

    return {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'cumulative_steps': cumulative_steps,
    }


def evaluate_agent(env: GridWorld, agent, n_episodes: int = 100) -> float:
    """评估智能体"""
    total_reward = 0

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, greedy=True)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                break

        total_reward += episode_reward

    return total_reward / n_episodes


# =============================================================================
# 可视化
# =============================================================================

def plot_results(results_dict: Dict[str, Dict], save_path: str = None):
    """绘制对比结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    # 1. Episode Rewards
    ax = axes[0]
    for (name, results), color in zip(results_dict.items(), colors):
        rewards = results['episode_rewards']
        # 平滑
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, color=color, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward (smoothed)')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Steps per Episode
    ax = axes[1]
    for (name, results), color in zip(results_dict.items(), colors):
        steps = results['episode_steps']
        window = 20
        smoothed = np.convolve(steps, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, color=color, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps per Episode (smoothed)')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Sample Efficiency (Cumulative Steps vs Performance)
    ax = axes[2]
    for (name, results), color in zip(results_dict.items(), colors):
        cum_steps = results['cumulative_steps']
        rewards = results['episode_rewards']
        # 累积平均奖励
        cum_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        ax.plot(cum_steps, cum_avg, label=name, color=color, alpha=0.8)
    ax.set_xlabel('Total Environment Steps')
    ax.set_ylabel('Cumulative Average Reward')
    ax.set_title('Sample Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_policy(env: GridWorld, agent, title: str = "Policy"):
    """可视化策略"""
    arrows = ['↑', '↓', '←', '→']

    print(f"\n=== {title} ===")
    for i in range(env.grid_size):
        row = ""
        for j in range(env.grid_size):
            if env.grid[i, j] == 1:
                row += " # "
            elif (i, j) == env.goal:
                row += " G "
            elif (i, j) == env.start:
                row += " S "
            else:
                state = (i, j)
                if state in agent.q_table:
                    action = np.argmax(agent.q_table[state])
                    row += f" {arrows[action]} "
                else:
                    row += " . "
        print(row)
    print()


# =============================================================================
# 主实验
# =============================================================================

def main():
    print("=" * 60)
    print("Dyna-Q vs Q-Learning 对比实验")
    print("=" * 60)
    print()

    # 环境
    env = GridWorld(grid_size=6, seed=42)

    print("GridWorld Environment:")
    env.reset()
    env.render()

    # 实验配置
    n_episodes = 200
    n_runs = 3  # 多次运行取平均

    configs = [
        ('Q-Learning', QLearning, {'n_actions': 4}, False),
        ('Dyna-Q (n=5)', DynaQ, {'n_actions': 4, 'planning_steps': 5}, True),
        ('Dyna-Q (n=10)', DynaQ, {'n_actions': 4, 'planning_steps': 10}, True),
        ('Dyna-Q (n=50)', DynaQ, {'n_actions': 4, 'planning_steps': 50}, True),
        ('Prioritized Sweeping', PrioritizedSweeping,
         {'n_actions': 4, 'planning_steps': 10}, True),
    ]

    all_results = {}

    for name, AgentClass, kwargs, use_planning in configs:
        print(f"\n{'=' * 40}")
        print(f"Training: {name}")
        print(f"{'=' * 40}")

        # 多次运行
        run_results = []
        start_time = time.time()

        for run in range(n_runs):
            np.random.seed(run * 100)
            agent = AgentClass(**kwargs)
            results = train_agent(
                env, agent, n_episodes=n_episodes, use_planning=use_planning
            )
            run_results.append(results)

        elapsed = time.time() - start_time

        # 平均结果
        avg_results = {
            'episode_rewards': np.mean(
                [r['episode_rewards'] for r in run_results], axis=0
            ),
            'episode_steps': np.mean(
                [r['episode_steps'] for r in run_results], axis=0
            ),
            'cumulative_steps': np.mean(
                [r['cumulative_steps'] for r in run_results], axis=0
            ),
        }
        all_results[name] = avg_results

        # 最终评估
        final_agent = AgentClass(**kwargs)
        train_agent(env, final_agent, n_episodes=n_episodes, use_planning=use_planning)
        eval_reward = evaluate_agent(env, final_agent, n_episodes=100)

        print(f"\nFinal Evaluation: {eval_reward:.3f}")
        print(f"Training Time: {elapsed:.1f}s ({elapsed/n_runs:.1f}s per run)")

        # 可视化策略
        visualize_policy(env, final_agent, title=name)

    # 绘制对比图
    print("\n" + "=" * 60)
    print("Plotting Results...")
    print("=" * 60)

    plot_results(all_results, save_path='./results_comparison/dyna_comparison.png')

    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nDyna-Q 核心洞察：")
    print("1. Planning steps 越多，学习越快（样本效率越高）")
    print("2. 但 planning 有计算开销，需要权衡")
    print("3. Prioritized Sweeping 更高效地分配 planning 资源")
    print("\nModel-Based RL 的价值：")
    print("- 减少真实环境交互（对机器人、自动驾驶很重要）")
    print("- 用计算换样本效率")
    print("- 模型准确性是关键瓶颈")


if __name__ == "__main__":
    main()
