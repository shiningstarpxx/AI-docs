# World Models 代码精读

> CarRacing World Model 实现详解

**代码文件**: `experiments/3_car_racing_world_model.py`

**论文**: World Models (Ha & Schmidhuber, 2018)

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    World Models 架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   观测 (96x96 RGB)                                          │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                           │
│   │  V: VAE     │ ──► z_t (32维)                           │
│   │  (视觉)     │                                           │
│   └─────────────┘                                           │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐         ┌─────────────┐                   │
│   │  M: MDN-RNN │ ◄────── │  a_t (动作) │                   │
│   │  (记忆)     │         └─────────────┘                   │
│   └─────────────┘                                           │
│         │                                                   │
│         ▼                                                   │
│   h_t (256维隐藏状态)                                        │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                           │
│   │ C: Controller│ ◄─── [z_t, h_t] (288维)                  │
│   │ (控制器)     │ ──► a_t (steering, gas, brake)           │
│   └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 训练流程

```
Stage 1: 收集数据
    随机策略 → 10,000 rollouts → (图像, 动作, 奖励) 序列

Stage 2a: 训练 VAE
    学会压缩 64×64 图像到 32 维潜在向量

Stage 2b: 训练 MDN-RNN
    学会预测 P(z_{t+1} | z_t, a_t, h_t)

Stage 3: 训练 Controller
    在"梦境"中用 CMA-ES 进化策略
```

---

## 2. 配置类 (Config)

**代码位置**: 第 43-127 行

```python
MODE = "paper"  # 三种模式: paper/medium/quick

class Config:
    # === 固定参数 (论文 Table 1) ===
    env_name = "CarRacing-v3"
    img_size = 64      # 图像尺寸
    latent_size = 32   # VAE 潜在空间维度
    hidden_size = 256  # LSTM 隐藏层大小
    n_gaussians = 5    # MDN 混合高斯分量数

    if MODE == "paper":
        # 数据收集
        random_rollouts = 10000       # 论文: 10000 rollouts
        max_steps_per_episode = 1000  # 每集最多 1000 步

        # VAE 训练
        vae_epochs = 10
        vae_batch_size = 100
        vae_lr = 1e-4

        # MDN-RNN 训练
        rnn_epochs = 20
        rnn_batch_size = 100
        rnn_seq_len = 999
        rnn_lr = 1e-4

        # CMA-ES
        population_size = 64
        generations = 300
        dream_rollout_length = 1000
        n_rollouts_per_eval = 16
        temperature = 1.0
```

### 参数解释

| 参数 | 值 | 含义 |
|:---|:---|:---|
| `latent_size=32` | 32 维 | 将 64×64×3=12,288 维图像压缩到 32 维 |
| `hidden_size=256` | 256 维 | LSTM 记忆容量，存储时序信息 |
| `n_gaussians=5` | 5 个 | 表达多模态未来（如转弯可左可右）|
| `temperature=1.0` | τ=1 | 控制梦境随机性 |

---

## 3. V: ConvVAE (Vision Model)

**代码位置**: 第 131-202 行

### 3.1 Encoder 结构

```python
self.encoder = nn.Sequential(
    nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64→32
    nn.ReLU(),
    nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32→16
    nn.ReLU(),
    nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16→8
    nn.ReLU(),
    nn.Conv2d(128, 256, 4, stride=2, padding=1),# 8→4
    nn.ReLU(),
)

# 4×4×256 = 4096 维 → 线性层 → 32 维
self.fc_mu = nn.Linear(256 * 4 * 4, latent_size)
self.fc_logvar = nn.Linear(256 * 4 * 4, latent_size)
```

### 维度变化图

```
输入: (batch, 3, 64, 64)     # RGB 图像
  ↓ Conv 4×4, stride=2
     (batch, 32, 32, 32)
  ↓ Conv 4×4, stride=2
     (batch, 64, 16, 16)
  ↓ Conv 4×4, stride=2
     (batch, 128, 8, 8)
  ↓ Conv 4×4, stride=2
     (batch, 256, 4, 4)
  ↓ Flatten
     (batch, 4096)
  ↓ Linear
输出: μ, log(σ²) 各 (batch, 32)
```

### 3.2 重参数化技巧 (Reparameterization Trick)

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)  # σ = exp(log(σ²)/2)
    eps = torch.randn_like(std)     # ε ~ N(0,1)
    return mu + eps * std           # z = μ + ε·σ
```

**为什么需要这个技巧？**

```
问题：直接从 N(μ, σ²) 采样，梯度无法反向传播
     因为采样操作不可微

解决：z = μ + ε·σ，其中 ε ~ N(0,1) 是固定噪声
     梯度可以通过 μ 和 σ 传回 encoder

数学上等价：z ~ N(μ, σ²)
```

### 3.3 Decoder 结构

```python
self.fc_decode = nn.Linear(latent_size, 256 * 4 * 4)

self.decoder = nn.Sequential(
    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4→8
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8→16
    nn.ReLU(),
    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16→32
    nn.ReLU(),
    nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 32→64
    nn.Sigmoid(),  # 输出 [0,1] 范围
)
```

### 3.4 VAE 损失函数

```python
def vae_loss(recon, x, mu, logvar, beta=1.0):
    # 重建损失：图像还原质量
    recon_loss = F.mse_loss(recon, x, reduction='sum')

    # KL 散度：潜在分布接近标准正态
    # D_KL(N(μ,σ²) || N(0,1)) = -½ Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss
```

**损失函数详解**：

```
总损失 = 重建损失 + β × KL 散度

重建损失 (MSE):
  - 衡量解码器重建图像的质量
  - 越小越好

KL 散度:
  - 衡量潜在分布 q(z|x) 与先验 p(z)=N(0,I) 的差异
  - 让潜在空间规整，便于采样
  - 公式推导见 02_vae_math.md

β = 1.0:
  - 标准 VAE 设置
  - β < 1: 更强调重建
  - β > 1: 更强调潜在空间规整 (β-VAE)
```

---

## 4. M: MDN-RNN (Memory Model)

**代码位置**: 第 206-303 行

### 4.1 核心结构

```python
class MDNRNN(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size, n_gaussians):
        # LSTM: 处理序列，维护隐藏状态
        input_size = latent_size + action_size  # 32 + 3 = 35
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # MDN 输出层：预测下一状态的分布参数
        self.fc_pi = nn.Linear(hidden_size, n_gaussians)              # 混合权重 π (5个)
        self.fc_mu = nn.Linear(hidden_size, n_gaussians * latent_size) # 均值 μ (5×32=160)
        self.fc_sigma = nn.Linear(hidden_size, n_gaussians * latent_size) # 标准差 σ

        # 辅助预测
        self.fc_reward = nn.Linear(hidden_size, 1)  # 预测奖励
        self.fc_done = nn.Linear(hidden_size, 1)    # 预测终止
```

### 4.2 为什么用 MDN？

```
场景：赛车在 T 字路口

        │
        │
  ┌─────┴─────┐
  │     ↑     │
  └───────────┘
      ↑ 当前位置

下一步可能：
  - 左转 (概率 0.5)
  - 右转 (概率 0.5)

单一高斯 N(μ, σ²):
  - 会预测"直走"（两个方向的均值）
  - 这是错误的！

MDN (5 个高斯混合):
  - π₁ = 0.5, μ₁ = 左转方向
  - π₂ = 0.5, μ₂ = 右转方向
  - 准确表达分叉的未来
```

### 4.3 前向传播

```python
def forward(self, z, action, hidden=None):
    batch_size, seq_len, _ = z.shape

    # 拼接输入: z_t (32维) + a_t (3维) = 35维
    x = torch.cat([z, action], dim=-1)

    # LSTM 处理序列
    lstm_out, hidden = self.lstm(x, hidden)

    # MDN 输出分布参数
    out_flat = lstm_out.reshape(batch_size * seq_len, -1)

    pi = F.softmax(self.fc_pi(out_flat), dim=-1)   # 混合权重，和为1
    mu = self.fc_mu(out_flat).view(-1, n_gaussians, latent_size)
    sigma = torch.exp(self.fc_sigma(out_flat))      # exp 保证 σ > 0
    sigma = torch.clamp(sigma, min=1e-4, max=10.0)  # 数值稳定

    # 奖励和终止预测
    reward = self.fc_reward(out_flat)
    done = torch.sigmoid(self.fc_done(out_flat))

    return pi, mu, sigma, reward, done, hidden
```

### 4.4 MDN 损失函数（负对数似然）

```python
def mdn_loss(self, pi, mu, sigma, target):
    # target: 真实的 z_{t+1}

    # 计算每个高斯分量的对数概率密度
    var = sigma ** 2
    log_prob = -0.5 * (
        math.log(2 * math.pi) +   # 常数项
        torch.log(var) +           # log(σ²)
        (target - mu) ** 2 / var   # (x-μ)²/σ²
    )
    log_prob = log_prob.sum(dim=-1)  # 对 32 维求和（假设独立）

    # 混合分布的对数似然：log(Σ πᵢ × p(x|μᵢ,σᵢ))
    # 使用 logsumexp 保证数值稳定
    log_pi = torch.log(pi + 1e-8)
    log_prob_mixture = torch.logsumexp(log_pi + log_prob, dim=-1)

    return -log_prob_mixture.mean()  # 负对数似然（越小越好）
```

**数学推导**：

```
混合高斯分布:
  p(x) = Σᵢ πᵢ × N(x | μᵢ, σᵢ²)

对数似然:
  log p(x) = log(Σᵢ πᵢ × exp(log N(x | μᵢ, σᵢ²)))
           = logsumexp(log πᵢ + log N(x | μᵢ, σᵢ²))

单个高斯的对数概率:
  log N(x | μ, σ²) = -½(log(2π) + log(σ²) + (x-μ)²/σ²)

负对数似然损失:
  L = -E[log p(x)]
```

### 4.5 从 MDN 采样

```python
def sample(self, pi, mu, sigma, temperature=1.0):
    batch_size = pi.shape[0]

    # 温度调节：控制随机性
    pi_temp = pi ** (1.0 / temperature)
    pi_temp = pi_temp / pi_temp.sum(dim=-1, keepdim=True)  # 重新归一化

    # 按权重选择一个高斯分量
    idx = torch.multinomial(pi_temp, 1).squeeze(-1)

    # 获取选中分量的参数
    batch_idx = torch.arange(batch_size, device=mu.device)
    mu_sel = mu[batch_idx, idx]
    sigma_sel = sigma[batch_idx, idx] * temperature

    # 从选中的高斯采样
    eps = torch.randn_like(mu_sel)
    return mu_sel + sigma_sel * eps
```

### 温度参数的作用

```
采样时: z ~ MDN(π, μ, σ) with temperature τ

τ < 1: 梦境更"确定"
  - 权重 π 更尖锐（接近 argmax）
  - σ 更小
  - 容易训练但可能过于"乐观"

τ = 1: 标准设置

τ > 1: 梦境更"随机"
  - 权重 π 更平滑
  - σ 更大
  - 训练困难但策略更鲁棒

论文发现：在 τ > 1 的"噩梦"中训练的 agent 更鲁棒！
```

---

## 5. C: Controller

**代码位置**: 第 307-342 行

### 5.1 极简线性结构

```python
class Controller:
    def __init__(self, input_dim, action_dim):
        # input_dim = 32 (z) + 256 (h) = 288
        # action_dim = 3 (steering, gas, brake)

        self.W = np.random.randn(action_dim, input_dim) * 0.1  # 3×288 = 864
        self.b = np.zeros(action_dim)                          # 3

        # 总参数：864 + 3 = 867 个
```

### 5.2 动作计算

```python
def get_action(self, state):
    # state = [z (32维), h (256维)] = 288 维
    raw = self.W @ state + self.b  # 线性变换 → 3 维

    # CarRacing 动作空间映射
    action = np.zeros(3)
    action[0] = np.tanh(raw[0])                    # steering: [-1, 1]
    action[1] = 1.0 / (1.0 + np.exp(-raw[1]))     # gas: [0, 1] (sigmoid)
    action[2] = 1.0 / (1.0 + np.exp(-raw[2]))     # brake: [0, 1] (sigmoid)
    return action
```

### 5.3 为什么用如此简单的 Controller？

论文明确指出：

> "We deliberately use a small controller with minimal parameters to prevent it from memorizing the game."

**设计理由**：

```
1. 防止过拟合世界模型
   - 如果 Controller 太复杂，可能学会利用世界模型的缺陷
   - 867 参数 vs 世界模型数百万参数

2. 强迫学习真正有用的策略
   - 简单 Controller 只能学到"大方向"
   - 无法记忆细节，必须依赖世界模型

3. 适合进化算法优化
   - 参数少，搜索空间小
   - CMA-ES 在低维空间效率很高
```

---

## 6. CMA-ES 进化算法

**代码位置**: 第 346-375 行

### 6.1 算法结构

```python
class SimpleCMAES:
    def __init__(self, dim, pop_size=64, sigma=0.5):
        self.dim = dim           # 参数维度 (867)
        self.pop_size = pop_size # 种群大小 (64)
        self.elite_size = pop_size // 4  # 精英数量 (16)

        self.mean = np.zeros(dim)   # 分布均值
        self.sigma = sigma          # 搜索步长
        self.C = np.ones(dim)       # 对角协方差矩阵
```

### 6.2 生成候选解 (ask)

```python
def ask(self):
    # 从当前分布采样 pop_size 个候选解
    noise = np.random.randn(self.pop_size, self.dim)
    return self.mean + self.sigma * np.sqrt(self.C) * noise
```

### 6.3 更新分布 (tell)

```python
def tell(self, population, fitness):
    # 选择精英（fitness 最高的 25%）
    elite_idx = np.argsort(fitness)[-self.elite_size:]
    elite = population[elite_idx]
    elite_fit = fitness[elite_idx]

    # 计算加权权重
    weights = np.exp(elite_fit - elite_fit.max())
    weights = weights / weights.sum()

    # 加权更新均值
    self.mean = (weights[:, None] * elite).sum(axis=0)

    # 更新协方差（简化版）
    diff = elite - self.mean
    self.C = 0.8 * self.C + 0.2 * (weights[:, None] * diff ** 2).sum(axis=0)
```

### 6.4 CMA-ES 迭代过程

```
初始化: mean = 0, C = I, σ = 0.5

第 1 代:
  ask()  → 生成 64 个候选 Controller
  评估   → 在梦境中评估每个 Controller
  tell() → 选择 16 个精英，更新分布

第 2 代:
  ask()  → 从新分布采样 64 个
  评估   → ...
  tell() → ...

  ...

第 300 代:
  分布收敛到最优区域
  mean 即为最优 Controller 参数
```

### 6.5 为什么用 CMA-ES 而非梯度下降？

```
梯度下降的问题:

1. 需要通过 M (世界模型) 反向传播
   - M 是随机的 (MDN 采样)
   - 梯度方差大，不稳定

2. 长轨迹的梯度问题
   - 1000 步的梯度链
   - 可能爆炸/消失

3. Controller 参数少 (867个)
   - 梯度信号弱
   - 可能不稳定

CMA-ES 的优势:

1. 无需梯度
   - 直接评估"最终得分"
   - 不管中间过程

2. 对噪声鲁棒
   - 每个候选评估 16 次取平均
   - 自然处理随机性

3. 并行高效
   - 64 个候选可并行评估
   - 适合分布式训练

4. 参数少时特别适合
   - 867 参数正好在 CMA-ES 舒适区
```

---

## 7. 训练流程详解

**代码位置**: 第 412-769 行

### Stage 1: 数据收集

```python
def collect_data(self):
    env = gym.make(self.config.env_name)

    for ep in range(self.config.random_rollouts):  # 10000 rollouts
        obs, _ = env.reset()

        for step in range(self.config.max_steps_per_episode):  # 最多 1000 步
            # 随机动作
            action = env.action_space.sample()

            # 预处理并存储
            frame = self.preprocess_frame(obs)  # 96×96 → 64×64, 归一化
            self.frames.append(frame)
            self.actions.append(action)

            # 环境步进
            obs, reward, terminated, truncated, _ = env.step(action)
            self.rewards.append(reward)
            self.dones.append(float(terminated or truncated))

            if terminated or truncated:
                break
```

**预处理函数**：

```python
def preprocess_frame(self, frame):
    # frame: 96×96×3 RGB
    img = Image.fromarray(frame)
    img = img.resize((64, 64))  # 缩放到 64×64
    return np.array(img) / 255.0  # 归一化到 [0, 1]
```

### Stage 2a: 训练 VAE

```python
def train_vae(self):
    optimizer = optim.Adam(self.vae.parameters(), lr=self.config.vae_lr)

    # 所有帧转为张量: (N, H, W, C) → (N, C, H, W)
    frames = torch.FloatTensor(self.frames).permute(0, 3, 1, 2).to(device)

    for epoch in range(self.config.vae_epochs):
        # 随机打乱
        idx = np.random.permutation(len(frames))

        for i in range(0, len(frames), batch_size):
            batch = frames[idx[i:i+batch_size]]

            # 前向传播
            recon, mu, logvar, _ = self.vae(batch)

            # 计算损失
            loss = vae_loss(recon, batch, mu, logvar) / batch.size(0)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Stage 2b: 训练 MDN-RNN

```python
def train_rnn(self):
    optimizer = optim.Adam(self.rnn.parameters(), lr=self.config.rnn_lr)

    # 先用 VAE 编码所有帧
    print("Encoding frames with VAE...")
    with torch.no_grad():
        z_all = []
        for i in range(0, len(frames), 256):
            batch = frames[i:i+256]
            mu, _ = self.vae.encode(batch)
            z_all.append(mu.cpu())
        z_all = torch.cat(z_all, dim=0)  # (N, 32)

    # 训练 RNN
    for epoch in range(self.config.rnn_epochs):
        for _ in range(100):  # 每个 epoch 100 个 batch
            # 随机采样序列起点
            starts = np.random.randint(0, n_samples, batch_size)

            # 构建序列
            z_seq = [z_all[s:s+seq_len] for s in starts]
            a_seq = [actions[s:s+seq_len] for s in starts]
            z_next = [z_all[s+1:s+seq_len+1] for s in starts]

            # 转为张量
            z_seq = torch.stack(z_seq).to(device)
            a_seq = torch.stack(a_seq).to(device)
            z_next = torch.stack(z_next).to(device)

            # 前向传播
            pi, mu, sigma, pred_r, pred_d, _ = self.rnn(z_seq, a_seq)

            # 计算损失
            mdn_loss = self.rnn.mdn_loss(pi, mu, sigma, z_next)
            reward_loss = F.mse_loss(pred_r, rewards)
            done_loss = F.binary_cross_entropy(pred_d, dones)
            loss = mdn_loss + reward_loss + done_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), 1.0)
            optimizer.step()
```

### Stage 3: 梦境训练 Controller

```python
def train_controller(self):
    # Controller 输入维度: z (32) + h (256) = 288
    input_dim = self.config.latent_size + self.config.hidden_size
    controller = Controller(input_dim, 3)  # 3 个动作

    # 初始化 CMA-ES
    cmaes = SimpleCMAES(controller.num_params, self.config.population_size)

    best_controller = None
    best_fitness = -float('inf')

    for gen in range(self.config.generations):
        # 生成候选解
        population = cmaes.ask()  # 64 个候选

        fitness = []
        for params in population:
            controller.set_params(params)

            # 在梦境中评估 16 次取平均
            rewards = [
                self.dream_rollout(controller, self.config.temperature)
                for _ in range(self.config.n_rollouts_per_eval)
            ]
            fitness.append(np.mean(rewards))

        # 更新分布
        cmaes.tell(population, np.array(fitness))

        # 记录最佳
        if max(fitness) > best_fitness:
            best_fitness = max(fitness)
            best_controller = Controller(input_dim, 3)
            best_controller.set_params(population[np.argmax(fitness)])

    return best_controller
```

### 梦境 Rollout 核心逻辑

```python
def dream_rollout(self, controller, temperature=1.0):
    # 从训练数据随机采样初始帧
    idx = np.random.randint(0, len(self.frames))
    frame = torch.FloatTensor(self.frames[idx]).permute(2, 0, 1).unsqueeze(0)

    # 编码初始状态
    with torch.no_grad():
        mu, _ = self.vae.encode(frame.to(device))
        z = mu.squeeze(0)  # (32,)

    hidden = None
    total_reward = 0

    for _ in range(self.config.dream_rollout_length):  # 1000 步
        # 获取 LSTM 隐藏状态
        if hidden is None:
            h = torch.zeros(self.config.hidden_size).to(device)
        else:
            h = hidden[0].squeeze(0).squeeze(0)  # (256,)

        # Controller 决策
        ctrl_input = torch.cat([z, h]).detach().cpu().numpy()  # (288,)
        action = controller.get_action(ctrl_input)  # (3,)
        action_t = torch.FloatTensor(action).to(device)

        # 世界模型预测下一状态
        z_in = z.unsqueeze(0).unsqueeze(0)    # (1, 1, 32)
        a_in = action_t.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)

        with torch.no_grad():
            pi, mu, sigma, reward, done, hidden = self.rnn(z_in, a_in, hidden)

        # 从 MDN 采样下一状态
        pi = pi.squeeze()
        mu = mu.squeeze()
        sigma = sigma.squeeze()
        z = self.rnn.sample(pi.unsqueeze(0), mu.unsqueeze(0), sigma.unsqueeze(0), temperature)
        z = z.squeeze(0)

        total_reward += reward.item()

        # 终止检查
        if done.item() > 0.5:
            break

    return total_reward
```

**关键点**：整个 rollout 完全在"梦境"（世界模型）中进行，不需要真实环境交互！

---

## 8. 评估与保存

### 真实环境评估

```python
def evaluate_real(self, controller, n_episodes=10):
    env = gym.make(self.config.env_name)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        hidden = None
        total_reward = 0

        # 编码初始帧
        frame = self.preprocess_frame(obs)
        z = self.vae.encode(frame_tensor)[0].squeeze(0)

        while not done:
            # Controller 决策（与梦境中相同）
            h = hidden[0].squeeze() if hidden else zeros(256)
            action = controller.get_action(cat([z, h]))

            # 真实环境步进
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # 编码新帧并更新 RNN
            z = self.vae.encode(preprocess(obs))[0].squeeze(0)
            _, _, _, _, _, hidden = self.rnn(z, action, hidden)

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)
```

---

## 9. 关键设计决策总结

| 设计 | 选择 | 原因 |
|:---|:---|:---|
| Vision | VAE 而非 AE | 潜在空间规整，便于采样和插值 |
| 输出分布 | MDN 而非单高斯 | 表达多模态未来分布 |
| 序列模型 | LSTM | 论文选择，GRU 效果类似 |
| Controller | 线性 (867 参数) | 防止过拟合世界模型 |
| 优化算法 | CMA-ES 而非梯度下降 | 无需通过随机采样反向传播 |
| 评估 | 16 rollouts 平均 | 减少随机性带来的方差 |
| 温度 | τ = 1.0 | 标准设置，可调节鲁棒性 |

---

## 10. 论文参数 vs 代码实现

| 参数 | 论文 | 代码 | 一致 |
|:---|:---|:---|:---|
| 图像尺寸 | 64×64 | 64×64 | ✓ |
| 潜在维度 | 32 | 32 | ✓ |
| LSTM 隐藏层 | 256 | 256 | ✓ |
| MDN 分量数 | 5 | 5 | ✓ |
| 训练 rollouts | 10,000 | 10,000 | ✓ |
| VAE lr | 0.0001 | 1e-4 | ✓ |
| RNN lr | 0.0001 | 1e-4 | ✓ |
| CMA-ES 种群 | 64 | 64 | ✓ |
| 评估次数 | 16 | 16 | ✓ |
| 进化代数 | 300+ | 300 | ✓ |

代码完全对齐论文设置。

---

## 11. 实验结果对比

| 实验 | 设置 | Dream Fitness | Real Reward |
|:---|:---|:---|:---|
| Quick (200 rollouts) | 快速测试 | ~107 | ~13 |
| Paper (10000 rollouts) | 论文设置 | 运行中... | 运行中... |
| **论文报告** | 完整训练 | - | **~900** |

---

*文档生成时间: 2024-12-09*

*基于苏格拉底式对话和代码分析*
