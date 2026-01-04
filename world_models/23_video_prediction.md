# 视频预测：时序世界建模的核心挑战

> 视频预测是世界模型的自然延伸：从状态预测到视觉预测，从单步到多步，从确定性到随机性。

## 1. 问题定义

### 1.1 任务形式化

```
输入: 过去 T 帧 {x₁, x₂, ..., x_T}
输出: 未来 K 帧 {x_{T+1}, x_{T+2}, ..., x_{T+K}}

或带动作条件:
输入: {x₁, ..., x_T} + {a₁, ..., a_{T+K-1}}
输出: {x_{T+1}, ..., x_{T+K}}
```

### 1.2 核心挑战

```
┌─────────────────────────────────────────────────────────────┐
│                   视频预测的挑战                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 高维输出空间                                             │
│     - 64x64x3 = 12,288 维                                   │
│     - 256x256x3 = 196,608 维                                │
│     - 需要高效的表示学习                                     │
│                                                              │
│  2. 时间一致性                                               │
│     - 物体应该平滑移动                                       │
│     - 不应该突然消失/出现                                    │
│     - 物理规律应该保持                                       │
│                                                              │
│  3. 未来的不确定性                                           │
│     - 多种可能的未来                                         │
│     - 球可能向左也可能向右                                   │
│     - 需要建模多模态分布                                     │
│                                                              │
│  4. 长期预测的误差累积                                       │
│     - 自回归预测误差会累积                                   │
│     - 预测越远，越模糊                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 评估指标

### 2.1 像素级指标

**PSNR (Peak Signal-to-Noise Ratio)**
$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$

**SSIM (Structural Similarity Index)**
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

### 2.2 感知指标

**LPIPS (Learned Perceptual Image Patch Similarity)**
```python
# 使用预训练网络（如 VGG）的特征
def lpips(x, y, net='vgg'):
    features_x = net(x)
    features_y = net(y)
    return distance(features_x, features_y)
```

### 2.3 视频质量指标

**FVD (Frechet Video Distance)**
```python
# 类似 FID，但针对视频
# 使用 I3D 网络提取时空特征
def fvd(real_videos, generated_videos):
    real_features = i3d(real_videos)
    gen_features = i3d(generated_videos)
    return frechet_distance(real_features, gen_features)
```

### 2.4 指标对比

| 指标 | 衡量内容 | 优点 | 缺点 |
|:---|:---|:---|:---|
| PSNR | 像素误差 | 简单直观 | 与人类感知不符 |
| SSIM | 结构相似性 | 考虑结构 | 对模糊不敏感 |
| LPIPS | 感知相似性 | 接近人类判断 | 需要预训练网络 |
| FVD | 视频分布距离 | 考虑时间 | 计算开销大 |

---

## 3. 经典方法

### 3.1 ConvLSTM

```
┌─────────────────────────────────────────────────────────────┐
│                    ConvLSTM 架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   输入帧序列: x₁, x₂, ..., x_T                               │
│        │                                                     │
│        ↓                                                     │
│   ┌─────────────────────────────────────────────┐           │
│   │              ConvLSTM Encoder               │           │
│   │  卷积操作替代全连接                          │           │
│   │  保留空间结构                               │           │
│   └─────────────────────────────────────────────┘           │
│        │                                                     │
│        ↓                                                     │
│   隐藏状态 h_T (空间保持)                                    │
│        │                                                     │
│        ↓                                                     │
│   ┌─────────────────────────────────────────────┐           │
│   │              ConvLSTM Decoder               │           │
│   │  自回归生成未来帧                            │           │
│   └─────────────────────────────────────────────┘           │
│        │                                                     │
│        ↓                                                     │
│   预测帧: x̂_{T+1}, x̂_{T+2}, ...                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**ConvLSTM 核心公式**：

$$i_t = \sigma(W_{xi} * x_t + W_{hi} * h_{t-1} + b_i)$$
$$f_t = \sigma(W_{xf} * x_t + W_{hf} * h_{t-1} + b_f)$$
$$o_t = \sigma(W_{xo} * x_t + W_{ho} * h_{t-1} + b_o)$$
$$\tilde{c}_t = \tanh(W_{xc} * x_t + W_{hc} * h_{t-1} + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

其中 $*$ 表示卷积操作。

### 3.2 PredRNN 系列

**PredRNN** 引入时空记忆流：

```python
class PredRNN(nn.Module):
    """
    双记忆流: 时间记忆 C + 空间记忆 M
    """
    def forward(self, x_seq):
        # 时间记忆 (跨时间步)
        C = [None] * self.n_layers

        # 空间记忆 (跨层传递)
        M = None

        outputs = []
        for t in range(seq_len):
            x = x_seq[:, t]

            for l in range(self.n_layers):
                if l == 0:
                    inputs = x
                else:
                    inputs = H[l-1]

                # ST-LSTM cell
                H[l], C[l], M = self.st_lstm[l](inputs, H[l], C[l], M)

            outputs.append(H[-1])

        return torch.stack(outputs, dim=1)
```

### 3.3 SVG (Stochastic Video Generation)

引入随机性处理未来的不确定性：

```
┌─────────────────────────────────────────────────────────────┐
│                    SVG 架构                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   确定性路径 (Deterministic)                                 │
│   ├── 捕获可预测的部分                                       │
│   └── h_t = f_det(h_{t-1}, x_{t-1})                         │
│                                                              │
│   随机路径 (Stochastic)                                      │
│   ├── 建模不确定性                                           │
│   └── z_t ~ q(z_t | x_{1:t}) 或 p(z_t | h_t)                │
│                                                              │
│   生成:                                                      │
│   x̂_t = g(h_t, z_t)                                         │
│                                                              │
│   训练 (VAE 风格):                                           │
│   L = E_q[log p(x|z)] - KL[q(z|x) || p(z|h)]                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 现代方法：Transformer 与扩散

### 4.1 Video Transformer

```python
class VideoTransformer(nn.Module):
    """
    时空 Transformer 用于视频预测
    """
    def __init__(self):
        # 时空位置编码
        self.pos_embed = SpatioTemporalPosEmbed()

        # 时空注意力
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                spatial_attn=SpatialAttention(),
                temporal_attn=TemporalAttention(),
                ffn=FeedForward()
            ) for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: [B, T, H, W, C]

        # Patch embedding
        x = self.patch_embed(x)  # [B, T, N_patches, D]

        # 添加位置编码
        x = x + self.pos_embed(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 解码
        x = self.decoder(x)

        return x
```

### 4.2 视频扩散模型

```
┌─────────────────────────────────────────────────────────────┐
│                 Video Diffusion Model                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   前向过程 (添加噪声):                                        │
│   x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε                            │
│                                                              │
│   反向过程 (去噪):                                           │
│   x_{t-1} = μ(x_t, t) + σ_t z                               │
│                                                              │
│   3D UNet 架构:                                              │
│   ├── 空间卷积 (处理单帧)                                    │
│   ├── 时间注意力 (跨帧关联)                                  │
│   └── 条件注入 (文本/图像/动作)                              │
│                                                              │
│   优势:                                                      │
│   - 高质量生成                                               │
│   - 多样性好                                                 │
│   - 可控生成                                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Sora 架构要点

OpenAI Sora 的关键技术：

```
1. Spacetime Latent Patches
   - 将视频压缩到潜在空间
   - 时空 patch 化
   - 处理可变分辨率和时长

2. Diffusion Transformer (DiT)
   - Transformer 替代 UNet
   - 更好的缩放性
   - 更强的长程依赖建模

3. 训练数据
   - 大规模视频数据
   - 多样化场景
   - 高质量标注
```

---

## 5. 动作条件视频预测

### 5.1 与世界模型的联系

```
世界模型:
  P(s_{t+1} | s_t, a_t)  状态转移
  ↓
动作条件视频预测:
  P(x_{t+1} | x_{1:t}, a_t)  观测预测

本质是同一个问题的不同层次
```

### 5.2 GameGAN / DIAMOND

**GameGAN**: 生成可交互的游戏环境

```python
class GameGAN(nn.Module):
    """
    学习游戏环境的动态
    """
    def __init__(self):
        # 动态引擎 (预测下一状态)
        self.dynamics = DynamicsEngine()

        # 渲染引擎 (状态→图像)
        self.renderer = NeuralRenderer()

        # 记忆模块 (长期依赖)
        self.memory = MemoryModule()

    def forward(self, x_t, a_t):
        # 编码当前观测
        h_t = self.encoder(x_t)

        # 更新记忆
        m_t = self.memory.update(h_t)

        # 动态预测
        h_next = self.dynamics(h_t, a_t, m_t)

        # 渲染
        x_next = self.renderer(h_next)

        return x_next
```

**DIAMOND**: 扩散模型作为世界模型

```
核心思想:
- 用扩散模型预测下一帧
- 动作作为条件
- 在 Atari 上达到 SOTA

优势:
- 多模态预测 (未来的多种可能)
- 高质量生成
- 与 RL 自然结合
```

---

## 6. 物理场景预测

### 6.1 学习物理规律

```python
class PhysicsPredictor(nn.Module):
    """
    学习物理交互的视频预测
    """
    def __init__(self):
        # 物体检测/跟踪
        self.object_detector = ObjectDetector()

        # 图神经网络 (物体间交互)
        self.interaction_net = InteractionNetwork()

        # 渲染器 (状态→图像)
        self.renderer = DifferentiableRenderer()

    def forward(self, x_seq):
        # 检测物体
        objects = self.object_detector(x_seq)

        # 预测物理交互
        for t in range(future_steps):
            # 物体间关系推理
            interactions = self.interaction_net(objects)

            # 更新物体状态
            objects = self.update_physics(objects, interactions)

        # 渲染
        x_pred = self.renderer(objects)

        return x_pred
```

### 6.2 归纳偏置的重要性

| 归纳偏置 | 作用 | 实现方式 |
|:---|:---|:---|
| 物体性 | 世界由物体组成 | 物体中心表示 |
| 交互性 | 物体间有相互作用 | 图神经网络 |
| 守恒律 | 能量/动量守恒 | 约束损失 |
| 时间对称 | 物理规律时间不变 | 可逆网络 |

---

## 7. 实践指南

### 7.1 数据集选择

| 数据集 | 类型 | 复杂度 | 用途 |
|:---|:---|:---|:---|
| Moving MNIST | 合成 | 简单 | 基线测试 |
| KTH | 人体动作 | 中等 | 动作预测 |
| BAIR Robot | 机器人 | 中等 | 动作条件 |
| Kinetics | 自然视频 | 高 | 通用预测 |

### 7.2 ConvLSTM 基线实现

```python
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class VideoPredictor(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_channels, 3, padding=1),
        )

        # ConvLSTM
        self.convlstm = ConvLSTMCell(hidden_channels, hidden_channels, 3)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_seq, n_future):
        B, T, C, H, W = x_seq.shape

        # 初始化隐藏状态
        h = torch.zeros(B, self.convlstm.hidden_channels, H, W, device=x_seq.device)
        c = torch.zeros_like(h)

        # 编码历史帧
        for t in range(T):
            x = self.encoder(x_seq[:, t])
            h, c = self.convlstm(x, (h, c))

        # 预测未来帧
        predictions = []
        x = self.encoder(x_seq[:, -1])
        for _ in range(n_future):
            h, c = self.convlstm(x, (h, c))
            pred = self.decoder(h)
            predictions.append(pred)
            x = self.encoder(pred)

        return torch.stack(predictions, dim=1)
```

### 7.3 训练技巧

```python
# 1. 预定采样 (Scheduled Sampling)
def scheduled_sampling(model, x_seq, teacher_forcing_ratio):
    """
    训练时逐渐减少 teacher forcing
    """
    for t in range(future_steps):
        if random.random() < teacher_forcing_ratio:
            # 使用真实帧
            input_frame = x_seq[:, t]
        else:
            # 使用预测帧
            input_frame = pred_frame

        pred_frame = model.step(input_frame)

# 2. 对抗训练
def adversarial_loss(discriminator, real_video, fake_video):
    real_score = discriminator(real_video)
    fake_score = discriminator(fake_video)
    return -torch.log(real_score).mean() - torch.log(1 - fake_score).mean()

# 3. 感知损失
def perceptual_loss(vgg, pred, target):
    pred_features = vgg(pred)
    target_features = vgg(target)
    return F.mse_loss(pred_features, target_features)
```

---

## 8. 总结

### 8.1 方法演进

```
确定性预测 (ConvLSTM)
    ↓
随机预测 (SVG, VRNN)
    ↓
Transformer 架构 (VideoGPT)
    ↓
扩散模型 (Video Diffusion, Sora)
```

### 8.2 与世界模型的关系

| 世界模型 | 视频预测 |
|:---|:---|
| 状态转移 P(s'|s,a) | 帧预测 P(x'|x,a) |
| 潜在空间 | 像素空间 |
| 用于规划 | 用于生成 |
| 强化学习 | 计算机视觉 |

**统一视角**：都是学习环境动态的表示

### 8.3 延伸阅读

**经典论文**：
- Srivastava et al. (2015). "Unsupervised Learning of Video Representations using LSTMs"
- Denton & Fergus (2018). "Stochastic Video Generation with a Learned Prior" (SVG)
- Ho et al. (2022). "Video Diffusion Models"

**前沿工作**：
- Sora (OpenAI, 2024)
- Genie (DeepMind, 2024)
- DIAMOND (2024)

---

*最后更新: 2025-12-18*
