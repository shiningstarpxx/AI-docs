# DreamerV3 代码走读

## 1. 代码结构概览

```
dreamerv3/
├── agent.py          # Agent 主类，训练循环
├── rssm.py           # RSSM、Encoder、Decoder
├── configs.yaml      # 配置文件
└── train.py          # 入口脚本
```

本文档基于官方 DreamerV3 实现：https://github.com/danijar/dreamerv3

---

## 2. RSSM 核心实现 (`rssm.py`)

### 2.1 关键参数

```python
class RSSM(nj.Module):
    deter: int = 4096      # 确定性状态维度
    hidden: int = 2048     # 隐藏层维度
    stoch: int = 32        # 随机变量数量
    classes: int = 32      # 每个随机变量的类别数
    norm: str = 'rms'      # 归一化方式
    act: str = 'gelu'      # 激活函数
    unimix: float = 0.01   # 均匀分布混合比例（防止坍缩）
    blocks: int = 8        # BlockLinear 的块数
    free_nats: float = 1.0 # KL 损失的下界
```

**设计解读**：
- `stoch=32, classes=32` → 总共 32×32 = 1024 个离散选项
- 离散潜在空间比连续高斯更稳定
- `unimix=0.01` 防止分布过于尖锐

### 2.2 状态空间定义

```python
@property
def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),        # h: 4096 维
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))  # z: 32×32
```

**总状态维度**：4096 + 32×32 = 5120 维

### 2.3 初始状态

```python
def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),      # 全零
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))  # 全零
    return carry
```

**设计**：从零状态开始，让模型从数据中学习初始化

### 2.4 核心动态函数 `_core()`

这是 RSSM 最关键的部分——GRU 风格的确定性状态更新：

```python
def _core(self, deter, stoch, action):
    # 1. 展平随机状态
    stoch = stoch.reshape((stoch.shape[0], -1))  # [B, 32*32] = [B, 1024]

    # 2. 动作归一化（防止极端值）
    action /= sg(jnp.maximum(1, jnp.abs(action)))

    # 3. 块设计（分组处理）
    g = self.blocks  # 8 块
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

    # 4. 输入嵌入（三个独立的 MLP）
    x0 = self.sub('dynin0', nn.Linear, self.hidden)(deter)  # h → 2048
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))

    x1 = self.sub('dynin1', nn.Linear, self.hidden)(stoch)  # z → 2048
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))

    x2 = self.sub('dynin2', nn.Linear, self.hidden)(action) # a → 2048
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))

    # 5. 组合并分组
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))

    # 6. 动态层
    for i in range(self.dynlayers):  # 默认 1 层
        x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g)(x)
        x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))

    # 7. GRU 门控（关键！）
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]

    # 8. 门控计算
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)  # 偏向保留旧信息（-1 偏置）

    # 9. 状态更新
    deter = update * cand + (1 - update) * deter
    return deter
```

**代码解读**：

```
┌───────────────────────────────────────────────────────────────┐
│  GRU 风格的状态更新                                            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  输入: h_{t-1} (deter), z_{t-1} (stoch), a_{t-1} (action)    │
│                                                               │
│  Step 1-4: 三个独立嵌入                                       │
│    x0 = MLP(h_{t-1})    确定性状态的嵌入                      │
│    x1 = MLP(z_{t-1})    随机状态的嵌入                        │
│    x2 = MLP(a_{t-1})    动作的嵌入                            │
│                                                               │
│  Step 5-6: BlockLinear 处理                                   │
│    将 4096 维分成 8 块，每块 512 维                           │
│    块内共享参数，减少计算量                                    │
│                                                               │
│  Step 7-9: GRU 门控                                           │
│    reset gate:  控制多少旧信息参与候选计算                    │
│    update gate: 控制新旧信息的混合比例                        │
│    bias=-1:     默认保留更多旧信息（长期记忆）                │
│                                                               │
│  输出: h_t (新的确定性状态)                                   │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**为什么用 `update - 1`？**
```python
update = jax.nn.sigmoid(update - 1)
```
- sigmoid(0) = 0.5，sigmoid(-1) ≈ 0.27
- 默认 update ≈ 0.27，意味着 73% 保留旧信息
- 这让模型倾向于**保持长期记忆**

### 2.5 先验分布 `_prior()`

```python
def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):  # 默认 2 层
        x = self.sub(f'prior{i}', nn.Linear, self.hidden)(x)
        x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)
```

**功能**：从确定性状态 h 预测随机状态 z 的分布（用于想象）

### 2.6 后验分布（观测时用）

在 `_observe()` 中：

```python
def _observe(self, carry, tokens, action, reset, training):
    # 1. Reset 处理（episode 边界）
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)

    # 2. 动态更新
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action)

    # 3. 后验计算（结合观测）
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):  # 默认 1 层
        x = self.sub(f'obs{i}', nn.Linear, self.hidden)(x)
        x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))

    # 4. 采样
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    return carry, (entry, feat)
```

**先验 vs 后验**：
```
先验 p(z|h):     只看确定性状态，纯预测
后验 q(z|h,o):   看确定性状态 + 真实观测，更准确

训练时用后验采样，想象时用先验
```

### 2.7 离散分布 `_dist()`

```python
def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)  # 加入均匀分布
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)  # 聚合
    return out
```

**unimix 的作用**：
```python
# 实际分布 = (1 - unimix) * softmax(logits) + unimix * uniform
# unimix = 0.01 意味着 1% 的均匀分布混合
# 防止分布过于尖锐，保持探索能力
```

### 2.8 KL 损失计算

```python
def loss(self, carry, tokens, acts, reset, training):
    # 1. 观测并获取先验/后验
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    prior = self._prior(feat['deter'])  # 先验 logits
    post = feat['logit']                 # 后验 logits

    # 2. KL Balancing（关键！）
    dyn = self._dist(sg(post)).kl(self._dist(prior))  # 固定后验，训练先验
    rep = self._dist(post).kl(self._dist(sg(prior)))  # 固定先验，训练后验

    # 3. Free bits
    if self.free_nats:
        dyn = jnp.maximum(dyn, self.free_nats)  # KL >= 1.0
        rep = jnp.maximum(rep, self.free_nats)

    losses = {'dyn': dyn, 'rep': rep}

    # 4. 熵监控
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()

    return carry, entries, losses, feat, metrics
```

**KL Balancing 图解**：
```
┌─────────────────────────────────────────────────────────────┐
│  KL Balancing: 分开更新先验和后验                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  传统 KL: D_KL(q || p) = E_q[log q - log p]                │
│    - 同时更新 q 和 p                                        │
│    - 问题：q 可能"躲避"到 p 覆盖的区域                      │
│                                                             │
│  DreamerV3 KL Balancing:                                   │
│    L_dyn = D_KL(sg(q) || p)  # 固定 q，让 p 拟合 q         │
│    L_rep = D_KL(q || sg(p))  # 固定 p，让 q 靠近 p         │
│                                                             │
│  在 agent.py 中组合:                                        │
│    loss = 0.5 * dyn + 0.1 * rep                            │
│    偏向训练先验（想象时只能用先验！）                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.9 想象函数 `imagine()`

```python
def imagine(self, carry, policy, length, training, single=False):
    if single:
        # 单步想象
        action = policy(sg(carry)) if callable(policy) else policy
        actemb = nn.DictConcat(self.act_space, 1)(action)
        deter = self._core(carry['deter'], carry['stoch'], actemb)
        logit = self._prior(deter)  # 用先验！
        stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
        carry = nn.cast(dict(deter=deter, stoch=stoch))
        feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
        return carry, (feat, action)
    else:
        # 多步想象（自回归）
        unroll = length if self.unroll else 1
        if callable(policy):
            carry, (feat, action) = nj.scan(
                lambda c, _: self.imagine(c, policy, 1, training, single=True),
                nn.cast(carry), (), length, unroll=unroll, axis=1)
        else:
            carry, (feat, action) = nj.scan(
                lambda c, a: self.imagine(c, a, 1, training, single=True),
                nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
        return carry, feat, action
```

**想象过程**：
```
s_0 (真实状态)
  │
  ├─ policy(s_0) → a_0
  │
  ├─ _core(h_0, z_0, a_0) → h_1
  │
  ├─ _prior(h_1) → p(z_1|h_1)
  │
  ├─ sample → z_1
  │
  └─ s_1 = [h_1, z_1]  (想象状态)
      │
      └─ 重复 15 步...
```

---

## 3. Encoder 实现

### 3.1 结构

```python
class Encoder(nj.Module):
    units: int = 1024      # MLP 隐藏层
    depth: int = 64        # CNN 基础通道
    mults: tuple = (2, 3, 4, 4)  # 通道倍增
    layers: int = 3        # MLP 层数
    kernel: int = 5        # 卷积核大小
    symlog: bool = True    # 是否使用 symlog 变换
```

### 3.2 前向过程

```python
def __call__(self, carry, obs, reset, training, single=False):
    outs = []

    # 1. 处理向量观测
    if self.veckeys:
        vspace = {k: self.obs_space[k] for k in self.veckeys}
        vecs = {k: obs[k] for k in self.veckeys}
        squish = nn.symlog if self.symlog else lambda x: x
        x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
        x = x.reshape((-1, *x.shape[bdims:]))
        for i in range(self.layers):
            x = self.sub(f'mlp{i}', nn.Linear, self.units)(x)
            x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
        outs.append(x)

    # 2. 处理图像观测
    if self.imgkeys:
        K = self.kernel
        imgs = [obs[k] for k in sorted(self.imgkeys)]
        x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5  # 归一化
        x = x.reshape((-1, *x.shape[bdims:]))
        for i, depth in enumerate(self.depths):  # (128, 192, 256, 256)
            x = self.sub(f'cnn{i}', nn.Conv2D, depth, K)(x)
            B, H, W, C = x.shape
            x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))  # 2x2 max pool
            x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
        x = x.reshape((x.shape[0], -1))  # 展平
        outs.append(x)

    x = jnp.concatenate(outs, -1)
    tokens = x.reshape((*bshape, *x.shape[1:]))
    return carry, {}, tokens
```

**symlog 变换**：
```python
def symlog(x):
    return sign(x) * ln(|x| + 1)

# 作用：压缩大数值，统一不同任务的尺度
# symlog(1) = 0.69
# symlog(100) = 4.62
# symlog(10000) = 9.21
```

---

## 4. Decoder 实现

### 4.1 图像解码

```python
def __call__(self, carry, feat, reset, training, single=False):
    # 特征输入
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)  # [B, 4096 + 1024] = [B, 5120]

    if self.imgkeys:
        # 计算初始分辨率
        factor = 2 ** (len(self.depths) - 1)  # 2^3 = 8
        minres = [int(x // factor) for x in self.imgres]  # 64/8 = 8
        shape = (*minres, self.depths[-1])  # (8, 8, 256)

        # BlockLinear 空间展开（关键技术）
        if self.bspace:
            u, g = math.prod(shape), self.bspace
            x0, x1 = nn.cast((feat['deter'], feat['stoch']))
            x1 = x1.reshape((*x1.shape[:-2], -1))
            x0 = x0.reshape((-1, x0.shape[-1]))
            x1 = x1.reshape((-1, x1.shape[-1]))
            x0 = self.sub('sp0', nn.BlockLinear, u, g)(x0)  # h → 8*8*256
            x0 = einops.rearrange(
                x0, '... (g h w c) -> ... h w (g c)',
                h=minres[0], w=minres[1], g=g)
            x1 = self.sub('sp1', nn.Linear, 2 * self.units)(x1)  # z → 2048
            x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
            x1 = self.sub('sp2', nn.Linear, shape)(x1)
            x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))

        # 转置卷积上采样
        for i, depth in reversed(list(enumerate(self.depths[:-1]))):
            x = x.repeat(2, -2).repeat(2, -3)  # 2x 上采样
            x = self.sub(f'conv{i}', nn.Conv2D, depth, K)(x)
            x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))

        # 最终输出
        x = x.repeat(2, -2).repeat(2, -3)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K)(x)
        x = jax.nn.sigmoid(x)  # [0, 1] 范围
```

**空间展开过程**：
```
特征 [5120]
    │
    ├─ h [4096] → BlockLinear → [8, 8, 256]
    │
    ├─ z [1024] → MLP → [8, 8, 256]
    │
    └─ 相加 → [8, 8, 256]
        │
        ├─ 上采样 2x → [16, 16, 192]
        │
        ├─ 上采样 2x → [32, 32, 128]
        │
        ├─ 上采样 2x → [64, 64, 3]
        │
        └─ sigmoid → RGB 图像
```

---

## 5. Agent 实现 (`agent.py`)

### 5.1 整体结构

```python
class Agent(embodied.jax.Agent):
    def __init__(self, obs_space, act_space, config):
        # 世界模型组件
        self.enc = rssm.Encoder(enc_space, ...)
        self.dyn = rssm.RSSM(act_space, ...)
        self.dec = rssm.Decoder(dec_space, ...)

        # 预测头
        self.rew = embodied.jax.MLPHead(scalar, ...)  # 奖励预测
        self.con = embodied.jax.MLPHead(binary, ...)  # 继续预测（非终止）

        # Actor-Critic
        self.pol = embodied.jax.MLPHead(act_space, ...)  # 策略
        self.val = embodied.jax.MLPHead(scalar, ...)     # 价值函数
        self.slowval = embodied.jax.SlowModel(...)       # EMA 慢价值函数

        # 归一化
        self.retnorm = embodied.jax.Normalize(...)  # 回报归一化
        self.valnorm = embodied.jax.Normalize(...)  # 价值归一化
        self.advnorm = embodied.jax.Normalize(...)  # 优势归一化
```

### 5.2 训练循环 `train()`

```python
def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)

    # 优化器一步更新
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)

    # 更新慢价值网络
    self.slowval.update()

    return carry, outs, metrics
```

### 5.3 损失函数 `loss()`

```python
def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}

    # ===== 世界模型损失 =====

    # 1. 编码
    enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)

    # 2. RSSM 动态 + KL 损失
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)  # {'dyn': ..., 'rep': ...}

    # 3. 解码重建
    dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)

    # 4. 奖励预测损失
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])

    # 5. 继续预测损失
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
        con *= 1 - 1 / self.config.horizon  # 折扣因子
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)

    # 6. 观测重建损失
    for key, recon in recons.items():
        space, value = self.obs_space[key], obs[key]
        target = f32(value) / 255 if isimage(space) else value
        losses[key] = recon.loss(sg(target))

    # ===== 想象训练 =====

    # 7. 从最后 K 步开始想象
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length  # 默认 15
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)

    # 8. 想象 H 步
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)

    # 9. 计算想象损失
    inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),      # 奖励预测
        self.con(inp, 2).prob(1),     # 继续概率
        self.pol(inp, 2),              # 策略
        self.val(inp, 2),              # 价值
        self.slowval(inp, 2),          # 慢价值
        self.retnorm, self.valnorm, self.advnorm,
        ...)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})

    # ===== 汇总 =====
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
    return loss, (carry, entries, outs, metrics)
```

### 5.4 想象损失 `imag_loss()`

这是 Actor-Critic 训练的核心：

```python
def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
    losses = {}

    # 1. 价值函数反归一化
    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val

    # 2. 折扣和权重
    disc = 1 if contdisc else 1 - 1 / horizon
    weight = jnp.cumprod(disc * con, 1) / disc
    last = jnp.zeros_like(con)
    term = 1 - con

    # 3. 计算 λ-returns
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

    # 4. 优势函数
    roffset, rscale = retnorm(ret, update)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale

    # 5. 策略损失（策略梯度 + 熵正则化）
    logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
    ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
    policy_loss = sg(weight[:, :-1]) * -(
        logpi * sg(adv_normed) + actent * sum(ents.values()))
    losses['policy'] = policy_loss

    # 6. 价值损失（TD + 慢目标正则化）
    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
    losses['value'] = sg(weight[:, :-1]) * (
        value.loss(sg(tar_padded)) +
        slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

    return losses, outs, metrics
```

**公式解读**：
```
策略损失:
  L_policy = -weight * (log π(a|s) * A_norm + β * H(π))

  - weight: 折扣累积权重，越远的步骤权重越小
  - log π(a|s): 动作的对数概率
  - A_norm: 归一化优势（标准化后的 TD 误差）
  - β = 3e-4: 熵正则化系数
  - H(π): 策略熵（鼓励探索）

价值损失:
  L_value = weight * (MSE(V, λ-return) + slowreg * MSE(V, V_slow))

  - λ-return: TD(λ) 目标
  - V_slow: EMA 慢更新的价值函数（稳定目标）
  - slowreg = 1.0: 正则化系数
```

### 5.5 λ-Return 计算

```python
def lambda_return(last, term, rew, val, boot, disc, lam):
    """
    计算 TD(λ) 回报

    参数:
        last: 是否是序列最后一步
        term: 是否是终止状态
        rew:  即时奖励
        val:  价值估计
        boot: bootstrap 值
        disc: 折扣因子
        lam:  λ 参数
    """
    chex.assert_equal_shape((last, term, rew, val, boot))
    rets = [boot[:, -1]]  # 从最后一步的 bootstrap 开始
    live = (1 - f32(term))[:, 1:] * disc  # 非终止 * 折扣
    cont = (1 - f32(last))[:, 1:] * lam   # 非结束 * λ
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]

    # 从后向前递归计算
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])

    return jnp.stack(list(reversed(rets))[:-1], 1)
```

**λ-Return 公式**：
```
G_t^λ = r_{t+1} + γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]

当 λ = 0: G_t = r_{t+1} + γ * V(s_{t+1})  (TD(0), 高偏差低方差)
当 λ = 1: G_t = r_{t+1} + γ * r_{t+2} + ...  (MC, 低偏差高方差)
当 λ = 0.95: 平衡偏差和方差

递归计算（从后向前）：
  G_H = V(s_H)  (bootstrap)
  G_{t} = r_{t+1} + γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}]
```

---

## 6. 关键技术总结

### 6.1 状态表示

```
┌─────────────────────────────────────────────────────────────┐
│  RSSM 状态 s = [h, z]                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  h (deter): 4096 维连续向量                                 │
│    - GRU 风格更新                                           │
│    - 长期记忆                                               │
│    - 确定性传播                                             │
│                                                             │
│  z (stoch): 32 个 32 类 categorical                        │
│    - 先验 p(z|h): 只用确定性状态                           │
│    - 后验 q(z|h,o): 结合观测                               │
│    - 离散表示更稳定                                         │
│                                                             │
│  总维度: 4096 + 32*32 = 5120                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│  DreamerV3 训练循环                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  while training:                                            │
│    # 1. 真实环境交互                                        │
│    data = env.step(policy)                                 │
│    replay_buffer.add(data)                                 │
│                                                             │
│    # 2. 世界模型训练                                        │
│    batch = replay_buffer.sample()                          │
│    L_wm = L_recon + L_rew + L_con + 0.5*L_dyn + 0.1*L_rep │
│    update(world_model, L_wm)                               │
│                                                             │
│    # 3. 想象训练                                            │
│    starts = get_states_from_batch()                        │
│    imagined_traj = imagine(starts, policy, H=15)           │
│    L_actor = policy_gradient_loss(imagined_traj)           │
│    L_critic = td_lambda_loss(imagined_traj)                │
│    update(actor_critic, L_actor + L_critic)                │
│                                                             │
│    # 4. 更新慢价值网络                                      │
│    slow_critic = 0.98 * slow_critic + 0.02 * critic        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 关键超参数

| 参数 | 默认值 | 作用 |
|:---|:---|:---|
| `deter` | 4096 | 确定性状态维度 |
| `stoch` | 32 | 随机变量数量 |
| `classes` | 32 | 每个变量的类别数 |
| `free_nats` | 1.0 | KL 损失下界 |
| `unimix` | 0.01 | 均匀分布混合比例 |
| `imag_length` | 15 | 想象步数 |
| `lambda` | 0.95 | TD(λ) 参数 |
| `actent` | 3e-4 | 策略熵正则化 |
| `slowreg` | 1.0 | 慢价值正则化 |

### 6.4 与 World Models 的对比

| 维度 | World Models | DreamerV3 |
|:---|:---|:---|
| **动态模型** | MDN-RNN (纯随机) | RSSM (确定+随机) |
| **潜在空间** | 连续高斯 | 离散 categorical |
| **控制器** | 线性 (867 参数) | 神经网络 Actor-Critic |
| **优化方法** | CMA-ES (无梯度) | 策略梯度 (有梯度) |
| **想象长度** | ~1000 步 | 15 步 |
| **训练方式** | 分阶段 | 端到端联合 |

---

## 7. 代码运行示例

### 7.1 安装

```bash
git clone https://github.com/danijar/dreamerv3.git
cd dreamerv3
pip install -e .
```

### 7.2 训练

```bash
# Atari
python dreamerv3/train.py \
  --logdir ~/logdir/atari_pong \
  --configs atari \
  --task atari_pong

# DMControl
python dreamerv3/train.py \
  --logdir ~/logdir/dmc_walker \
  --configs dmc_vision \
  --task dmc_walker_walk
```

### 7.3 关键配置

```yaml
# configs.yaml 摘要
atari:
  task: atari_pong
  encoder: {mlp_keys: [], cnn_keys: [image]}
  decoder: {mlp_keys: [], cnn_keys: [image]}
  dyn:
    deter: 4096
    stoch: 32
    classes: 32
  imag_length: 15
  batch_size: 16
  batch_length: 64
```

---

## 8. 总结

### 关键洞察

1. **双轨设计**：确定性 h 负责长期记忆，随机性 z 负责不确定性建模
2. **离散优于连续**：32×32 categorical 比连续高斯更稳定
3. **KL Balancing**：分开训练先验和后验，偏向让先验更强
4. **symlog 归一化**：统一不同任务的奖励尺度
5. **短想象 + Critic**：15 步想象 + λ-return 估计剩余价值

### 代码质量

- **模块化清晰**：RSSM、Encoder、Decoder 分离
- **JAX + Ninjax**：高效的函数式编程
- **配置灵活**：YAML 配置支持多任务

### 学习建议

1. 先理解 `_core()` 的 GRU 更新机制
2. 再理解 `loss()` 中的 KL Balancing
3. 最后理解 `imag_loss()` 的策略梯度
4. 动手实验：修改超参数观察效果
