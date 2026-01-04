"""
MacBook MPS Scaling Law 实验框架
==================================

作者: peixingxin
日期: 2025-12-25
目标: 在 MacBook (Apple Silicon) 上高效验证 Scaling Law

特点:
- ✅ 充分利用 MPS 加速
- ✅ 智能内存管理
- ✅ 早停与外推结合
- ✅ 完整的监控与可视化
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import gc
from scipy.optimize import curve_fit

# ============================================================================
# 1. MPS 设备管理
# ============================================================================

def get_mps_device():
    """获取 MPS 设备（如果可用）"""
    if torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("⚠️ MPS not available because PyTorch was not built with MPS enabled.")
            return torch.device("cpu")
        print("✅ Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    else:
        print("⚠️ MPS device not found, using CPU")
        return torch.device("cpu")

def clear_mps_cache():
    """清理 MPS 缓存"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def get_memory_usage():
    """获取内存使用情况"""
    import psutil
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return f"{mem_mb:.0f} MB"

# ============================================================================
# 2. 模型定义
# ============================================================================

@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 256   # 较短的序列（节省内存）
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    
    @property
    def n_params(self):
        """估算参数量"""
        # 粗略估计
        embed_params = self.vocab_size * self.d_model
        layer_params = (
            4 * self.d_model * self.d_model +  # Attention QKV + O
            2 * self.d_model * self.d_ff       # FFN
        ) * self.n_layers
        return embed_params + layer_params

class SimpleGPT(nn.Module):
    """简化的 GPT 模型（用于 Scaling 实验）"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token + Position embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        
        # Embeddings
        tok_emb = self.token_embed(input_ids)
        pos_emb = self.pos_embed(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        
        return logits, loss

class TransformerBlock(nn.Module):
    """Transformer block"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, 
            config.n_heads, 
            dropout=config.dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # FFN
        x = x + self.mlp(self.ln2(x))
        return x

# ============================================================================
# 3. 数据加载
# ============================================================================

class DummyTextDataset(Dataset):
    """虚拟数据集（用于快速测试）"""
    
    def __init__(self, n_tokens: int, seq_len: int, vocab_size: int):
        self.n_samples = n_tokens // seq_len
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 生成随机序列
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }

# ============================================================================
# 4. 训练器
# ============================================================================

class MPSTrainer:
    """针对 MPS 优化的训练器"""
    
    def __init__(
        self, 
        model: nn.Module,
        device: torch.device,
        config: Dict
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 3e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_steps', 10000)
        )
        
        # History
        self.history = {'loss': [], 'step': []}
        
    def train(
        self, 
        train_loader: DataLoader, 
        max_steps: Optional[int] = None,
        eval_interval: int = 100,
        early_stop: bool = True
    ):
        """训练循环"""
        self.model.train()
        step = 0
        max_steps = max_steps or self.config.get('max_steps', 10000)
        
        print(f"🚀 Starting training (max {max_steps} steps)")
        print(f"   Device: {self.device}")
        print(f"   Memory: {get_memory_usage()}")
        
        start_time = time.time()
        
        while step < max_steps:
            for batch in train_loader:
                # 移动到设备
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward
                _, loss = self.model(input_ids, targets=labels)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                # 记录
                self.history['loss'].append(loss.item())
                self.history['step'].append(step)
                
                # 日志
                if step % eval_interval == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = (step * input_ids.numel()) / elapsed
                    print(f"Step {step}/{max_steps} | "
                          f"Loss: {loss.item():.4f} | "
                          f"LR: {self.scheduler.get_last_lr()[0]:.6f} | "
                          f"Tokens/s: {tokens_per_sec:.0f} | "
                          f"Mem: {get_memory_usage()}")
                
                # 早停检查
                if early_stop and step > 0.2 * max_steps:
                    if self._should_early_stop():
                        print(f"⚡ Early stopping at step {step}")
                        return self._get_results()
                
                step += 1
                if step >= max_steps:
                    break
                
                # 定期清理内存
                if step % 500 == 0:
                    clear_mps_cache()
        
        print(f"✅ Training completed in {time.time() - start_time:.1f}s")
        return self._get_results()
    
    def _should_early_stop(self) -> bool:
        """判断是否应该早停"""
        if len(self.history['loss']) < 100:
            return False
        
        # 检查最近的 loss 是否收敛
        recent_losses = self.history['loss'][-100:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # 如果标准差很小，说明已经收敛
        return loss_std / loss_mean < 0.01
    
    def _get_results(self) -> Dict:
        """获取训练结果"""
        final_loss = np.mean(self.history['loss'][-100:])
        return {
            'final_loss': final_loss,
            'history': self.history,
            'n_steps': len(self.history['step'])
        }

# ============================================================================
# 5. Scaling Law 实验
# ============================================================================

class ScalingExperiment:
    """Scaling Law 实验管理器"""
    
    def __init__(self, device: torch.device, save_dir: str = "./results"):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_experiment(
        self,
        n_params_list: List[int],
        n_tokens_list: List[int],
        mode: str = 'quick'
    ):
        """运行完整实验"""
        print("=" * 80)
        print("📊 Scaling Law Experiment")
        print("=" * 80)
        print(f"Mode: {mode}")
        print(f"Parameter scales: {[f'{n/1e6:.1f}M' for n in n_params_list]}")
        print(f"Data scales: {[f'{n/1e6:.1f}M' for n in n_tokens_list]}")
        print("=" * 80)
        
        total_experiments = len(n_params_list) * len(n_tokens_list)
        current = 0
        
        for n_params in n_params_list:
            for n_tokens in n_tokens_list:
                current += 1
                print(f"\n[{current}/{total_experiments}] Running experiment:")
                print(f"  Params: {n_params/1e6:.1f}M")
                print(f"  Tokens: {n_tokens/1e6:.1f}M")
                
                # 运行单次实验
                result = self._run_single_experiment(n_params, n_tokens, mode)
                result['n_params'] = n_params
                result['n_tokens'] = n_tokens
                self.results.append(result)
                
                # 保存中间结果
                self._save_results()
                
                # 清理内存
                clear_mps_cache()
        
        print("\n" + "=" * 80)
        print("✅ All experiments completed!")
        print("=" * 80)
        
        return self.results
    
    def _run_single_experiment(
        self, 
        n_params: int, 
        n_tokens: int,
        mode: str
    ) -> Dict:
        """运行单个实验"""
        # 1. 创建模型配置
        config = self._params_to_config(n_params)
        model = SimpleGPT(config)
        
        print(f"  Model: {config.n_layers} layers, {config.d_model} dim")
        print(f"  Actual params: {config.n_params/1e6:.1f}M")
        
        # 2. 创建数据
        dataset = DummyTextDataset(
            n_tokens=int(n_tokens),
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self._get_batch_size(n_params, mode),
            shuffle=True,
            num_workers=0  # MPS 不支持多进程
        )
        
        # 3. 训练
        trainer = MPSTrainer(
            model=model,
            device=self.device,
            config={
                'lr': 3e-4,
                'weight_decay': 0.01,
                'max_steps': self._get_max_steps(n_params, n_tokens, mode)
            }
        )
        
        result = trainer.train(
            dataloader,
            eval_interval=100,
            early_stop=(mode != 'full')
        )
        
        return result
    
    def _params_to_config(self, n_params: int) -> ModelConfig:
        """根据参数量生成配置"""
        # 简单的启发式规则
        if n_params < 10e6:  # < 10M
            return ModelConfig(n_layers=4, d_model=256, n_heads=4)
        elif n_params < 50e6:  # < 50M
            return ModelConfig(n_layers=6, d_model=384, n_heads=6)
        elif n_params < 150e6:  # < 150M
            return ModelConfig(n_layers=8, d_model=512, n_heads=8)
        elif n_params < 500e6:  # < 500M
            return ModelConfig(n_layers=12, d_model=768, n_heads=12)
        else:  # >= 500M
            return ModelConfig(n_layers=16, d_model=1024, n_heads=16)
    
    def _get_batch_size(self, n_params: int, mode: str) -> int:
        """根据模型大小动态调整 batch size"""
        if mode == 'quick':
            base_bs = 32
        elif mode == 'dev':
            base_bs = 16
        else:
            base_bs = 8
        
        # 大模型用小 batch
        if n_params > 100e6:
            return max(1, base_bs // 4)
        elif n_params > 50e6:
            return max(2, base_bs // 2)
        else:
            return base_bs
    
    def _get_max_steps(self, n_params: int, n_tokens: int, mode: str) -> int:
        """计算训练步数"""
        batch_size = self._get_batch_size(n_params, mode)
        seq_len = 256
        
        # 总 token 数 = steps * batch_size * seq_len
        # steps = n_tokens / (batch_size * seq_len)
        steps = int(n_tokens / (batch_size * seq_len))
        
        # 限制最大步数
        if mode == 'quick':
            return min(steps, 1000)
        elif mode == 'dev':
            return min(steps, 5000)
        else:
            return steps
    
    def _save_results(self):
        """保存结果"""
        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

# ============================================================================
# 6. 结果分析
# ============================================================================

class ScalingAnalyzer:
    """Scaling Law 分析器"""
    
    def __init__(self, results: List[Dict]):
        self.results = results
    
    def fit_power_law(self):
        """拟合幂律"""
        # 提取数据
        N = np.array([r['n_params'] for r in self.results])
        D = np.array([r['n_tokens'] for r in self.results])
        L = np.array([r['final_loss'] for r in self.results])
        
        # 拟合参数量的 scaling
        N_unique = np.unique(N)
        L_vs_N = []
        for n in N_unique:
            mask = N == n
            L_vs_N.append(np.mean(L[mask]))
        L_vs_N = np.array(L_vs_N)
        
        def power_law(x, a, alpha, c):
            return a * x**(-alpha) + c
        
        params_N, _ = curve_fit(power_law, N_unique, L_vs_N, p0=[100, 0.1, 2.0])
        a_n, alpha_n, c_n = params_N
        
        # 拟合数据量的 scaling
        D_unique = np.unique(D)
        L_vs_D = []
        for d in D_unique:
            mask = D == d
            L_vs_D.append(np.mean(L[mask]))
        L_vs_D = np.array(L_vs_D)
        
        params_D, _ = curve_fit(power_law, D_unique, L_vs_D, p0=[100, 0.1, 2.0])
        a_d, alpha_d, c_d = params_D
        
        print("\n" + "=" * 80)
        print("📈 Scaling Law 拟合结果")
        print("=" * 80)
        print(f"参数量 Scaling: L(N) = {a_n:.2f} * N^(-{alpha_n:.3f}) + {c_n:.3f}")
        print(f"  指数 α_n = {alpha_n:.3f}")
        print(f"  (Kaplan 2020: α_n ≈ 0.076)")
        print()
        print(f"数据量 Scaling: L(D) = {a_d:.2f} * D^(-{alpha_d:.3f}) + {c_d:.3f}")
        print(f"  指数 α_d = {alpha_d:.3f}")
        print(f"  (Kaplan 2020: α_d ≈ 0.095)")
        print("=" * 80)
        
        return {
            'N_scaling': params_N,
            'D_scaling': params_D
        }
    
    def extrapolate(self, target_params: float):
        """外推到目标规模"""
        params = self.fit_power_law()
        a_n, alpha_n, c_n = params['N_scaling']
        
        predicted_loss = a_n * target_params**(-alpha_n) + c_n
        
        print(f"\n外推预测:")
        print(f"  目标规模: {target_params/1e9:.1f}B 参数")
        print(f"  预测 loss: {predicted_loss:.3f}")
        
        return predicted_loss
    
    def plot(self, save_path: str = 'scaling_curves.png'):
        """绘制 scaling 曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 提取数据
        N = np.array([r['n_params'] for r in self.results])
        D = np.array([r['n_tokens'] for r in self.results])
        L = np.array([r['final_loss'] for r in self.results])
        
        # 左图：参数量 scaling
        axes[0].scatter(N, L, alpha=0.6, s=50)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Parameters (N)', fontsize=12)
        axes[0].set_ylabel('Loss (L)', fontsize=12)
        axes[0].set_title('Parameter Scaling', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # 拟合曲线
        N_sorted = np.sort(np.unique(N))
        params = self.fit_power_law()
        a_n, alpha_n, c_n = params['N_scaling']
        L_fit = a_n * N_sorted**(-alpha_n) + c_n
        axes[0].plot(N_sorted, L_fit, 'r--', linewidth=2, 
                     label=f'L(N) ∝ N^(-{alpha_n:.3f})')
        axes[0].legend()
        
        # 右图：数据量 scaling
        axes[1].scatter(D, L, alpha=0.6, s=50, color='green')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Training Tokens (D)', fontsize=12)
        axes[1].set_ylabel('Loss (L)', fontsize=12)
        axes[1].set_title('Data Scaling', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # 拟合曲线
        D_sorted = np.sort(np.unique(D))
        a_d, alpha_d, c_d = params['D_scaling']
        L_fit = a_d * D_sorted**(-alpha_d) + c_d
        axes[1].plot(D_sorted, L_fit, 'r--', linewidth=2,
                     label=f'L(D) ∝ D^(-{alpha_d:.3f})')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")

# ============================================================================
# 7. 主程序
# ============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'dev', 'full'], default='quick')
    args = parser.parse_args()
    
    # 1. 设备
    device = get_mps_device()
    
    # 2. 实验配置
    if args.mode == 'quick':
        print("🚀 Quick mode: 快速验证（~2小时）")
        n_params_list = [5e6, 20e6, 80e6]  # 5M, 20M, 80M
        n_tokens_list = [10e6, 50e6]       # 10M, 50M
    elif args.mode == 'dev':
        print("🚀 Dev mode: 开发模式（~1天）")
        n_params_list = [5e6, 20e6, 80e6, 200e6]  # 5M, 20M, 80M, 200M
        n_tokens_list = [10e6, 50e6, 200e6]       # 10M, 50M, 200M
    else:
        print("🚀 Full mode: 完整实验（~1周）")
        n_params_list = [5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6]
        n_tokens_list = [10e6, 50e6, 200e6, 500e6]
    
    # 3. 运行实验
    experiment = ScalingExperiment(device=device, save_dir=f'./results_{args.mode}')
    results = experiment.run_experiment(n_params_list, n_tokens_list, mode=args.mode)
    
    # 4. 分析结果
    analyzer = ScalingAnalyzer(results)
    analyzer.fit_power_law()
    
    # 5. 外推预测
    print("\n" + "=" * 80)
    print("🔮 外推预测")
    print("=" * 80)
    analyzer.extrapolate(1.5e9)   # GPT-2 XL
    analyzer.extrapolate(175e9)   # GPT-3
    
    # 6. 可视化
    analyzer.plot(save_path=f'./results_{args.mode}/scaling_curves.png')
    
    print("\n✅ 实验完成！")

if __name__ == '__main__':
    main()
