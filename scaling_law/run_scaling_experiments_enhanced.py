"""
增强版 Scaling Law 实验脚本
==========================

改进:
1. 增加训练步数 (默认 3000)
2. 添加 warmup + cosine decay 学习率调度
3. 训练曲线可视化
4. 预测外推到 GPT-3/4 规模
5. 4子图完整展示

作者: peixingxin
日期: 2026-01-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy.optimize import curve_fit
import gc

# ============================================================================
# 配置
# ============================================================================

@dataclass
class ScalingConfig:
    """实验配置"""
    # 模型规模
    n_params_list: List[float] = None
    n_tokens_list: List[float] = None
    
    # 训练配置
    max_steps: int = 3000  # 增加到 3000
    batch_size: int = 32
    seq_len: int = 128
    learning_rate: float = 3e-4
    warmup_steps: int = 300  # warmup
    
    # 设备
    device: str = "mps"
    
    # 保存路径
    save_dir: str = "./scaling_results"
    
    def __post_init__(self):
        if self.n_params_list is None:
            self.n_params_list = [5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6]
        if self.n_tokens_list is None:
            self.n_tokens_list = [10e6, 50e6, 100e6, 200e6, 500e6, 1e9]


# ============================================================================
# 模型定义
# ============================================================================

class SimpleTransformer(nn.Module):
    """简化的 Transformer 语言模型"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(512, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x + self.pos_embedding[:seq_len, :]
        x = self.transformer(x)
        logits = self.output(x)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def create_model_from_params(target_params: float, vocab_size: int = 10000) -> SimpleTransformer:
    """根据目标参数量创建模型"""
    
    configs = [
        (128, 4, 4),    # ~2.5M
        (192, 4, 4),    # ~5M
        (256, 4, 6),    # ~10M
        (384, 6, 6),    # ~23M
        (512, 8, 8),    # ~50M
        (640, 8, 10),   # ~100M
        (768, 12, 12),  # ~124M
        (896, 14, 14),  # ~200M
        (1024, 16, 16), # ~355M
        (1152, 16, 18), # ~500M
    ]
    
    best_config = None
    min_diff = float('inf')
    
    for d_model, n_heads, n_layers in configs:
        estimated_params = (
            vocab_size * d_model * 2 +
            n_layers * (
                3 * d_model * d_model +
                d_model * d_model +
                2 * d_model * 4 * d_model
            )
        )
        
        diff = abs(estimated_params - target_params)
        if diff < min_diff:
            min_diff = diff
            best_config = (d_model, n_heads, n_layers)
    
    d_model, n_heads, n_layers = best_config
    model = SimpleTransformer(vocab_size, d_model, n_heads, n_layers)
    
    actual_params = model.count_parameters()
    print(f"  Target: {target_params/1e6:.1f}M, Actual: {actual_params/1e6:.1f}M")
    
    return model


# ============================================================================
# 数据集
# ============================================================================

class DummyTextDataset(Dataset):
    """虚拟文本数据集"""
    
    def __init__(self, n_tokens: int, seq_len: int = 128, vocab_size: int = 10000):
        self.n_tokens = int(n_tokens)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_samples = self.n_tokens // seq_len
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x


# ============================================================================
# 训练与评估
# ============================================================================

def train_model(model: nn.Module, dataloader: DataLoader, device: torch.device,
                max_steps: int = 3000, lr: float = 3e-4, warmup_steps: int = 300) -> Tuple[List[float], Dict]:
    """训练模型并记录损失曲线"""
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 学习率调度器 (warmup + cosine decay)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    losses = []
    step = 0
    
    print(f"  Training {max_steps} steps (warmup={warmup_steps})...")
    start_time = time.time()
    
    while step < max_steps:
        for batch in dataloader:
            if step >= max_steps:
                break
            
            batch = batch.to(device)
            
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            if step % 200 == 0:
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                print(f"    Step {step}/{max_steps} | Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
            
            step += 1
    
    total_time = time.time() - start_time
    final_loss = get_final_loss(losses)
    print(f"  ✅ Completed in {total_time:.1f}s | Final Loss: {final_loss:.4f}")
    
    metadata = {
        'total_time': total_time,
        'final_loss': final_loss,
        'initial_loss': losses[0] if losses else None,
        'min_loss': min(losses) if losses else None,
        'losses': losses,  # 保存完整 loss 曲线
    }
    
    return losses, metadata


def get_final_loss(losses: List[float], window: int = 200) -> float:
    """获取最终损失（取最后窗口的平均）"""
    if len(losses) < window:
        return np.mean(losses)
    return np.mean(losses[-window:])


# ============================================================================
# Scaling Law 分析
# ============================================================================

def power_law(x, a, b, c):
    """幂律函数: y = c + a / x^b"""
    return c + a / (x ** b)


def fit_scaling_law(x_data: np.ndarray, y_data: np.ndarray, 
                    x_name: str = "N") -> Tuple[float, float, float, float]:
    """拟合 Scaling Law，返回 (a, b, c, r2)"""
    
    p0 = [100, 0.08, 1.8]
    
    try:
        params, _ = curve_fit(power_law, x_data, y_data, p0=p0, maxfev=10000)
        a, b, c = params
        
        # 计算 R²
        y_pred = power_law(x_data, *params)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        print(f"\n拟合结果 (L({x_name})):")
        print(f"  L({x_name}) = {c:.3f} + {a:.1f} / {x_name}^{b:.3f}")
        print(f"  R² = {r2:.4f}")
        
        return a, b, c, r2
    
    except Exception as e:
        print(f"拟合失败: {e}")
        return None, None, None, None


# ============================================================================
# 可视化
# ============================================================================

def plot_scaling_curves(results: Dict, save_dir: Path):
    """生成完整的 Scaling Law 可视化图表 (2x2)"""
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    param_results = results.get('param_scaling', {})
    data_results = results.get('data_scaling', {})
    param_metadata = results.get('param_metadata', {})
    data_metadata = results.get('data_metadata', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # ========== 子图 1: 参数 Scaling + 预测 ==========
    ax1 = axes[0, 0]
    
    if param_results:
        n_params = np.array(sorted(param_results.keys()))
        losses = np.array([param_results[n] for n in n_params])
        
        # 实验数据
        ax1.loglog(n_params, losses, 'o', markersize=12, linewidth=3,
                   label='实验数据', color='#2563eb', markeredgewidth=2, 
                   markeredgecolor='white')
        
        # 拟合曲线和预测
        a, b, c, r2 = fit_scaling_law(n_params, losses, "N")
        if a is not None:
            # 外推到 GPT-4 规模
            n_range = np.logspace(np.log10(n_params.min()), 
                                  np.log10(1.8e12), 200)  # 到 1.8T
            loss_fit = power_law(n_range, a, b, c)
            
            # 分两段绘制：已知范围和外推范围
            n_known = n_range[n_range <= n_params.max()]
            n_extrap = n_range[n_range > n_params.max()]
            
            ax1.loglog(n_known, power_law(n_known, a, b, c), '-', linewidth=3,
                      label=f'拟合 (R²={r2:.3f})', color='#dc2626')
            ax1.loglog(n_extrap, power_law(n_extrap, a, b, c), '--', linewidth=3,
                      label='外推预测', color='#dc2626', alpha=0.6)
        
            # Kaplan et al. (2020) 理论
            kaplan_loss = power_law(n_range, 450, 0.076, 1.69)
            ax1.loglog(n_range, kaplan_loss, ':', linewidth=2,
                      label='Kaplan (2020)', color='#059669', alpha=0.7)
            
            # 标注重要模型并预测 loss
            important_models = [
                (124e6, "GPT-2 Small"),
                (1.5e9, "GPT-2 XL"),
                (6.7e9, "GPT-J 6B"),
                (175e9, "GPT-3 175B"),
                (1.8e12, "GPT-4 (估计)"),
            ]
            
            for n, name in important_models:
                if n >= n_params.min():
                    loss_pred = power_law(n, a, b, c)
                    ax1.plot(n, loss_pred, 'v', markersize=10, color='#7c3aed')
                    ax1.text(n, loss_pred * 0.85, name, fontsize=10, ha='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('参数量 (N)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss (L)', fontsize=14, fontweight='bold')
    ax1.set_title('参数 Scaling Law 及外推预测', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='upper right')
    
    # ========== 子图 2: 数据 Scaling + 预测 ==========
    ax2 = axes[0, 1]
    
    if data_results:
        n_tokens = np.array(sorted(data_results.keys()))
        losses = np.array([data_results[d] for d in n_tokens])
        
        ax2.loglog(n_tokens, losses, 's', markersize=12, linewidth=3,
                   label='实验数据', color='#2563eb', markeredgewidth=2,
                   markeredgecolor='white')
        
        a, b, c, r2 = fit_scaling_law(n_tokens, losses, "D")
        if a is not None:
            d_range = np.logspace(np.log10(n_tokens.min()),
                                  np.log10(15e12), 200)  # 到 15T tokens
            
            d_known = d_range[d_range <= n_tokens.max()]
            d_extrap = d_range[d_range > n_tokens.max()]
            
            ax2.loglog(d_known, power_law(d_known, a, b, c), '-', linewidth=3,
                      label=f'拟合 (R²={r2:.3f})', color='#dc2626')
            ax2.loglog(d_extrap, power_law(d_extrap, a, b, c), '--', linewidth=3,
                      label='外推预测', color='#dc2626', alpha=0.6)
            
            hestness_loss = power_law(d_range, 180, 0.095, 1.85)
            ax2.loglog(d_range, hestness_loss, ':', linewidth=2,
                      label='Hestness (2018)', color='#059669', alpha=0.7)
            
            important_data = [
                (300e9, "GPT-3"),
                (1.4e12, "LLaMA"),
                (2e12, "Llama 2"),
                (15e12, "Llama 3 (估计)"),
            ]
            
            for d, name in important_data:
                if d >= n_tokens.min():
                    loss_pred = power_law(d, a, b, c)
                    ax2.plot(d, loss_pred, 'v', markersize=10, color='#7c3aed')
                    ax2.text(d, loss_pred * 0.85, name, fontsize=10, ha='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('数据量 (D, tokens)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss (L)', fontsize=14, fontweight='bold')
    ax2.set_title('数据 Scaling Law 及外推预测', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=11, loc='upper right')
    
    # ========== 子图 3: 训练曲线 (参数维度) ==========
    ax3 = axes[1, 0]
    
    if param_metadata:
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(param_metadata)))
        for idx, (n_param, metadata) in enumerate(sorted(param_metadata.items())):
            losses = metadata.get('losses', [])
            if losses:
                steps = np.arange(len(losses))
                ax3.plot(steps, losses, label=f'{n_param/1e6:.0f}M params',
                        color=colors[idx], linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('训练步数', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax3.set_title('训练曲线对比 (不同参数量)', fontsize=15, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9, ncol=2, loc='upper right')
    
    # ========== 子图 4: 训练曲线 (数据维度) ==========
    ax4 = axes[1, 1]
    
    if data_metadata:
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(data_metadata)))
        for idx, (n_tokens, metadata) in enumerate(sorted(data_metadata.items())):
            losses = metadata.get('losses', [])
            if losses:
                steps = np.arange(len(losses))
                label_str = f'{n_tokens/1e9:.1f}B' if n_tokens >= 1e9 else f'{n_tokens/1e6:.0f}M'
                ax4.plot(steps, losses, label=f'{label_str} tokens',
                        color=colors[idx], linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('训练步数', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax4.set_title('训练曲线对比 (不同数据量)', fontsize=15, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9, ncol=2, loc='upper right')
    
    plt.suptitle('Scaling Law 完整分析 (实验 + 预测)', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = save_dir / 'scaling_laws_complete.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 完整图表已保存: {save_path}")
    plt.close()


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiments(config: ScalingConfig):
    """运行完整的 Scaling Law 实验"""
    
    print("=" * 80)
    print("🚀 增强版 Scaling Law 实验")
    print("=" * 80)
    print(f"训练步数: {config.max_steps}")
    print(f"Warmup 步数: {config.warmup_steps}")
    print(f"学习率: {config.learning_rate}")
    print("=" * 80)
    
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    
    results = {
        'param_scaling': {},
        'data_scaling': {},
        'param_metadata': {},  # 保存训练曲线等元数据
        'data_metadata': {},
        'config': config.__dict__
    }
    
    # ========== 实验 1: 参数 Scaling ==========
    print("\n" + "=" * 80)
    print("📊 实验 1: 参数 Scaling (固定数据量 100M)")
    print("=" * 80)
    
    fixed_tokens = 100e6
    
    for n_params in config.n_params_list:
        print(f"\n🔹 参数量: {n_params/1e6:.1f}M")
        
        model = create_model_from_params(n_params)
        actual_params = model.count_parameters()
        
        dataset = DummyTextDataset(fixed_tokens, seq_len=config.seq_len)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        losses, metadata = train_model(model, dataloader, device, 
                                      max_steps=config.max_steps, 
                                      lr=config.learning_rate,
                                      warmup_steps=config.warmup_steps)
        
        results['param_scaling'][actual_params] = metadata['final_loss']
        results['param_metadata'][actual_params] = metadata
        
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
    
    # ========== 实验 2: 数据 Scaling ==========
    print("\n" + "=" * 80)
    print("📊 实验 2: 数据 Scaling (固定参数量 50M)")
    print("=" * 80)
    
    fixed_params = 50e6
    
    for n_tokens in config.n_tokens_list:
        print(f"\n🔹 数据量: {n_tokens/1e6:.1f}M tokens")
        
        model = create_model_from_params(fixed_params)
        
        dataset = DummyTextDataset(n_tokens, seq_len=config.seq_len)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        losses, metadata = train_model(model, dataloader, device,
                                      max_steps=config.max_steps,
                                      lr=config.learning_rate,
                                      warmup_steps=config.warmup_steps)
        
        results['data_scaling'][n_tokens] = metadata['final_loss']
        results['data_metadata'][n_tokens] = metadata
        
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
    
    # 保存结果
    results_path = save_dir / 'results.json'
    
    # 转换为可序列化格式
    json_results = {
        'param_scaling': {str(k): v for k, v in results['param_scaling'].items()},
        'data_scaling': {str(k): v for k, v in results['data_scaling'].items()},
        'param_metadata': {
            str(k): {mk: mv for mk, mv in v.items() if mk != 'losses'}  # 排除 losses
            for k, v in results['param_metadata'].items()
        },
        'data_metadata': {
            str(k): {mk: mv for mk, mv in v.items() if mk != 'losses'}
            for k, v in results['data_metadata'].items()
        },
        'config': results['config']
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 结果已保存: {results_path}")
    
    # 生成可视化
    print("\n" + "=" * 80)
    print("📈 生成完整可视化图表")
    print("=" * 80)
    plot_scaling_curves(results, save_dir)
    
    print("\n" + "=" * 80)
    print("✅ 实验完成！")
    print("=" * 80)
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版 Scaling Law 实验')
    parser.add_argument('--mode', choices=['quick', 'standard', 'full'], 
                       default='quick', help='实验模式')
    parser.add_argument('--max-steps', type=int, default=None, 
                       help='每个实验的最大训练步数 (默认: quick=3000, standard=5000, full=8000)')
    args = parser.parse_args()
    
    # 配置
    if args.mode == 'quick':
        print("🚀 Quick mode: ~2-3 小时")
        max_steps = args.max_steps or 3000
        config = ScalingConfig(
            n_params_list=[5e6, 20e6, 50e6],
            n_tokens_list=[10e6, 50e6, 100e6],
            max_steps=max_steps,
            warmup_steps=max_steps // 10,
            save_dir='./scaling_results_quick_v2'
        )
    elif args.mode == 'standard':
        print("🚀 Standard mode: ~6-8 小时")
        max_steps = args.max_steps or 5000
        config = ScalingConfig(
            n_params_list=[5e6, 10e6, 20e6, 50e6, 100e6],
            n_tokens_list=[10e6, 50e6, 100e6, 200e6, 500e6],
            max_steps=max_steps,
            warmup_steps=max_steps // 10,
            save_dir='./scaling_results_standard_v2'
        )
    else:  # full
        print("🚀 Full mode: ~1.5-2 天")
        max_steps = args.max_steps or 8000
        config = ScalingConfig(
            n_params_list=[5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6],
            n_tokens_list=[10e6, 50e6, 100e6, 200e6, 500e6, 1e9],
            max_steps=max_steps,
            warmup_steps=max_steps // 10,
            save_dir='./scaling_results_full_v2'
        )
    
    print(f"训练配置: {max_steps} 步, warmup {config.warmup_steps} 步")
    
    # 运行实验
    results = run_experiments(config)
    
    print("\n📁 生成的文件:")
    print(f"  - {config.save_dir}/results.json")
    print(f"  - {config.save_dir}/scaling_laws_complete.png")
    print("\n提示: 可使用 compare_quick_vs_real.py 对比快速版和真实版结果")


if __name__ == '__main__':
    main()
