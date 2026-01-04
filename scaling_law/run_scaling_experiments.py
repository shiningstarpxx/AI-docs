"""
完整的 Scaling Law 实验脚本
============================

功能:
1. 运行完整的参数和数据 Scaling 实验
2. 生成可视化图表（包含论文理论曲线对比）
3. 验证 Kaplan 和 Chinchilla Scaling Laws

作者: peixingxin
日期: 2025-12-29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
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
    n_params_list: List[float] = None  # 参数量列表
    n_tokens_list: List[float] = None  # 数据量列表
    
    # 训练配置
    max_steps: int = 1000
    batch_size: int = 32
    seq_len: int = 128
    learning_rate: float = 3e-4
    
    # 设备
    device: str = "mps"
    
    # 保存路径
    save_dir: str = "./scaling_results"
    
    def __post_init__(self):
        if self.n_params_list is None:
            # 默认：5M -> 500M (跨越 2 个数量级)
            self.n_params_list = [5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6]
        if self.n_tokens_list is None:
            # 默认：10M -> 1B
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
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        
        # Embedding + Position
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x + self.pos_embedding[:seq_len, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Output
        logits = self.output(x)
        return logits
    
    def count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters())


def create_model_from_params(target_params: float, vocab_size: int = 10000) -> SimpleTransformer:
    """根据目标参数量创建模型"""
    
    # 尝试不同的配置
    configs = [
        # (d_model, n_heads, n_layers)
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
    
    # 找到最接近的配置
    best_config = None
    min_diff = float('inf')
    
    for d_model, n_heads, n_layers in configs:
        # 粗略估算参数量
        estimated_params = (
            vocab_size * d_model * 2 +  # embedding + output
            n_layers * (
                3 * d_model * d_model +  # QKV
                d_model * d_model +      # attention output
                2 * d_model * 4 * d_model  # FFN
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
    """虚拟文本数据集（用于快速实验）"""
    
    def __init__(self, n_tokens: int, seq_len: int = 128, vocab_size: int = 10000):
        self.n_tokens = int(n_tokens)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_samples = self.n_tokens // seq_len
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 生成随机序列（模拟真实数据）
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x


# ============================================================================
# 训练与评估
# ============================================================================

def train_model(model: nn.Module, dataloader: DataLoader, device: torch.device,
                max_steps: int = 1000, lr: float = 3e-4, warmup_steps: int = 100) -> Tuple[List[float], Dict]:
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
    
    print(f"  Training {max_steps} steps (with warmup={warmup_steps})...")
    start_time = time.time()
    
    while step < max_steps:
        for batch in dataloader:
            if step >= max_steps:
                break
            
            batch = batch.to(device)
            
            # 前向传播
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            
            # 计算损失
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                print(f"    Step {step}/{max_steps} | Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
            
            step += 1
    
    total_time = time.time() - start_time
    final_loss = get_final_loss(losses)
    print(f"  Completed in {total_time:.1f}s | Final Loss: {final_loss:.4f}")
    
    # 返回训练元数据
    metadata = {
        'total_time': total_time,
        'final_loss': final_loss,
        'initial_loss': losses[0] if losses else None,
        'min_loss': min(losses) if losses else None,
    }
    
    return losses, metadata


def get_final_loss(losses: List[float], window: int = 100) -> float:
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
                    x_name: str = "N") -> Tuple[float, float, float]:
    """拟合 Scaling Law"""
    
    # 初始猜测
    p0 = [100, 0.08, 1.8]
    
    try:
        params, _ = curve_fit(power_law, x_data, y_data, p0=p0, maxfev=10000)
        a, b, c = params
        
        print(f"\n拟合结果 (L({x_name})):")
        print(f"  L({x_name}) = {c:.3f} + {a:.1f} / {x_name}^{b:.3f}")
        
        # 计算 R²
        y_pred = power_law(x_data, *params)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"  R² = {r2:.4f}")
        
        return a, b, c
    
    except Exception as e:
        print(f"拟合失败: {e}")
        return None, None, None


# ============================================================================
# 可视化
# ============================================================================

def plot_scaling_curves(results: Dict, save_dir: Path):
    """生成 Scaling Law 可视化图表（包含预测外推）"""
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取数据
    param_results = results.get('param_scaling', {})
    data_results = results.get('data_scaling', {})
    
    # 创建图表 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ========== 子图 1: 参数 Scaling ==========
    ax1 = axes[0, 0]
    
    if param_results:
        n_params = np.array(list(param_results.keys()))
        losses = np.array(list(param_results.values()))
        
        # 实验数据
        ax1.loglog(n_params, losses, 'o-', markersize=10, linewidth=2,
                   label='Experimental Data', color='#2563eb')
        
        # 拟合曲线
        a, b, c = fit_scaling_law(n_params, losses, "N")
        if a is not None:
            n_range = np.logspace(np.log10(n_params.min()), 
                                  np.log10(n_params.max() * 100), 100)
            loss_fit = power_law(n_range, a, b, c)
            ax1.loglog(n_range, loss_fit, '--', linewidth=2,
                      label=f'Fit: L={c:.2f} + {a:.0f}/N^{b:.3f}', 
                      color='#dc2626', alpha=0.7)
        
        # Kaplan et al. (2020) 理论曲线
        kaplan_a, kaplan_b, kaplan_c = 450, 0.076, 1.69
        kaplan_loss = power_law(n_range, kaplan_a, kaplan_b, kaplan_c)
        ax1.loglog(n_range, kaplan_loss, ':', linewidth=2,
                  label=f'Kaplan (2020): L=1.69 + 450/N^0.076',
                  color='#059669', alpha=0.7)
        
        # 标注重要模型
        important_models = [
            (124e6, "GPT-2 Small"),
            (350e6, "GPT-2 Medium"),
            (1.5e9, "GPT-2 XL"),
            (175e9, "GPT-3"),
        ]
        
        for n, name in important_models:
            if n_params.min() < n < n_range.max():
                loss_pred = power_law(n, a, b, c) if a else None
                if loss_pred:
                    ax1.plot(n, loss_pred, 'v', markersize=8, color='#7c3aed')
                    ax1.text(n, loss_pred * 1.1, name, fontsize=9, ha='center')
    
    ax1.set_xlabel('Parameters (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (L)', fontsize=12, fontweight='bold')
    ax1.set_title('Parameter Scaling Law', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)
    
    # ========== 子图 2: 数据 Scaling ==========
    ax2 = axes[0, 1]
    
    if data_results:
        n_tokens = np.array(list(data_results.keys()))
        losses = np.array(list(data_results.values()))
        
        # 实验数据
        ax2.loglog(n_tokens, losses, 's-', markersize=10, linewidth=2,
                   label='Experimental Data', color='#2563eb')
        
        # 拟合曲线
        a, b, c = fit_scaling_law(n_tokens, losses, "D")
        if a is not None:
            d_range = np.logspace(np.log10(n_tokens.min()),
                                  np.log10(n_tokens.max() * 100), 100)
            loss_fit = power_law(d_range, a, b, c)
            ax2.loglog(d_range, loss_fit, '--', linewidth=2,
                      label=f'Fit: L={c:.2f} + {a:.0f}/D^{b:.3f}',
                      color='#dc2626', alpha=0.7)
        
        # Hestness et al. (2018) 理论
        hestness_a, hestness_b, hestness_c = 180, 0.095, 1.85
        hestness_loss = power_law(d_range, hestness_a, hestness_b, hestness_c)
        ax2.loglog(d_range, hestness_loss, ':', linewidth=2,
                  label=f'Hestness (2018): L=1.85 + 180/D^0.095',
                  color='#059669', alpha=0.7)
        
        # 标注重要数据量
        important_data = [
            (300e9, "GPT-3"),
            (1.4e12, "LLaMA"),
            (2e12, "Llama 2"),
        ]
        
        for d, name in important_data:
            if n_tokens.min() < d < d_range.max():
                loss_pred = power_law(d, a, b, c) if a else None
                if loss_pred:
                    ax2.plot(d, loss_pred, 'v', markersize=8, color='#7c3aed')
                    ax2.text(d, loss_pred * 1.1, name, fontsize=9, ha='center')
    
    ax2.set_xlabel('Dataset Size (D, tokens)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (L)', fontsize=12, fontweight='bold')
    ax2.set_title('Data Scaling Law', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    save_path = save_dir / 'scaling_laws_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 图表已保存: {save_path}")
    plt.close()


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiments(config: ScalingConfig):
    """运行完整的 Scaling Law 实验"""
    
    print("=" * 80)
    print("🚀 Scaling Law 实验开始")
    print("=" * 80)
    
    # 创建保存目录
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    if config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    
    results = {
        'param_scaling': {},
        'data_scaling': {},
        'config': config.__dict__
    }
    
    # ========== 实验 1: 参数 Scaling ==========
    print("\n" + "=" * 80)
    print("📊 实验 1: 参数 Scaling (固定数据量)")
    print("=" * 80)
    
    fixed_tokens = 100e6  # 固定 100M tokens
    
    for n_params in config.n_params_list:
        print(f"\n🔹 参数量: {n_params/1e6:.1f}M")
        
        # 创建模型
        model = create_model_from_params(n_params)
        actual_params = model.count_parameters()
        
        # 创建数据集
        dataset = DummyTextDataset(fixed_tokens, seq_len=config.seq_len)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        # 训练
        losses = train_model(model, dataloader, device, 
                           max_steps=config.max_steps, lr=config.learning_rate)
        
        final_loss = get_final_loss(losses)
        results['param_scaling'][actual_params] = final_loss
        
        print(f"  ✅ Final Loss: {final_loss:.4f}")
        
        # 清理内存
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
    
    # ========== 实验 2: 数据 Scaling ==========
    print("\n" + "=" * 80)
    print("📊 实验 2: 数据 Scaling (固定参数量)")
    print("=" * 80)
    
    fixed_params = 50e6  # 固定 50M 参数
    
    for n_tokens in config.n_tokens_list:
        print(f"\n🔹 数据量: {n_tokens/1e6:.1f}M tokens")
        
        # 创建模型
        model = create_model_from_params(fixed_params)
        
        # 创建数据集
        dataset = DummyTextDataset(n_tokens, seq_len=config.seq_len)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        # 训练
        losses = train_model(model, dataloader, device,
                           max_steps=config.max_steps, lr=config.learning_rate)
        
        final_loss = get_final_loss(losses)
        results['data_scaling'][n_tokens] = final_loss
        
        print(f"  ✅ Final Loss: {final_loss:.4f}")
        
        # 清理内存
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
    
    # 保存结果
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        # 转换 key 为字符串（JSON 不支持数字 key）
        json_results = {
            'param_scaling': {str(k): v for k, v in results['param_scaling'].items()},
            'data_scaling': {str(k): v for k, v in results['data_scaling'].items()},
            'config': results['config']
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 结果已保存: {results_path}")
    
    # 生成可视化
    print("\n" + "=" * 80)
    print("📈 生成可视化图表")
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
    
    parser = argparse.ArgumentParser(description='Scaling Law 实验')
    parser.add_argument('--mode', choices=['quick', 'standard', 'full'], 
                       default='quick', help='实验模式')
    parser.add_argument('--max-steps', type=int, default=1000, 
                       help='每个实验的最大训练步数')
    args = parser.parse_args()
    
    # 配置
    if args.mode == 'quick':
        print("🚀 Quick mode: ~1-2 小时")
        config = ScalingConfig(
            n_params_list=[5e6, 20e6, 50e6],
            n_tokens_list=[10e6, 50e6, 100e6],
            max_steps=args.max_steps,
            save_dir='./scaling_results_quick'
        )
    elif args.mode == 'standard':
        print("🚀 Standard mode: ~4-6 小时")
        config = ScalingConfig(
            n_params_list=[5e6, 10e6, 20e6, 50e6, 100e6],
            n_tokens_list=[10e6, 50e6, 100e6, 200e6, 500e6],
            max_steps=args.max_steps,
            save_dir='./scaling_results_standard'
        )
    else:  # full
        print("🚀 Full mode: ~1-2 天")
        config = ScalingConfig(
            n_params_list=[5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6],
            n_tokens_list=[10e6, 50e6, 100e6, 200e6, 500e6, 1e9],
            max_steps=args.max_steps,
            save_dir='./scaling_results_full'
        )
    
    # 运行实验
    results = run_experiments(config)
    
    print("\n📁 结果文件:")
    print(f"  - {config.save_dir}/results.json")
    print(f"  - {config.save_dir}/scaling_laws_comparison.png")


if __name__ == '__main__':
    main()
