"""
快速 Scaling Law 演示
====================
使用模拟数据快速生成 Scaling Law 可视化

作者: peixingxin
日期: 2025-12-29
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json

# ============================================================================
# 理论曲线（基于论文）
# ============================================================================

def kaplan_loss(N):
    """Kaplan et al. (2020) 参数 Scaling Law"""
    return 1.69 + 450 / (N ** 0.076)

def hestness_loss(D):
    """Hestness et al. (2018) 数据 Scaling Law"""
    return 1.85 + 180 / (D ** 0.095)

# ============================================================================
# 模拟实验数据（添加噪声）
# ============================================================================

def generate_synthetic_data():
    """生成模拟的实验数据"""
    
    # 参数 Scaling 实验数据
    n_params = np.array([5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6])
    
    # 基于理论曲线 + 随机噪声
    param_losses = kaplan_loss(n_params) + np.random.normal(0, 0.05, len(n_params))
    
    # 数据 Scaling 实验数据
    n_tokens = np.array([10e6, 50e6, 100e6, 200e6, 500e6, 1e9])
    data_losses = hestness_loss(n_tokens) + np.random.normal(0, 0.04, len(n_tokens))
    
    return {
        'param_scaling': dict(zip(n_params, param_losses)),
        'data_scaling': dict(zip(n_tokens, data_losses))
    }

# ============================================================================
# 可视化
# ============================================================================

def plot_scaling_laws_with_theory(results, save_dir='./scaling_demo'):
    """生成包含理论曲线对比的图表"""
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建图表
    fig = plt.figure(figsize=(18, 12))
    
    # ========== 图 1: 参数 Scaling ==========
    ax1 = plt.subplot(2, 2, 1)
    
    n_params = np.array(list(results['param_scaling'].keys()))
    param_losses = np.array(list(results['param_scaling'].values()))
    
    # 实验数据点
    ax1.loglog(n_params, param_losses, 'o', markersize=12, linewidth=3,
               label='Experimental Data', color='#2563eb', markeredgewidth=2, 
               markeredgecolor='white')
    
    # 理论曲线
    n_range = np.logspace(np.log10(1e6), np.log10(1e11), 100)
    kaplan_curve = kaplan_loss(n_range)
    ax1.loglog(n_range, kaplan_curve, '--', linewidth=3,
              label='Kaplan et al. (2020): L = 1.69 + 450/N^0.076',
              color='#dc2626', alpha=0.8)
    
    # 标注重要模型
    important_models = [
        (124e6, 2.69, "GPT-2\nSmall"),
        (355e6, 2.45, "GPT-2\nMedium"),
        (1.5e9, 2.15, "GPT-2\nXL"),
        (175e9, 1.85, "GPT-3"),
    ]
    
    for n, loss, name in important_models:
        if 1e6 < n < 1e11:
            ax1.plot(n, kaplan_loss(n), 'v', markersize=10, color='#7c3aed')
            ax1.annotate(name, xy=(n, kaplan_loss(n)), xytext=(n*1.5, kaplan_loss(n)*1.05),
                        fontsize=9, ha='left', bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='yellow', alpha=0.3))
    
    ax1.set_xlabel('Parameters (N)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss (L)', fontsize=13, fontweight='bold')
    ax1.set_title('Parameter Scaling Law', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_xlim(1e6, 1e11)
    ax1.set_ylim(1.6, 3.5)
    
    # ========== 图 2: 数据 Scaling ==========
    ax2 = plt.subplot(2, 2, 2)
    
    n_tokens = np.array(list(results['data_scaling'].keys()))
    data_losses = np.array(list(results['data_scaling'].values()))
    
    # 实验数据点
    ax2.loglog(n_tokens, data_losses, 's', markersize=12, linewidth=3,
               label='Experimental Data', color='#2563eb', markeredgewidth=2,
               markeredgecolor='white')
    
    # 理论曲线
    d_range = np.logspace(np.log10(1e6), np.log10(1e13), 100)
    hestness_curve = hestness_loss(d_range)
    ax2.loglog(d_range, hestness_curve, '--', linewidth=3,
              label='Hestness et al. (2018): L = 1.85 + 180/D^0.095',
              color='#059669', alpha=0.8)
    
    # 标注重要数据集
    important_data = [
        (40e9, "GPT-2\n40B"),
        (300e9, "GPT-3\n300B"),
        (1.4e12, "LLaMA\n1.4T"),
        (2e12, "Llama 2\n2T"),
    ]
    
    for d, name in important_data:
        if 1e6 < d < 1e13:
            ax2.plot(d, hestness_loss(d), 'v', markersize=10, color='#7c3aed')
            ax2.annotate(name, xy=(d, hestness_loss(d)), xytext=(d*1.5, hestness_loss(d)*1.03),
                        fontsize=9, ha='left', bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='lightgreen', alpha=0.3))
    
    ax2.set_xlabel('Dataset Size (D, tokens)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Loss (L)', fontsize=13, fontweight='bold')
    ax2.set_title('Data Scaling Law', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle='--')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_xlim(1e6, 1e13)
    ax2.set_ylim(1.7, 3.0)
    
    # ========== 图 3: 参数 Scaling (线性-对数) ==========
    ax3 = plt.subplot(2, 2, 3)
    
    log_n_params = np.log10(n_params)
    ax3.plot(log_n_params, param_losses, 'o-', markersize=10, linewidth=2,
            label='Experimental Data', color='#2563eb')
    
    log_n_range = np.log10(n_range)
    ax3.plot(log_n_range, kaplan_curve, '--', linewidth=2,
            label='Kaplan Theory', color='#dc2626', alpha=0.7)
    
    ax3.set_xlabel('log10(Parameters)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Loss (L)', fontsize=13, fontweight='bold')
    ax3.set_title('Parameter Scaling (Linear-Log)', fontsize=15, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    # ========== 图 4: 数据 Scaling (线性-对数) ==========
    ax4 = plt.subplot(2, 2, 4)
    
    log_n_tokens = np.log10(n_tokens)
    ax4.plot(log_n_tokens, data_losses, 's-', markersize=10, linewidth=2,
            label='Experimental Data', color='#2563eb')
    
    log_d_range = np.log10(d_range)
    ax4.plot(log_d_range, hestness_curve, '--', linewidth=2,
            label='Hestness Theory', color='#059669', alpha=0.7)
    
    ax4.set_xlabel('log10(Dataset Size, tokens)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Loss (L)', fontsize=13, fontweight='bold')
    ax4.set_title('Data Scaling (Linear-Log)', fontsize=15, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)
    
    plt.suptitle('Scaling Laws: Experimental Results vs Theory', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'scaling_laws_with_theory.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {save_path}")
    plt.close()
    
    # ========== 图 5: Chinchilla 最优配置分析 ==========
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # 计算预算曲线
    compute_budgets = [1e19, 1e20, 1e21, 1e22, 1e23]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    
    for C, color in zip(compute_budgets, colors):
        # Chinchilla 最优配置: N ≈ D/20, C = 6*N*D
        # 因此: N_opt = (C/120)^0.5, D_opt = 20*N_opt
        
        N_opt = (C / 120) ** 0.5
        D_opt = 20 * N_opt
        
        # 绘制等计算量曲线
        N_range_const_C = np.logspace(np.log10(N_opt/100), np.log10(N_opt*100), 100)
        D_range_const_C = C / (6 * N_range_const_C)
        
        ax.loglog(N_range_const_C, D_range_const_C, '-', linewidth=2, 
                 color=color, alpha=0.6, label=f'C = {C:.0e} FLOPs')
        
        # 标记最优点
        ax.plot(N_opt, D_opt, 'o', markersize=12, color=color, 
               markeredgewidth=2, markeredgecolor='white')
        ax.text(N_opt*1.5, D_opt, f'{N_opt/1e9:.0f}B', fontsize=10, color=color, 
               fontweight='bold')
    
    # Chinchilla 最优线: D = 20*N
    N_line = np.logspace(6, 12, 100)
    D_line = 20 * N_line
    ax.loglog(N_line, D_line, 'k--', linewidth=3, 
             label='Chinchilla Optimal: D = 20×N')
    
    # 标注实际模型
    actual_models = [
        (175e9, 300e9, "GPT-3\n(欠训练)", '#dc2626'),
        (70e9, 1.4e12, "Chinchilla\n(最优)", '#059669'),
        (280e9, 300e9, "Gopher\n(欠训练)", '#dc2626'),
    ]
    
    for N, D, name, color in actual_models:
        ax.plot(N, D, 'D', markersize=15, color=color, markeredgewidth=2, 
               markeredgecolor='white')
        ax.annotate(name, xy=(N, D), xytext=(N*0.3, D*2),
                   fontsize=11, ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.2),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    ax.set_xlabel('Parameters (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dataset Size (D, tokens)', fontsize=14, fontweight='bold')
    ax.set_title('Chinchilla Optimal Scaling: Compute-Optimal Training',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(1e6, 1e12)
    ax.set_ylim(1e8, 1e14)
    
    plt.tight_layout()
    save_path2 = Path(save_dir) / 'chinchilla_optimal_scaling.png'
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {save_path2}")
    plt.close()
    
    print(f"\n📊 所有图表已生成在: {save_dir}/")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    print("=" * 80)
    print("🎨 Scaling Law 可视化演示")
    print("=" * 80)
    
    # 生成模拟数据
    print("\n📊 生成模拟实验数据...")
    results = generate_synthetic_data()
    
    # 输出数据
    print("\n参数 Scaling 数据:")
    for n, loss in results['param_scaling'].items():
        print(f"  {n/1e6:6.1f}M params → Loss: {loss:.4f}")
    
    print("\n数据 Scaling 数据:")
    for d, loss in results['data_scaling'].items():
        print(f"  {d/1e6:8.1f}M tokens → Loss: {loss:.4f}")
    
    # 生成可视化
    print("\n" + "=" * 80)
    print("📈 生成可视化图表（包含理论曲线对比）")
    print("=" * 80)
    plot_scaling_laws_with_theory(results)
    
    # 保存数据
    save_dir = Path('./scaling_demo')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'results.json', 'w') as f:
        json_results = {
            'param_scaling': {str(k): v for k, v in results['param_scaling'].items()},
            'data_scaling': {str(k): v for k, v in results['data_scaling'].items()}
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 数据已保存: {save_dir}/results.json")
    
    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print(f"  1. {save_dir}/scaling_laws_with_theory.png")
    print(f"  2. {save_dir}/chinchilla_optimal_scaling.png")
    print(f"  3. {save_dir}/results.json")

if __name__ == '__main__':
    main()
