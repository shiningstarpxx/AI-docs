"""
快速版 vs 真实版对比分析
========================

对比模拟数据和实际训练结果，验证：
1. 模拟数据的准确性
2. 早停外推的可靠性
3. Scaling Law 的预测能力

作者: peixingxin
日期: 2025-12-29
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# ============================================================================
# 加载数据
# ============================================================================

def load_quick_results():
    """加载快速版结果（模拟数据）"""
    path = Path('scaling_demo/results.json')
    if not path.exists():
        print("❌ 快速版结果不存在，请先运行: python quick_scaling_demo.py")
        return None
    
    with open(path) as f:
        data = json.load(f)
    
    # 转换 key 为 float
    param_scaling = {float(k): v for k, v in data['param_scaling'].items()}
    data_scaling = {float(k): v for k, v in data['data_scaling'].items()}
    
    return {
        'param_scaling': param_scaling,
        'data_scaling': data_scaling,
        'source': 'quick (模拟)'
    }


def load_real_results(mode='standard'):
    """加载真实版结果（实际训练）"""
    # 尝试不同的结果文件
    possible_paths = [
        f'./scaling_results_{mode}/results.json',
        './scaling_results_quick/results.json',
        './scaling_results_standard/results.json',
    ]
    
    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✅ 找到真实实验结果: {path}")
            with open(path) as f:
                data = json.load(f)
            
            # 转换 key 为 float
            param_scaling = {float(k): v for k, v in data['param_scaling'].items()}
            data_scaling = {float(k): v for k, v in data['data_scaling'].items()}
            
            return {
                'param_scaling': param_scaling,
                'data_scaling': data_scaling,
                'source': f'real ({mode})'
            }
    
    print("❌ 真实实验结果不存在")
    print("请先运行: python run_scaling_experiments.py --mode standard")
    return None


# ============================================================================
# 对比分析
# ============================================================================

def compare_results(quick_data, real_data):
    """对比快速版和真实版结果"""
    
    print("\n" + "=" * 80)
    print("📊 快速版 vs 真实版对比分析")
    print("=" * 80)
    
    # 参数 Scaling 对比
    print("\n【参数 Scaling 对比】")
    print("-" * 80)
    print(f"{'模型规模':>12s} | {'快速版 Loss':>12s} | {'真实版 Loss':>12s} | {'误差':>10s} | {'相对误差':>10s}")
    print("-" * 80)
    
    param_errors = []
    for n in sorted(quick_data['param_scaling'].keys()):
        if n in real_data['param_scaling']:
            quick_loss = quick_data['param_scaling'][n]
            real_loss = real_data['param_scaling'][n]
            error = abs(quick_loss - real_loss)
            rel_error = error / real_loss * 100
            param_errors.append(rel_error)
            
            print(f"{n/1e6:10.1f}M | {quick_loss:12.4f} | {real_loss:12.4f} | "
                  f"{error:10.4f} | {rel_error:9.2f}%")
    
    avg_param_error = np.mean(param_errors) if param_errors else 0
    print("-" * 80)
    print(f"{'平均相对误差':>47s} | {avg_param_error:9.2f}%")
    
    # 数据 Scaling 对比
    print("\n【数据 Scaling 对比】")
    print("-" * 80)
    print(f"{'数据规模':>12s} | {'快速版 Loss':>12s} | {'真实版 Loss':>12s} | {'误差':>10s} | {'相对误差':>10s}")
    print("-" * 80)
    
    data_errors = []
    for d in sorted(quick_data['data_scaling'].keys()):
        if d in real_data['data_scaling']:
            quick_loss = quick_data['data_scaling'][d]
            real_loss = real_data['data_scaling'][d]
            error = abs(quick_loss - real_loss)
            rel_error = error / real_loss * 100
            data_errors.append(rel_error)
            
            if d >= 1e9:
                d_str = f"{d/1e9:.1f}B"
            else:
                d_str = f"{d/1e6:.0f}M"
            
            print(f"{d_str:>12s} | {quick_loss:12.4f} | {real_loss:12.4f} | "
                  f"{error:10.4f} | {rel_error:9.2f}%")
    
    avg_data_error = np.mean(data_errors) if data_errors else 0
    print("-" * 80)
    print(f"{'平均相对误差':>47s} | {avg_data_error:9.2f}%")
    
    # 总结
    print("\n" + "=" * 80)
    print("📈 总结")
    print("=" * 80)
    print(f"参数 Scaling 平均误差: {avg_param_error:.2f}%")
    print(f"数据 Scaling 平均误差: {avg_data_error:.2f}%")
    print(f"总体平均误差: {(avg_param_error + avg_data_error) / 2:.2f}%")
    
    if (avg_param_error + avg_data_error) / 2 < 5:
        print("\n✅ 快速版预测非常准确！误差 < 5%")
    elif (avg_param_error + avg_data_error) / 2 < 10:
        print("\n✅ 快速版预测较准确，误差 < 10%")
    else:
        print("\n⚠️  快速版预测存在一定偏差，建议调整模拟参数")
    
    return param_errors, data_errors


# ============================================================================
# 可视化对比
# ============================================================================

def plot_comparison(quick_data, real_data, save_dir='./comparison_results'):
    """生成对比图表"""
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ========== 图 1: 参数 Scaling 对比 ==========
    ax1 = axes[0, 0]
    
    # 快速版数据
    quick_params = np.array(sorted(quick_data['param_scaling'].keys()))
    quick_param_losses = np.array([quick_data['param_scaling'][n] for n in quick_params])
    
    # 真实版数据
    real_params = np.array(sorted(real_data['param_scaling'].keys()))
    real_param_losses = np.array([real_data['param_scaling'][n] for n in real_params])
    
    ax1.loglog(quick_params, quick_param_losses, 'o-', markersize=10, linewidth=2,
              label='快速版 (模拟)', color='#3b82f6', alpha=0.7)
    ax1.loglog(real_params, real_param_losses, 's--', markersize=10, linewidth=2,
              label='真实版 (训练)', color='#ef4444', alpha=0.7)
    
    ax1.set_xlabel('Parameters (N)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss (L)', fontsize=13, fontweight='bold')
    ax1.set_title('参数 Scaling: 快速版 vs 真实版', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=12)
    
    # ========== 图 2: 数据 Scaling 对比 ==========
    ax2 = axes[0, 1]
    
    # 快速版数据
    quick_tokens = np.array(sorted(quick_data['data_scaling'].keys()))
    quick_data_losses = np.array([quick_data['data_scaling'][d] for d in quick_tokens])
    
    # 真实版数据
    real_tokens = np.array(sorted(real_data['data_scaling'].keys()))
    real_data_losses = np.array([real_data['data_scaling'][d] for d in real_tokens])
    
    ax2.loglog(quick_tokens, quick_data_losses, 'o-', markersize=10, linewidth=2,
              label='快速版 (模拟)', color='#3b82f6', alpha=0.7)
    ax2.loglog(real_tokens, real_data_losses, 's--', markersize=10, linewidth=2,
              label='真实版 (训练)', color='#ef4444', alpha=0.7)
    
    ax2.set_xlabel('Dataset Size (D, tokens)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Loss (L)', fontsize=13, fontweight='bold')
    ax2.set_title('数据 Scaling: 快速版 vs 真实版', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=12)
    
    # ========== 图 3: 参数 Scaling 误差分析 ==========
    ax3 = axes[1, 0]
    
    # 计算重叠的点
    common_params = sorted(set(quick_params) & set(real_params))
    if common_params:
        errors = []
        for n in common_params:
            quick_loss = quick_data['param_scaling'][n]
            real_loss = real_data['param_scaling'][n]
            rel_error = abs(quick_loss - real_loss) / real_loss * 100
            errors.append(rel_error)
        
        ax3.semilogx(common_params, errors, 'o-', markersize=10, linewidth=2,
                    color='#10b981', markeredgewidth=2, markeredgecolor='white')
        ax3.axhline(y=5, color='#f59e0b', linestyle='--', linewidth=2, label='5% 误差线')
        ax3.axhline(y=10, color='#ef4444', linestyle='--', linewidth=2, label='10% 误差线')
        
        ax3.set_xlabel('Parameters (N)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('相对误差 (%)', fontsize=13, fontweight='bold')
        ax3.set_title('参数 Scaling 误差分析', fontsize=15, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=11)
    
    # ========== 图 4: 数据 Scaling 误差分析 ==========
    ax4 = axes[1, 1]
    
    # 计算重叠的点
    common_tokens = sorted(set(quick_tokens) & set(real_tokens))
    if common_tokens:
        errors = []
        for d in common_tokens:
            quick_loss = quick_data['data_scaling'][d]
            real_loss = real_data['data_scaling'][d]
            rel_error = abs(quick_loss - real_loss) / real_loss * 100
            errors.append(rel_error)
        
        ax4.semilogx(common_tokens, errors, 's-', markersize=10, linewidth=2,
                    color='#10b981', markeredgewidth=2, markeredgecolor='white')
        ax4.axhline(y=5, color='#f59e0b', linestyle='--', linewidth=2, label='5% 误差线')
        ax4.axhline(y=10, color='#ef4444', linestyle='--', linewidth=2, label='10% 误差线')
        
        ax4.set_xlabel('Dataset Size (D, tokens)', fontsize=13, fontweight='bold')
        ax4.set_ylabel('相对误差 (%)', fontsize=13, fontweight='bold')
        ax4.set_title('数据 Scaling 误差分析', fontsize=15, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=11)
    
    plt.suptitle('快速版 vs 真实版 完整对比', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'quick_vs_real_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图表已保存: {save_path}")
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    print("=" * 80)
    print("🔬 快速版 vs 真实版对比分析")
    print("=" * 80)
    
    # 加载数据
    print("\n📂 加载实验数据...")
    quick_data = load_quick_results()
    
    if quick_data is None:
        return
    
    real_data = load_real_results('standard')
    
    if real_data is None:
        # 尝试其他模式
        real_data = load_real_results('quick')
        if real_data is None:
            print("\n❌ 无法找到真实实验结果")
            print("\n请先运行真实实验：")
            print("  python run_scaling_experiments.py --mode quick")
            print("或者：")
            print("  python run_scaling_experiments.py --mode standard")
            return
    
    print(f"\n✅ 数据加载完成")
    print(f"  快速版: {len(quick_data['param_scaling'])} 个参数点, "
          f"{len(quick_data['data_scaling'])} 个数据点")
    print(f"  真实版: {len(real_data['param_scaling'])} 个参数点, "
          f"{len(real_data['data_scaling'])} 个数据点")
    
    # 对比分析
    param_errors, data_errors = compare_results(quick_data, real_data)
    
    # 可视化
    print("\n" + "=" * 80)
    print("📈 生成对比图表")
    print("=" * 80)
    plot_comparison(quick_data, real_data)
    
    # 保存对比报告
    save_dir = Path('./comparison_results')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'quick_source': quick_data['source'],
        'real_source': real_data['source'],
        'param_scaling': {
            'avg_error': float(np.mean(param_errors)) if param_errors else 0,
            'max_error': float(np.max(param_errors)) if param_errors else 0,
            'min_error': float(np.min(param_errors)) if param_errors else 0,
        },
        'data_scaling': {
            'avg_error': float(np.mean(data_errors)) if data_errors else 0,
            'max_error': float(np.max(data_errors)) if data_errors else 0,
            'min_error': float(np.min(data_errors)) if data_errors else 0,
        },
        'overall': {
            'avg_error': float((np.mean(param_errors) + np.mean(data_errors)) / 2) 
                        if param_errors and data_errors else 0
        }
    }
    
    report_path = save_dir / 'comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 对比报告已保存: {report_path}")
    
    print("\n" + "=" * 80)
    print("✅ 对比分析完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print(f"  - {save_dir}/quick_vs_real_comparison.png")
    print(f"  - {save_dir}/comparison_report.json")


if __name__ == '__main__':
    main()
