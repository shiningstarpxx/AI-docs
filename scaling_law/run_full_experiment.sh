#!/bin/bash

# Scaling Law 完整实验脚本
# =======================
# 包含两个版本：
# 1. 快速版：模拟数据 + 理论验证（1分钟）
# 2. 真实版：实际训练 + 对比验证（2-6小时）

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "🚀 Scaling Law 双版本实验"
echo "=============================================="
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行: ./quickstart.sh"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 询问运行模式
echo "请选择运行模式："
echo "  1) 快速版 - 模拟数据验证（~1分钟）"
echo "  2) 真实版 - 实际训练验证（~2-6小时）"
echo "  3) 完整版 - 先快速后真实（完整对比）"
echo ""
read -p "请输入选择 (1/2/3): " mode

case $mode in
    1)
        echo ""
        echo "=============================================="
        echo "📊 运行快速版（模拟数据）"
        echo "=============================================="
        python quick_scaling_demo.py
        echo ""
        echo "✅ 快速版完成！"
        echo ""
        echo "📁 生成的文件："
        echo "  - scaling_demo/scaling_laws_with_theory.png"
        echo "  - scaling_demo/chinchilla_optimal_scaling.png"
        echo "  - scaling_demo/results.json"
        ;;
    
    2)
        echo ""
        echo "=============================================="
        echo "🔬 运行真实版（实际训练）"
        echo "=============================================="
        echo ""
        echo "选择训练模式："
        echo "  a) quick    - 快速训练（~1-2小时）"
        echo "  b) standard - 标准训练（~4-6小时）"
        echo "  c) full     - 完整训练（~1-2天）"
        echo ""
        read -p "请输入选择 (a/b/c): " train_mode
        
        case $train_mode in
            a)
                echo ""
                echo "🚀 启动快速训练模式..."
                nohup python run_scaling_experiments.py --mode quick --max-steps 500 \
                    > real_experiment_quick.log 2>&1 &
                pid=$!
                echo "✅ 训练已在后台启动 (PID: $pid)"
                echo ""
                echo "📝 监控命令："
                echo "  tail -f real_experiment_quick.log"
                echo ""
                echo "⏱️  预计完成时间: 1-2 小时"
                ;;
            b)
                echo ""
                echo "🚀 启动标准训练模式..."
                nohup python run_scaling_experiments.py --mode standard --max-steps 1000 \
                    > real_experiment_standard.log 2>&1 &
                pid=$!
                echo "✅ 训练已在后台启动 (PID: $pid)"
                echo ""
                echo "📝 监控命令："
                echo "  tail -f real_experiment_standard.log"
                echo ""
                echo "⏱️  预计完成时间: 4-6 小时"
                ;;
            c)
                echo ""
                echo "🚀 启动完整训练模式..."
                nohup python run_scaling_experiments.py --mode full --max-steps 2000 \
                    > real_experiment_full.log 2>&1 &
                pid=$!
                echo "✅ 训练已在后台启动 (PID: $pid)"
                echo ""
                echo "📝 监控命令："
                echo "  tail -f real_experiment_full.log"
                echo ""
                echo "⏱️  预计完成时间: 1-2 天"
                ;;
            *)
                echo "❌ 无效选择"
                exit 1
                ;;
        esac
        ;;
    
    3)
        echo ""
        echo "=============================================="
        echo "🎯 完整版：快速 + 真实对比"
        echo "=============================================="
        
        # 第一步：快速版
        echo ""
        echo "第 1 步：运行快速版（模拟数据）"
        echo "----------------------------------------"
        python quick_scaling_demo.py
        echo "✅ 快速版完成"
        
        # 第二步：真实版
        echo ""
        echo "第 2 步：运行真实版（实际训练）"
        echo "----------------------------------------"
        echo "启动标准模式训练..."
        nohup python run_scaling_experiments.py --mode standard --max-steps 1000 \
            > real_experiment_comparison.log 2>&1 &
        pid=$!
        
        echo "✅ 真实训练已在后台启动 (PID: $pid)"
        echo ""
        echo "📝 监控命令："
        echo "  tail -f real_experiment_comparison.log"
        echo ""
        
        # 第三步：等待完成后生成对比
        echo "第 3 步：训练完成后运行对比分析"
        echo "----------------------------------------"
        echo "训练完成后，请运行："
        echo "  python compare_quick_vs_real.py"
        echo ""
        echo "⏱️  预计总时间: 4-6 小时"
        ;;
    
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "✅ 脚本执行完成"
echo "=============================================="
