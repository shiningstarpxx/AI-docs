#!/bin/bash
# Scaling Law 实验启动脚本 V2.0
# 一键运行增强版实验

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 标题
echo ""
echo "========================================================================"
echo "  🚀 Scaling Law 实验启动器 V2.0"
echo "========================================================================"
echo ""

# 检查Python环境
print_info "检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 未找到，请先安装 Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python 版本: $PYTHON_VERSION"

# 检查虚拟环境
if [ -d "venv" ]; then
    print_info "找到虚拟环境，激活中..."
    source venv/bin/activate
    print_success "虚拟环境已激活"
else
    print_warning "虚拟环境不存在"
    read -p "是否创建虚拟环境? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "创建虚拟环境..."
        python3 -m venv venv
        source venv/bin/activate
        print_info "安装依赖..."
        pip install --upgrade pip
        pip install torch torchvision torchaudio numpy scipy matplotlib psutil
        print_success "虚拟环境创建完成"
    fi
fi

# 检查MPS
print_info "检查 MPS (Apple Silicon GPU)..."
MPS_AVAILABLE=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null)
if [ "$MPS_AVAILABLE" = "True" ]; then
    print_success "MPS 可用 ✅"
else
    print_warning "MPS 不可用，将使用 CPU"
fi

echo ""
echo "========================================================================"
echo "  📋 选择实验模式"
echo "========================================================================"
echo ""
echo "  1) 快速演示 (1 分钟) - 模拟数据，理论验证"
echo "  2) Quick V2 (2-3 小时) - 3000 步真实训练 [推荐]"
echo "  3) Standard V2 (6-8 小时) - 5000 步精确训练"
echo "  4) Full V2 (1.5-2 天) - 8000 步完整实验"
echo "  5) 完整流程 (快速演示 + 真实训练 + 对比分析)"
echo "  6) 对比分析 (需要先运行快速演示和真实训练)"
echo "  7) 查看现有结果"
echo "  8) 退出"
echo ""
read -p "请选择 (1-8): " -n 1 -r choice
echo ""

case $choice in
    1)
        print_info "运行快速演示..."
        python3 quick_scaling_demo.py
        print_success "快速演示完成!"
        print_info "查看结果:"
        echo "  - scaling_demo/scaling_laws_with_theory.png"
        echo "  - scaling_demo/chinchilla_optimal_scaling.png"
        ;;
        
    2)
        print_info "运行 Quick V2 模式 (3000 步)..."
        print_warning "预计耗时: 2-3 小时"
        read -p "是否在后台运行? (y/n): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            LOG_FILE="experiment_v2_quick_$(date +%Y%m%d_%H%M%S).log"
            nohup python3 run_scaling_experiments_enhanced.py --mode quick > "$LOG_FILE" 2>&1 &
            PID=$!
            print_success "实验已在后台启动 (PID: $PID)"
            print_info "监控进度: tail -f $LOG_FILE"
            print_info "查看进程: ps aux | grep $PID"
        else
            python3 run_scaling_experiments_enhanced.py --mode quick
            print_success "实验完成!"
        fi
        
        print_info "结果将保存到: scaling_results_quick_v2/"
        ;;
        
    3)
        print_info "运行 Standard V2 模式 (5000 步)..."
        print_warning "预计耗时: 6-8 小时"
        LOG_FILE="experiment_v2_standard_$(date +%Y%m%d_%H%M%S).log"
        nohup python3 run_scaling_experiments_enhanced.py --mode standard > "$LOG_FILE" 2>&1 &
        PID=$!
        print_success "实验已在后台启动 (PID: $PID)"
        print_info "监控进度: tail -f $LOG_FILE"
        ;;
        
    4)
        print_info "运行 Full V2 模式 (8000 步)..."
        print_warning "预计耗时: 1.5-2 天"
        LOG_FILE="experiment_v2_full_$(date +%Y%m%d_%H%M%S).log"
        nohup python3 run_scaling_experiments_enhanced.py --mode full > "$LOG_FILE" 2>&1 &
        PID=$!
        print_success "实验已在后台启动 (PID: $PID)"
        print_info "监控进度: tail -f $LOG_FILE"
        ;;
        
    5)
        print_info "运行完整流程..."
        
        print_info "Step 1/3: 快速演示"
        python3 quick_scaling_demo.py
        print_success "快速演示完成!"
        
        print_info "Step 2/3: 真实训练 (Quick V2)"
        LOG_FILE="experiment_v2_quick_$(date +%Y%m%d_%H%M%S).log"
        python3 run_scaling_experiments_enhanced.py --mode quick 2>&1 | tee "$LOG_FILE"
        print_success "真实训练完成!"
        
        print_info "Step 3/3: 对比分析"
        python3 compare_quick_vs_real.py
        print_success "对比分析完成!"
        
        print_success "完整流程执行完毕!"
        print_info "查看结果:"
        echo "  - scaling_demo/ (快速演示)"
        echo "  - scaling_results_quick_v2/ (真实训练)"
        echo "  - comparison_results/ (对比分析)"
        ;;
        
    6)
        print_info "运行对比分析..."
        
        if [ ! -f "scaling_demo/results.json" ]; then
            print_error "找不到快速演示结果，请先运行: 选项 1"
            exit 1
        fi
        
        if [ ! -f "scaling_results_quick_v2/results.json" ]; then
            print_error "找不到真实训练结果，请先运行: 选项 2"
            exit 1
        fi
        
        python3 compare_quick_vs_real.py
        print_success "对比分析完成!"
        print_info "查看结果: comparison_results/"
        ;;
        
    7)
        print_info "查看现有结果..."
        echo ""
        
        if [ -d "scaling_demo" ]; then
            echo "📊 快速演示结果:"
            ls -lh scaling_demo/*.png 2>/dev/null || echo "  (无图表)"
        fi
        
        if [ -d "scaling_results_quick_v2" ]; then
            echo ""
            echo "🔬 Quick V2 结果:"
            ls -lh scaling_results_quick_v2/*.png 2>/dev/null || echo "  (无图表)"
        fi
        
        if [ -d "scaling_results_quick" ]; then
            echo ""
            echo "🔬 Quick V1 结果 (旧版):"
            ls -lh scaling_results_quick/*.png 2>/dev/null || echo "  (无图表)"
        fi
        
        if [ -d "comparison_results" ]; then
            echo ""
            echo "📈 对比分析结果:"
            ls -lh comparison_results/*.png 2>/dev/null || echo "  (无图表)"
        fi
        
        echo ""
        print_info "打开图表 (Mac):"
        echo "  open scaling_demo/scaling_laws_with_theory.png"
        echo "  open scaling_results_quick_v2/scaling_laws_complete.png"
        ;;
        
    8)
        print_info "退出"
        exit 0
        ;;
        
    *)
        print_error "无效选择"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "  ✅ 操作完成"
echo "========================================================================"
echo ""
print_info "更多信息请查看:"
echo "  - EXPERIMENT_GUIDE_V2.md (实验指南)"
echo "  - README_V2.md (项目说明)"
echo ""
