#!/bin/bash
#
# Scaling Law 研究项目 - 快速启动脚本
# 作者: peixingxin
# 日期: 2025-12-25
#

set -e  # 遇到错误立即退出

echo "=================================================="
echo "🚀 Scaling Law 研究项目 - 快速启动"
echo "=================================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否在正确的目录
if [ ! -f "research_plan.md" ]; then
    echo -e "${RED}❌ 错误：请在 scaling_law 目录下运行此脚本${NC}"
    exit 1
fi

# 步骤 1：检查 Python 版本
echo "1️⃣ 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}   Python 版本: $python_version${NC}"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "${RED}   ❌ Python 版本过低，需要 3.8+${NC}"
    exit 1
fi

# 步骤 2：创建虚拟环境（如果不存在）
echo ""
echo "2️⃣ 设置虚拟环境..."
if [ ! -d "venv" ]; then
    echo "   创建虚拟环境..."
    python3 -m venv venv
    echo -e "${GREEN}   ✅ 虚拟环境创建成功${NC}"
else
    echo -e "${YELLOW}   ⚠️  虚拟环境已存在，跳过创建${NC}"
fi

# 激活虚拟环境
echo "   激活虚拟环境..."
source venv/bin/activate

# 步骤 3：安装依赖
echo ""
echo "3️⃣ 安装依赖包..."

# 检查是否已安装
if python -c "import torch, numpy, scipy, matplotlib" 2>/dev/null; then
    echo -e "${YELLOW}   ⚠️  依赖包已安装，跳过${NC}"
else
    echo "   安装 PyTorch..."
    pip install --quiet torch torchvision torchaudio
    
    echo "   安装其他依赖..."
    pip install --quiet numpy scipy matplotlib psutil
    
    echo -e "${GREEN}   ✅ 依赖安装完成${NC}"
fi

# 步骤 4：验证 MPS
echo ""
echo "4️⃣ 验证 MPS 可用性..."
mps_check=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>&1)

if [ "$mps_check" = "True" ]; then
    echo -e "${GREEN}   ✅ MPS 可用！${NC}"
    pytorch_version=$(python3 -c "import torch; print(torch.__version__)")
    echo "   PyTorch 版本: $pytorch_version"
else
    echo -e "${RED}   ❌ MPS 不可用${NC}"
    echo "   可能的原因："
    echo "   1. 不是 Apple Silicon Mac"
    echo "   2. PyTorch 版本过低"
    echo "   3. 未安装 MPS 支持"
fi

# 步骤 5：运行测试
echo ""
echo "5️⃣ 运行快速测试..."
echo ""

if python test_mps_framework.py; then
    echo ""
    echo -e "${GREEN}=================================================="
    echo "✅ 环境配置成功！"
    echo "==================================================${NC}"
    echo ""
    echo "📚 下一步操作："
    echo ""
    echo "1. 运行快速验证（1-2 小时）："
    echo -e "   ${GREEN}python mps_framework_example.py --mode quick${NC}"
    echo ""
    echo "2. 查看完整文档："
    echo "   - research_plan.md - 完整学习计划"
    echo "   - MPS_FRAMEWORK_README.md - 使用指南"
    echo "   - CURRENT_STATUS.md - 项目状态"
    echo ""
    echo "3. 后台运行长时间实验："
    echo -e "   ${GREEN}nohup python mps_framework_example.py --mode dev > train.log 2>&1 &${NC}"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}=================================================="
    echo "❌ 测试失败"
    echo "==================================================${NC}"
    echo ""
    echo "请检查以下问题："
    echo "1. Python 版本是否 >= 3.8"
    echo "2. PyTorch 是否正确安装"
    echo "3. 是否在 Apple Silicon Mac 上运行"
    echo ""
    echo "获取帮助："
    echo "- 查看 MPS_FRAMEWORK_README.md 的故障排除部分"
    echo "- 运行 'python test_mps_framework.py' 获取详细错误信息"
    echo ""
    exit 1
fi
