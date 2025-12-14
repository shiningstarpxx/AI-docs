#!/bin/bash
# 监控训练进度

echo "=== World Models 训练监控 ==="
echo ""

# 检查正在运行的进程
echo "📊 正在运行的训练:"
ps aux | grep -E "(python.*2_simple|python.*1_baseline|python.*3_mini)" | grep -v grep | awk '{print $2, $11, $12}'
echo ""

# 检查结果目录
echo "📁 已完成的实验:"
for dir in results_*/; do
    if [ -d "$dir" ]; then
        echo "  - $dir"
        if [ -f "$dir/training_history.json" ]; then
            # 提取最终性能
            if command -v jq &> /dev/null; then
                eval_reward=$(jq '.evaluation_rewards[-1] // "N/A"' "$dir/training_history.json")
                echo "    最终性能: $eval_reward"
            fi
        fi
    fi
done
echo ""

# 磁盘使用
echo "💾 存储占用:"
du -sh results_*/ 2>/dev/null | sort -h
echo ""

echo "💡 提示: 使用 'tail -f <log_file>' 查看训练日志"
