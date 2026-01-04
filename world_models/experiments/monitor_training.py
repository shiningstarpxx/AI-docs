#!/usr/bin/env python3
"""
实时监控训练进度脚本
支持监控原版和改进版的训练进度
"""

import os
import sys
import time
import pickle
import numpy as np
from datetime import datetime, timedelta
import argparse

def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')

def format_time(seconds):
    """格式化时间"""
    return str(timedelta(seconds=int(seconds)))

def get_memory_usage():
    """获取内存使用"""
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        return f"{mem_mb:.0f} MB"
    except:
        return "N/A"

def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def get_log_tail(log_path, n=20):
    """获取日志最后n行"""
    if not os.path.exists(log_path):
        return []
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-n:]]
    except:
        return []

def estimate_eta(checkpoint):
    """估算剩余时间"""
    if not checkpoint or checkpoint["training_state"]["start_time"] is None:
        return "Unknown"
    
    elapsed = time.time() - checkpoint["training_state"]["start_time"]
    stage = checkpoint["training_state"]["stage"]
    
    # 根据阶段估算进度
    progress = 0
    if stage == "data_collection":
        total_rollouts = checkpoint.get("config", {}).get("random_rollouts", 10000)
        current = checkpoint["training_state"]["data_collection_progress"]
        progress = 0.1 * (current / total_rollouts)
    elif stage == "vae_training":
        total_epochs = checkpoint.get("config", {}).get("vae_epochs", 10)
        current = checkpoint["training_state"]["vae_epoch"]
        progress = 0.1 + 0.2 * (current / total_epochs)
    elif stage == "rnn_training":
        total_epochs = checkpoint.get("config", {}).get("rnn_epochs", 20)
        current = checkpoint["training_state"]["rnn_epoch"]
        progress = 0.3 + 0.25 * (current / total_epochs)
    elif stage == "controller_training":
        total_gens = checkpoint.get("config", {}).get("generations", 300)
        current = checkpoint["training_state"]["cmaes_generation"]
        progress = 0.55 + 0.45 * (current / total_gens)
    elif stage == "done":
        progress = 1.0
    
    if progress > 0:
        total_estimated = elapsed / progress
        remaining = total_estimated - elapsed
        return format_time(remaining)
    return "Calculating..."

def display_progress_bar(current, total, width=40):
    """显示进度条"""
    if total == 0:
        return "[" + " " * width + "] 0%"
    
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent*100:.1f}%"

def monitor_training(result_dir, refresh_interval=5):
    """监控训练进度"""
    checkpoint_path = f"{result_dir}/checkpoint.pkl"
    log_path = f"{result_dir}/training.log"
    
    print(f"\n{'='*80}")
    print(f"监控目录: {result_dir}")
    print(f"刷新间隔: {refresh_interval}秒 (按 Ctrl+C 退出)")
    print(f"{'='*80}\n")
    
    try:
        while True:
            clear_screen()
            
            # 标题
            print(f"\n{'='*80}")
            print(f"🚀 World Models 训练监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
            
            # 加载checkpoint
            checkpoint = load_checkpoint(checkpoint_path)
            
            if not checkpoint:
                print("⚠️  未找到checkpoint文件，等待训练开始...\n")
                print(f"Checkpoint路径: {checkpoint_path}")
                print(f"下次刷新: {refresh_interval}秒后")
                time.sleep(refresh_interval)
                continue
            
            state = checkpoint["training_state"]
            history = checkpoint["history"]
            
            # 基本信息
            print(f"📊 训练状态")
            print(f"{'─'*80}")
            print(f"  当前阶段: {state['stage'].upper()}")
            print(f"  模式: {checkpoint.get('config_mode', 'Unknown')}")
            print(f"  内存使用: {get_memory_usage()}")
            
            # 计算运行时间
            if state["start_time"]:
                elapsed = time.time() - state["start_time"]
                print(f"  已运行: {format_time(elapsed)}")
                eta = estimate_eta(checkpoint)
                print(f"  预计剩余: {eta}")
            print()
            
            # 各阶段进度
            print(f"📈 训练进度")
            print(f"{'─'*80}")
            
            # Stage 1: 数据收集
            if state["stage"] in ["data_collection", "vae_training", "rnn_training", "controller_training", "done"]:
                total_rollouts = 10000  # 默认值
                current_rollouts = state["data_collection_progress"]
                bar = display_progress_bar(current_rollouts, total_rollouts)
                print(f"  1️⃣  数据收集: {current_rollouts}/{total_rollouts}")
                print(f"      {bar}")
                print(f"      Chunks: {checkpoint.get('num_chunks_saved', 0)}")
            else:
                print(f"  1️⃣  数据收集: 待开始")
            print()
            
            # Stage 2a: VAE训练
            if state["stage"] in ["vae_training", "rnn_training", "controller_training", "done"]:
                total_epochs = 10
                current_epoch = state["vae_epoch"]
                bar = display_progress_bar(current_epoch, total_epochs)
                print(f"  2️⃣a VAE训练: {current_epoch}/{total_epochs} epochs")
                print(f"      {bar}")
                if history["vae_loss"]:
                    print(f"      最新Loss: {history['vae_loss'][-1]:.4f}")
            else:
                print(f"  2️⃣a VAE训练: 待开始")
            print()
            
            # Stage 2b: RNN训练
            if state["stage"] in ["rnn_training", "controller_training", "done"]:
                total_epochs = 20
                current_epoch = state["rnn_epoch"]
                bar = display_progress_bar(current_epoch, total_epochs)
                print(f"  2️⃣b RNN训练: {current_epoch}/{total_epochs} epochs")
                print(f"      {bar}")
                if history["rnn_loss"]:
                    print(f"      最新Loss: {history['rnn_loss'][-1]:.4f}")
            else:
                print(f"  2️⃣b RNN训练: 待开始")
            print()
            
            # Stage 3: Controller训练
            if state["stage"] in ["controller_training", "done"]:
                total_gens = 300
                current_gen = state["cmaes_generation"]
                bar = display_progress_bar(current_gen, total_gens)
                print(f"  3️⃣  Controller训练: {current_gen}/{total_gens} generations")
                print(f"      {bar}")
                if history["dream_fitness"]:
                    print(f"      最佳梦境适应度: {checkpoint.get('best_fitness', 0):.2f}")
                    print(f"      最近10代均值: {np.mean(history['dream_fitness'][-10:]):.2f}")
            else:
                print(f"  3️⃣  Controller训练: 待开始")
            print()
            
            # 训练曲线趋势
            print(f"📉 训练趋势 (最近10次)")
            print(f"{'─'*80}")
            
            if len(history["dream_fitness"]) >= 2:
                recent = history["dream_fitness"][-10:]
                if len(recent) >= 2:
                    trend = "📈" if recent[-1] > recent[0] else "📉"
                    print(f"  梦境适应度: {trend} {recent[0]:.2f} → {recent[-1]:.2f}")
            
            if history["real_reward"]:
                print(f"  真实环境奖励: {history['real_reward'][-1]:.2f}")
            print()
            
            # 最近日志
            print(f"📝 最近日志 (最后5条)")
            print(f"{'─'*80}")
            log_lines = get_log_tail(log_path, 5)
            if log_lines:
                for line in log_lines:
                    # 截断过长的行
                    if len(line) > 78:
                        line = line[:75] + "..."
                    print(f"  {line}")
            else:
                print("  (暂无日志)")
            print()
            
            # 底部提示
            print(f"{'─'*80}")
            print(f"⏱️  下次刷新: {refresh_interval}秒后 | 按 Ctrl+C 退出")
            print(f"{'='*80}\n")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止\n")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="监控World Models训练进度")
    parser.add_argument("--dir", type=str, 
                       default="./results_car_racing_paper",
                       help="结果目录 (默认: ./results_car_racing_paper)")
    parser.add_argument("--interval", type=int, default=5,
                       help="刷新间隔(秒) (默认: 5)")
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    result_dir = os.path.abspath(args.dir)
    
    if not os.path.exists(result_dir):
        print(f"❌ 错误: 目录不存在: {result_dir}")
        sys.exit(1)
    
    monitor_training(result_dir, args.interval)

if __name__ == "__main__":
    main()
