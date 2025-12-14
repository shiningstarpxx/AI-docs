"""
检查 Simple World Model V2 训练进度
"""
import os
import json
import time

def check_progress():
    result_dir = "./results_simple_wm_v2"
    
    print("=" * 60)
    print("📊 Simple World Model V2 训练进度检查")
    print("=" * 60)
    
    # 检查结果目录
    if not os.path.exists(result_dir):
        print("\n⏳ 状态: 训练中 (结果目录尚未创建)")
        print("   - 当前阶段: 可能在 DQN 预训练或数据收集")
        print("   - 预计还需: 30-40 分钟")
        return
    
    # 检查训练历史文件
    history_file = f"{result_dir}/training_history.json"
    if not os.path.exists(history_file):
        print("\n⏳ 状态: 训练中 (历史文件尚未保存)")
        print(f"   - 结果目录已创建: {result_dir}")
        return
    
    # 读取训练历史
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print("\n✅ 状态: 训练完成或正在进行中\n")
    
    # 阶段 0: DQN 预训练
    if history.get("dqn_pretrain_rewards"):
        rewards = history["dqn_pretrain_rewards"]
        print(f"🎯 阶段 0: DQN 预训练")
        print(f"   - Episodes: {len(rewards)}")
        print(f"   - 最终 20 轮平均: {sum(rewards[-20:])/20:.2f}")
        print()
    
    # 阶段 1: 数据收集
    if history.get("data_collection_rewards"):
        rewards = history["data_collection_rewards"]
        print(f"📦 阶段 1: 数据收集")
        print(f"   - Episodes: {len(rewards)}")
        print(f"   - 平均奖励: {sum(rewards)/len(rewards):.2f}")
        print(f"   - 最大奖励: {max(rewards):.0f}")
        print()
    
    # 阶段 2: 世界模型训练
    if history.get("world_model_losses"):
        losses = history["world_model_losses"]
        print(f"🌍 阶段 2: 世界模型训练")
        print(f"   - Epochs: {len(losses)}")
        print(f"   - 最终 Loss: {losses[-1]:.6f}")
        print(f"   - 初始 Loss: {losses[0]:.6f}")
        print(f"   - 下降比例: {(1 - losses[-1]/losses[0])*100:.1f}%")
        print()
    
    # 阶段 3: 梦境训练
    if history.get("controller_dream_rewards"):
        rewards = history["controller_dream_rewards"]
        print(f"💭 阶段 3: 梦境训练控制器")
        print(f"   - 检查点: {len(rewards)}")
        print(f"   - 最终梦境奖励: {rewards[-1]:.2f}")
        if len(rewards) > 1:
            print(f"   - 初始梦境奖励: {rewards[0]:.2f}")
            print(f"   - 提升: {rewards[-1] - rewards[0]:.2f}")
        print()
    
    # 最终评估
    if history.get("evaluation_rewards"):
        eval_reward = history["evaluation_rewards"][0]
        print("=" * 60)
        print(f"🎉 最终评估结果: {eval_reward:.2f}")
        print("=" * 60)
        
        # 对比
        print("\n📈 性能对比:")
        print(f"   V1 (失败): 17.06")
        print(f"   V2 (改进): {eval_reward:.2f}")
        
        if eval_reward > 100:
            print(f"\n✅ 成功! V2 性能 > 100 (目标达成)")
            if eval_reward > 150:
                print(f"🌟 优秀! V2 性能 > 150 (接近 DQN baseline 193)")
        else:
            print(f"\n⚠️  V2 性能仍 < 100, 但相比 V1 有提升")
    else:
        print("⏳ 训练尚未完成，等待最终评估...")

if __name__ == "__main__":
    check_progress()
