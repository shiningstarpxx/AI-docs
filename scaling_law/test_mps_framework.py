#!/usr/bin/env python3
"""
快速测试 MPS 框架的可用性
运行时间：< 5 分钟
"""

import torch
import sys
from pathlib import Path

def test_mps_availability():
    """测试 MPS 是否可用"""
    print("=" * 60)
    print("1️⃣ 测试 MPS 可用性")
    print("=" * 60)
    
    if not torch.backends.mps.is_available():
        print("❌ MPS 不可用")
        if not torch.backends.mps.is_built():
            print("   原因：PyTorch 未编译 MPS 支持")
            print("   解决：pip install --pre torch torchvision torchaudio")
        return False
    
    print("✅ MPS 可用")
    print(f"   PyTorch 版本: {torch.__version__}")
    return True

def test_mps_computation():
    """测试 MPS 计算"""
    print("\n" + "=" * 60)
    print("2️⃣ 测试 MPS 计算能力")
    print("=" * 60)
    
    device = torch.device("mps")
    
    # 简单计算
    try:
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print("✅ 矩阵乘法: 成功")
    except Exception as e:
        print(f"❌ 矩阵乘法失败: {e}")
        return False
    
    # 神经网络计算
    try:
        model = torch.nn.Linear(100, 10).to(device)
        input_data = torch.randn(32, 100, device=device)
        output = model(input_data)
        print("✅ 神经网络前向: 成功")
    except Exception as e:
        print(f"❌ 神经网络计算失败: {e}")
        return False
    
    # 反向传播
    try:
        loss = output.sum()
        loss.backward()
        print("✅ 反向传播: 成功")
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")
        return False
    
    return True

def test_memory_management():
    """测试内存管理"""
    print("\n" + "=" * 60)
    print("3️⃣ 测试内存管理")
    print("=" * 60)
    
    import gc
    import psutil
    
    process = psutil.Process()
    initial_mem = process.memory_info().rss / (1024 * 1024)
    print(f"初始内存: {initial_mem:.0f} MB")
    
    # 创建大张量
    device = torch.device("mps")
    tensors = []
    for i in range(10):
        tensors.append(torch.randn(1000, 1000, device=device))
    
    after_alloc_mem = process.memory_info().rss / (1024 * 1024)
    print(f"分配后内存: {after_alloc_mem:.0f} MB (+{after_alloc_mem - initial_mem:.0f} MB)")
    
    # 清理
    del tensors
    torch.mps.empty_cache()
    gc.collect()
    
    after_clean_mem = process.memory_info().rss / (1024 * 1024)
    print(f"清理后内存: {after_clean_mem:.0f} MB")
    
    if after_clean_mem < after_alloc_mem * 0.8:
        print("✅ 内存清理: 成功")
        return True
    else:
        print("⚠️ 内存清理: 部分成功")
        return True

def test_training_speed():
    """测试训练速度"""
    print("\n" + "=" * 60)
    print("4️⃣ 测试训练速度")
    print("=" * 60)
    
    import time
    
    # CPU 速度
    model_cpu = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10)
    )
    
    data_cpu = torch.randn(64, 512)
    
    start = time.time()
    for _ in range(100):
        output = model_cpu(data_cpu)
        loss = output.sum()
        loss.backward()
    cpu_time = time.time() - start
    
    print(f"CPU 速度: {100/cpu_time:.1f} iters/s")
    
    # MPS 速度
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model_mps = model_cpu.to(device)
        data_mps = data_cpu.to(device)
        
        # Warmup
        for _ in range(10):
            output = model_mps(data_mps)
            loss = output.sum()
            loss.backward()
        
        start = time.time()
        for _ in range(100):
            output = model_mps(data_mps)
            loss = output.sum()
            loss.backward()
        mps_time = time.time() - start
        
        print(f"MPS 速度: {100/mps_time:.1f} iters/s")
        print(f"加速比: {cpu_time/mps_time:.1f}x")
        
        if mps_time < cpu_time:
            print("✅ MPS 比 CPU 快")
            return True
        else:
            print("⚠️ MPS 没有加速（可能是模型太小）")
            return True
    
    return False

def test_framework_import():
    """测试框架导入"""
    print("\n" + "=" * 60)
    print("5️⃣ 测试框架导入")
    print("=" * 60)
    
    try:
        from mps_framework_example import (
            get_mps_device,
            clear_mps_cache,
            ModelConfig,
            SimpleGPT,
            ScalingExperiment
        )
        print("✅ 框架导入: 成功")
        return True
    except ImportError as e:
        print(f"❌ 框架导入失败: {e}")
        return False

def test_mini_experiment():
    """运行微型实验"""
    print("\n" + "=" * 60)
    print("6️⃣ 运行微型实验")
    print("=" * 60)
    
    try:
        from mps_framework_example import (
            get_mps_device,
            ModelConfig,
            SimpleGPT,
            DummyTextDataset,
            MPSTrainer
        )
        from torch.utils.data import DataLoader
        
        device = get_mps_device()
        
        # 超小模型
        config = ModelConfig(n_layers=2, d_model=128, n_heads=2, max_seq_len=64)
        model = SimpleGPT(config)
        print(f"   模型参数: {config.n_params/1e6:.1f}M")
        
        # 超小数据
        dataset = DummyTextDataset(n_tokens=10000, seq_len=64, vocab_size=config.vocab_size)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"   数据量: 10K tokens")
        
        # 训练 100 步
        trainer = MPSTrainer(
            model=model,
            device=device,
            config={'lr': 3e-4, 'max_steps': 100}
        )
        
        print("   开始训练...")
        result = trainer.train(dataloader, max_steps=100, eval_interval=50, early_stop=False)
        
        print(f"   最终 loss: {result['final_loss']:.3f}")
        print("✅ 微型实验: 成功")
        return True
        
    except Exception as e:
        print(f"❌ 微型实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n🧪 MPS 框架快速测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("MPS 可用性", test_mps_availability()))
    
    if results[0][1]:  # 只有 MPS 可用才继续
        results.append(("MPS 计算", test_mps_computation()))
        results.append(("内存管理", test_memory_management()))
        results.append(("训练速度", test_training_speed()))
    
    results.append(("框架导入", test_framework_import()))
    
    if results[-1][1]:  # 框架导入成功才运行实验
        results.append(("微型实验", test_mini_experiment()))
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！你可以开始运行完整实验了：")
        print("   python mps_framework_example.py --mode quick")
    else:
        print("\n⚠️ 部分测试失败，请检查环境配置")
        sys.exit(1)

if __name__ == '__main__':
    main()
