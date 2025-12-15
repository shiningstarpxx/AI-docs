# Tasks: DreamerV3 代码走读

## 1. 代码仓库熟悉
- [ ] 1.1 Clone 官方仓库 (github.com/danijar/dreamerv3)
- [ ] 1.2 理解目录结构和模块划分
- [ ] 1.3 配置文件和超参数分析

## 2. World Model 实现
- [ ] 2.1 RSSM 实现细节 (`nets.py` 或相关文件)
- [ ] 2.2 Encoder/Decoder 网络结构
- [ ] 2.3 离散潜在空间 (32×32 Categorical) 实现
- [ ] 2.4 KL Balancing 的具体代码

## 3. Actor-Critic 实现
- [ ] 3.1 Actor 网络和动作采样
- [ ] 3.2 Critic 网络和价值估计
- [ ] 3.3 λ-Returns 计算实现
- [ ] 3.4 想象轨迹 (imagination) 生成

## 4. 训练循环分析
- [ ] 4.1 数据收集和 replay buffer
- [ ] 4.2 World Model 训练步骤
- [ ] 4.3 Policy 训练步骤
- [ ] 4.4 symlog 归一化的应用位置

## 5. 工程 Trick 总结
- [ ] 5.1 梯度裁剪、学习率调度
- [ ] 5.2 并行化和效率优化
- [ ] 5.3 数值稳定性处理

## 6. 文档输出
- [ ] 6.1 创建 `17_dreamerv3_code_walkthrough.md` 代码走读文档
- [ ] 6.2 整理关键代码片段和注释
