# World Models 项目 TODO

## 待优化项

### [2025-12-20 周五] MPS/GPU 利用率优化

**问题描述：**
当前 CMA-ES 训练阶段 MPS 利用率不高，主要瓶颈在于：
1. Dream rollouts 是串行执行的（1000 步 Python 循环）
2. 64 个 controller 逐个评估
3. CMA-ES 参数更新、fitness 计算在 CPU
4. 原论文设计针对 2018 年 CPU 集群，未利用现代 GPU 批量并行能力

**优化方向：**

1. **批量化 dream rollouts**
   - 同时跑多个 controller 的 rollout
   - 将 population_size=64 的评估并行化
   - 预期提升：~10-50x

2. **向量化 RNN 推理**
   - 同时处理多条轨迹（batch inference）
   - 将 `n_rollouts_per_eval=16` 批量化
   - 减少 Python 循环开销

3. **实现参考：**
   ```python
   # 当前：串行
   for controller in population:
       for rollout in range(16):
           reward = dream_rollout(controller)  # 1000 步循环

   # 优化后：批量并行
   # batch_size = population_size * n_rollouts = 64 * 16 = 1024
   rewards = batched_dream_rollout(all_controllers)  # 向量化
   ```

**相关文件：**
- `world_models/experiments/3_car_racing_world_model.py`
- 重点函数：`dream_rollout()`, `evaluate_controller()`

**预期效果：**
- 训练时间从 ~3 天缩短到 ~0.5-1 天
- 更充分利用 MPS/GPU 计算能力

---

## 已完成

### [2025-12-15] 文档整理
- [x] `11_world_models_vs_dreamer.md` - World Models vs Dreamer 深度对比
- [x] `12_future_world_models.md` - 新一代世界模型方向
- [x] `13_genie_jepa_comparison.md` - Genie + JEPA + 三方对比
- [x] `World_Models_Deep_Dive.md/pdf` - 技术分享 PPT

### [2025-12-11] 训练启动
- [x] CarRacing paper 模式训练启动
- [x] 数据收集完成 (10000 rollouts, 10M frames)
- [x] VAE 训练完成 (10 epochs)
- [x] MDN-RNN 训练完成 (20 epochs)
- [ ] CMA-ES 控制器训练进行中 (Generation 70/300, All-time best: -9.9)
