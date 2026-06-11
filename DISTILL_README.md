# HIMLoco → Depth FM Diffusion Distillation

> 分支: `depth_fm_distill` | 日期: 2026-06-10

## 概述

从 HIMLoco 盲态教师策略出发，通过蒸馏训练一个以**深度图**为输入、**Flow Matching 扩散策略**为输出的学生网络。

```
Phase 1 (教师训练):  本体感知 → HIMEstimator → MLP Actor → 关节动作
                         ↑ 纯盲态，不感知地形

Phase 2 (学生蒸馏):  深度图 + 本体感知 → DepthEncoder + HIMEstimator
                                       → FMDiffusionPolicy → 关节动作
                         ↑ 教师 scandot 隐变量监督学生 depth 隐变量
```

---

## 快速开始

### 环境要求

- NVIDIA GPU (RTX Titan 24GB 可)
- Isaac Gym Preview 4
- PyTorch 1.10+ (CUDA 11.3)
- Python 3.7+
- OpenCV (可选，深度图可视化用)

### Step 0: 验证深度相机渲染 (推荐先做)

在开始训练前，先确认仿真相机配置正确：

```bash
cd HIMLoco

# GUI 模式 — 显示 IsaacGym 窗口 + OpenCV 深度图
python test_depth_camera.py --iters 200 --gui

# 或保存样本图像
python test_depth_camera.py --iters 50 --save
```

**检查清单**：

| 检查项 | 正常 | 异常处理 |
|--------|:--:|------|
| 深度图不是全黑 | ✅ 能看到地面渐变 | 全黑 → 检查 FOV/相机位姿 |
| 深度值范围 | ✅ 0.3~2.0m | 全是 >2m → 相机仰角太大 |
| 裁剪线位置 | ✅ 红线内全是地形 | 有机器人部件 → 调整 crop 参数 |
| 有效像素比例 | ✅ >80% | <20% → 检查 near_clip/far_clip |

如发现问题，在 `test_depth_camera.py` 的 `DepthConfig` 和 `depth_fm/configs/go2_distill_config.py` 中同步调整参数。

### Step 1: Phase 1 — 训练教师

```bash
cd legged_gym/legged_gym/scripts

# 训练盲态 HIM 教师 (与 HIMLoco 原版一致)
python train_teacher.py --task=go2 --headless
```

**输入**: 本体感知 (270维)
**输出**: `./logs/rough_go2/model_final.pt`

**预期训练时间**: RTX Titan 24GB, 4096 envs, 20000 iterations → 约 3-5 小时

### Step 2: Phase 2 — 蒸馏学生

```bash
# 正常训练 (无 GUI，相机离屏渲染)
python train_student.py --task=go2

# 手动指定教师权重 + 指定 GPU
python train_student.py --task=go2 \
    --teacher_path logs/rough_go2/Jun10_21-51-33_/model_5000.pt \
    --rl_device cuda:1 --sim_device cuda:1

# 调试模式: 显示仿真 viewer (3D 机器人+地形)
python train_student.py --task=go2 --show_sim

# 调试模式: 显示深度图 OpenCV 窗口
python train_student.py --task=go2 --show_depth

# 全部显示 (调试用)
python train_student.py --task=go2 --show_sim --show_depth
```

**显示选项**:

| 标志 | 作用 | 正常训练 |
|------|------|:---:|
| (默认) | 无 GUI，GPU 离屏渲染 | ✅ |
| `--show_sim` | Isaac Gym 3D 仿真 viewer | ❌ 吃 GPU |
| `--show_depth` | OpenCV 深度图窗口 | ❌ 吃 GPU |

**教师权重查找逻辑**:
1. `--teacher_path` CLI 参数优先
2. 否则递归搜索 `depth_fm/configs/go2_distill_config.py` 中 `teacher_ckpt_path` 目录
3. 自动选 `model_*.pt` 中迭代数最大的

**输入**: 本体感知 (270维) + 深度图 (2×58×98)
**输出**: `./logs/go2_distill/student_final.pt`

**预期训练时间**: RTX Titan 24GB, 4096 envs, 20000 iterations → 约 4-8 小时

### Step 3: 回放学生策略

```bash
python play_student.py \
    --task=go2 \
    --checkpoint=./logs/go2_distill/student_final.pt
```

---

## 参数调优指南

### 蒸馏损失系数

| 参数 | 默认值 | 作用 | 调参方向 |
|------|:---:|------|------|
| `latent_loss_coef` | 0.1 | 隐空间对齐 (depth_latent → scandot_latent) | 训练不稳定 → **增大**到 0.5-1.0 |
| `action_loss_coef` | 0.0 | 行为克隆 (student_action → teacher_action) | 学生动作偏差大 → 设为 0.1-0.5 |

在 `depth_fm/configs/go2_distill_config.py` 中修改 `DistillModelCfg` 类。

### FM 推理步数

| 参数 | 默认值 | 延迟 (AGX Orin) | 质量 |
|------|:---:|:---:|:---:|
| `fm_num_steps_infer = 3` | — | ~5ms | 可接受 |
| `fm_num_steps_infer = 5` | **默认** | ~8ms | 好 |
| `fm_num_steps_infer = 10` | — | ~15ms | 更好，可能超 20ms 预算 |

### 深度相机参数

在 `depth_fm/configs/go2_distill_config.py` 的 `GO2DistillCfg.depth` 中：

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `position` | `[0.30, 0.0, 0.35]` | 相机在 Go2 base link 下的安装位置 |
| `angle` | `[-3, 3]` | 下倾角随机化范围 (仿真域随机化) |
| `dis_noise` | `0.005` | 深度图噪声，真机 gap 大时增大到 0.01-0.02 |
| `update_interval` | `5` | 每 5 个控制步更新一次深度 (10Hz) |
| `far_clip` | `2.0` | 深度最远范围 (m)，太远噪声大 |
| `near_clip` | `0.0` | 深度最近范围 (m) |

---

## 训练监控

```bash
# 启动 TensorBoard
tensorboard --logdir=./logs/go2_distill/
```

### 关键指标

| 指标 | 正常范围 | 异常信号 |
|------|:---:|------|
| `fm_loss` | 0.01 → 0.001 (逐步下降) | >0.1 不下降 → 学习率太大或条件编码有问题 |
| `latent_loss` | 0.01 → 0.005 (逐步下降) | >0.1 → latent_loss_coef 可能太小 |
| `episode_reward` | 与教师接近 (80-100%) | <50% 教师 → 蒸馏不够，加大 action_loss_coef |
| `episode_length` | 接近 max (20s) | 频繁提前终止 → 学生策略不稳定 |

---

## 真机部署预备

### 真机 D435i → 仿真对齐

```
真机管线:
  D435i 424×240
    → crop(top=4, bottom=4, left=16, right=16)  # 等比对应仿真 98×58 裁剪
    → 392×232
    → bilinear resize → 98×58
    → clamp [0, 2m]
    → normalize → [-0.5, 0.5]

仿真管线:
  IsaacGym 106×60
    → crop(bottom=2, left=4, right=4)
    → 98×58
    → clamp [0, 2m]
    → normalize → [-0.5, 0.5]
```

### ONNX 导出 (后续)

```python
# 待完成: 将学生模型导出为 ONNX
# 输入: obs_history [1, 270], depth_images [1, 2, 58, 98]
# 输出: action [12]
```

---

## 文件索引

```
depth_fm/
├── modules/
│   ├── scandot_encoder.py          # 教师特权地形编码器
│   ├── depth_encoder.py            # 学生深度图编码器
│   ├── fm_diffusion.py             # FM 扩散策略 (速度场网络 + 采样器)
│   └── him_distill_model.py        # 蒸馏联合模型
├── algorithms/
│   └── fm_distillation.py          # FM 蒸馏训练算法
├── runners/
│   └── distill_runner.py           # 蒸馏训练 Runner
├── configs/
│   └── go2_distill_config.py       # Go2 蒸馏超参数
└── envs/
    └── depth_robot.py              # 带深度相机的仿真环境

legged_gym/legged_gym/
├── envs/base/
│   └── legged_robot_config.py      # 新增 class depth 配置块
└── scripts/
    ├── train_teacher.py            # Phase 1 入口
    ├── train_student.py            # Phase 2 入口
    └── play_student.py             # 学生推理/回放
```

---

## 常见问题

**Q: 教师训练收敛慢?**
A: 检查 HIMLoco 原版 `go2_config.py` 中的 reward scales 和 terrain 配置。确保 `max_curriculum = 2.0` 足够大。

**Q: 蒸馏时 depth_latent 和 scandot_latent 不收敛?**
A: (1) 检查 scandot 和 depth 是否覆盖相同的物理 FOV; (2) 增大 `latent_loss_coef` 到 0.5; (3) 先设置 `action_loss_coef=0.5` 辅助收敛，再逐步降低。

**Q: FM 扩散推理太慢?**
A: (1) 降低 `fm_num_steps_infer` 到 3; (2) 实现 CUDA Graph 封装 K 步循环; (3) TensorRT FP16 导出。

**Q: 真机深度图 sim-to-real gap 大?**
A: (1) 增大 `dis_noise` 到 0.01-0.02; (2) 训练时加入随机 dropout 区域 (模拟遮挡/缺失); (3) 仿真中模拟延迟 (`latency_depth=0.08s`)。

---

> 文档生成时间: 2026-06-10 | 分支: depth_fm_distill
