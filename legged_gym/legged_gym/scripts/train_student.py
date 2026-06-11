"""
Phase 2: 学生蒸馏训练脚本。

加载 Phase 1 训练的教师权重，用深度图 + FM 扩散策略蒸馏学生。

用法:
    # 必须先跑 Phase 1 训练教师:
    python legged_gym/legged_gym/scripts/train_teacher.py --task=go2 --headless

    # 然后运行蒸馏:
    python legged_gym/legged_gym/scripts/train_student.py --task=go2 --headless

教师权重路径:
    ./logs/rough_go2/model_final.pt  (train_teacher.py 的输出)

蒸馏输出:
    ./logs/go2_distill/student_final.pt      — 最终学生权重
    ./logs/go2_distill/student_ckpt_XXXXX.pt — 中间 checkpoint
"""

import numpy as np
import os
import sys
from datetime import datetime

HIMLOCO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
if HIMLOCO_ROOT not in sys.path:
    sys.path.insert(0, HIMLOCO_ROOT)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

# 深度+FM+蒸馏模块
from depth_fm.modules.him_distill_model import HIMDistillModel
from depth_fm.algorithms.fm_distillation import FMDistillation
from depth_fm.runners.distill_runner import DistillRunner
from depth_fm.configs.go2_distill_config import DistillModelCfg
from depth_fm.envs.camera_patch import patch_depth_camera


def train_student(args, headless=True):
    """学生蒸馏训练主函数"""

    # ================================================================
    # 配置
    # ================================================================
    args.headless = headless
    student_cfg = DistillModelCfg()

    # 解析教师路径: 优先 CLI > 自动搜索 config 目录
    if getattr(args, 'teacher_path', None):
        teacher_path = args.teacher_path
        if not os.path.isabs(teacher_path):
            teacher_path = os.path.join(HIMLOCO_ROOT, teacher_path)
        print(f"[Student] 手动指定教师: {teacher_path}")
    else:
        teacher_root = student_cfg.teacher_ckpt_path
        if not os.path.isabs(teacher_root):
            teacher_root = os.path.join(HIMLOCO_ROOT, teacher_root)
        model_files = []
        for r, _, fs in os.walk(teacher_root):
            for f in fs:
                if f.startswith('model_') and f.endswith('.pt'):
                    model_files.append(os.path.join(r, f))
        if model_files:
            teacher_path = sorted(model_files)[-1]
            print(f"[Student] 自动选择最新教师: {os.path.relpath(teacher_path, HIMLOCO_ROOT)}")
        else:
            print(f"[Student] ⚠ 在 {teacher_root} 下未找到 model_*.pt")
            print(f"[Student] 请先运行 train_teacher.py 或 --teacher_path 手动指定")
            return

    log_dir = os.path.join(HIMLOCO_ROOT, 'logs', 'go2_distill')
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print(f"[Student Distillation] Task: {args.task}")
    print(f"[Student Distillation] Teacher: {student_cfg.teacher_ckpt_path}")
    print(f"[Student Distillation] Log:    {log_dir}")
    print(f"[Student Distillation] Start:  {datetime.now()}")
    print("=" * 60)

    # ================================================================
    # 1. 创建环境（自带深度相机）
    # ================================================================
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    _, train_cfg = task_registry.get_cfgs(args.task)

    # 激活深度相机 (打补丁)
    if hasattr(env.cfg, 'depth'):
        env.cfg.depth.use_camera = True
    env = patch_depth_camera(env)

    print(f"\n[Student] Env created: {env.num_envs} envs")

    # ================================================================
    # 2. 加载教师，创建蒸馏模型
    # ================================================================
    device = env.device if hasattr(env, 'device') else 'cuda:0'

    distill_model = HIMDistillModel(
        teacher_ckpt_path=teacher_path,
        num_actor_obs=student_cfg.num_actor_obs,
        num_critic_obs=student_cfg.num_critic_obs,
        num_one_step_obs=student_cfg.num_one_step_obs,
        num_actions=student_cfg.num_actions,
        actor_hidden_dims=student_cfg.actor_hidden_dims,
        critic_hidden_dims=student_cfg.critic_hidden_dims,
        num_scandots=student_cfg.num_scandots,
        scandot_grid_h=student_cfg.scandot_grid_h,
        scandot_grid_w=student_cfg.scandot_grid_w,
        scandot_latent_dim=student_cfg.scandot_latent_dim,
        depth_num_frames=student_cfg.depth_num_frames,
        depth_height=student_cfg.depth_height,
        depth_width=student_cfg.depth_width,
        depth_latent_dim=student_cfg.depth_latent_dim,
        fm_horizon=student_cfg.fm_horizon,
        fm_hidden_dim=student_cfg.fm_hidden_dim,
        fm_num_steps_infer=student_cfg.fm_num_steps_infer,
        latent_loss_coef=student_cfg.latent_loss_coef,
        action_loss_coef=student_cfg.action_loss_coef,
        device=device,
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in distill_model.parameters())
    trainable_params = sum(
        p.numel() for p in distill_model.get_trainable_parameters()
    )
    print(f"[Student] Total params: {total_params:,}")
    print(f"[Student] Trainable:   {trainable_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")

    # ================================================================
    # 3. 创建蒸馏算法
    # ================================================================
    distill_algo = FMDistillation(
        distill_model=distill_model,
        learning_rate=student_cfg.learning_rate,
        max_grad_norm=student_cfg.max_grad_norm,
        num_epochs_per_update=student_cfg.num_epochs_per_update,
        num_mini_batches=student_cfg.num_mini_batches,
        latent_loss_coef=student_cfg.latent_loss_coef,
        action_loss_coef=student_cfg.action_loss_coef,
        device=device,
        log_dir=log_dir,
    )

    # ================================================================
    # 4. 创建 Runner，开始蒸馏训练
    # ================================================================
    runner = DistillRunner(
        env=env,
        distill_model=distill_model,
        distill_algo=distill_algo,
        num_steps_per_env=train_cfg.runner.num_steps_per_env,
        save_interval=train_cfg.runner.save_interval,
        log_dir=log_dir,
        device=device,
    )

    resume_path = getattr(args, 'resume_path', None)
    if resume_path and os.path.exists(resume_path):
        print(f"[Student] Resuming from {resume_path}")
        distill_algo.load_student(resume_path)

    max_iter = train_cfg.runner.max_iterations
    runner.learn(num_iterations=max_iter, init_at_random_ep_len=True)

    print(f"\n[Student Distillation] Complete at {datetime.now()}")
    print(f"[Student] → 运行 play_student.py 查看效果")


if __name__ == '__main__':
    # 先提取 --teacher_path (get_args 不认识的参数)
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument('--teacher_path', type=str, default=None)
    _my_args, _remaining = _parser.parse_known_args()

    sys.argv = [sys.argv[0]] + _remaining
    args = get_args()
    args.teacher_path = _my_args.teacher_path
    train_student(args, headless=True)
