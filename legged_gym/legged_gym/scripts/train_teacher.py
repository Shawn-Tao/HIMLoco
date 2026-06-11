"""
Phase 1: 教师训练脚本。

基于 HIMLoco 原有训练流程，新增 scandot encoder 的记录能力。

用法:
    python legged_gym/legged_gym/scripts/train_teacher.py --task=go2 --headless

输出:
    ./logs/rough_go2/model_final.pt  — 完整的教师模型权重
    ./logs/rough_go2/model_XXXXX.pt  — 中间 checkpoint

说明:
    教师训练完全独立于学生蒸馏。训练完成后，model_final.pt 中的权重
    会被 Phase 2 的蒸馏脚本加载。
"""

import numpy as np
import os
import sys
from datetime import datetime

# 将 HIMLoco 根目录加入 path，确保 depth_fm 模块可导入
HIMLOCO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
if HIMLOCO_ROOT not in sys.path:
    sys.path.insert(0, HIMLOCO_ROOT)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def train(args, headless=True):
    """
    教师训练主函数。

    与 HIMLoco 原版 train.py 一致，区别仅在于:
      1. 日志路径清晰
      2. 保存的 checkpoint 包含完整的 actor_critic state_dict
         (包括 HIMEstimator, Actor, Critic 所有权重)
    """
    args.headless = headless
    # args.resume 由命令行控制，不在此覆盖

    print("=" * 60)
    print(f"[Teacher Training] Task: {args.task}")
    print(f"[Teacher Training] Start time: {datetime.now()}")
    print(f"[Teacher Training] Output: ./logs/rough_go2/")
    print("=" * 60)

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )

    # 训练
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )

    print(f"\n[Teacher Training] Complete at {datetime.now()}")
    print(f"[Teacher Training] → 下一步: 运行 train_student.py 开始蒸馏")


if __name__ == '__main__':
    args = get_args()
    train(args, headless=True)
