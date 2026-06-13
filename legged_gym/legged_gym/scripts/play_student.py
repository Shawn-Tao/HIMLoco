"""
学生推理脚本: 加载蒸馏训练好的学生权重，在仿真中回放。

模型 = HIMEstimator(教师权重) + DepthEncoder(学生权重) + FMDiffusionPolicy(学生权重)

用法:
    python legged_gym/legged_gym/scripts/play_student.py \
        --task=go2 \
        --teacher_path logs/rough_go2/Jun11_15-57-07_/model_6000.pt \
        --checkpoint logs/go2_distill/student_ckpt_001350.pt
"""

import numpy as np
import os
import sys

HIMLOCO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
if HIMLOCO_ROOT not in sys.path:
    sys.path.insert(0, HIMLOCO_ROOT)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

from depth_fm.modules.him_distill_model import HIMDistillModel
from depth_fm.configs.go2_distill_config import DistillModelCfg
from depth_fm.envs.camera_patch import patch_depth_camera


def play(args, teacher_path: str = None, checkpoint_path: str = None):
    """加载学生模型并在仿真中回放"""

    # 1. 创建环境
    args.headless = False  # 需要 viewer 显示 + 深度相机图形上下文
    env, _ = task_registry.make_env(name=args.task, args=args)
    print(f"[Play] Env: {env.num_envs} envs")

    # 激活深度相机
    if hasattr(env.cfg, 'depth'):
        env.cfg.depth.use_camera = True
    env = patch_depth_camera(env)

    device = env.device if hasattr(env, 'device') else 'cuda:0'

    # 2. 创建模型
    cfg = DistillModelCfg()

    # 用教师路径初始化 HIM 估计器（不能传 None，否则是随机权重）
    if teacher_path is None:
        teacher_path = cfg.teacher_ckpt_path
        if not os.path.isabs(teacher_path):
            teacher_path = os.path.join(HIMLOCO_ROOT, teacher_path)
        # 自动搜索 model_*.pt
        if os.path.isdir(teacher_path):
            models = []
            for r, _, fs in os.walk(teacher_path):
                for f in fs:
                    if f.startswith('model_') and f.endswith('.pt'):
                        models.append(os.path.join(r, f))
            if models:
                teacher_path = sorted(models)[-1]
        print(f"[Play] Teacher: {teacher_path}")

    model = HIMDistillModel(
        teacher_ckpt_path=teacher_path,
        num_actor_obs=cfg.num_actor_obs,
        num_critic_obs=cfg.num_critic_obs,
        num_one_step_obs=cfg.num_one_step_obs,
        num_actions=cfg.num_actions,
        actor_hidden_dims=cfg.actor_hidden_dims,
        critic_hidden_dims=cfg.critic_hidden_dims,
        num_scandots=cfg.num_scandots,
        scandot_grid_h=cfg.scandot_grid_h,
        scandot_grid_w=cfg.scandot_grid_w,
        scandot_latent_dim=cfg.scandot_latent_dim,
        depth_num_frames=cfg.depth_num_frames,
        depth_height=cfg.depth_height,
        depth_width=cfg.depth_width,
        depth_latent_dim=cfg.depth_latent_dim,
        fm_horizon=cfg.fm_horizon,
        fm_hidden_dim=cfg.fm_hidden_dim,
        fm_num_steps_infer=cfg.fm_num_steps_infer,
        device=device,
    ).to(device)

    # 3. 加载学生权重 (覆盖 depth_encoder + fm_policy)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.depth_encoder.load_state_dict(ckpt['depth_encoder'])
        model.fm_policy.load_state_dict(ckpt['fm_policy'])
        print(f"[Play] Student loaded from {checkpoint_path}")
    else:
        print(f"[Play] ⚠ No checkpoint at {checkpoint_path}, using random student")

    model.eval_all()

    # 4. 主循环
    env.reset()
    print("\n[Play] Running... Press Ctrl+C to stop.\n")

    try:
        while True:
            obs_history = env.obs_buf
            depth = env.extras.get('depth', None)

            if depth is not None and depth.shape[0] > 0:
                condition, _ = model.get_student_condition(
                    obs_history[:1], depth[:1]
                )
                action_seq = model.fm_policy.sample(
                    condition, num_steps=cfg.fm_num_steps_infer
                )
                action = action_seq[:, 0, :]  # [1, 12] → 取第一步
            else:
                # 深度未就绪
                with torch.no_grad():
                    _, him_z = model.him_estimator(obs_history[:1])
                dummy_cond = torch.cat([
                    obs_history[:1, :cfg.num_one_step_obs],
                    torch.zeros(1, 3, device=device),
                    him_z,
                    torch.zeros(1, cfg.depth_latent_dim, device=device),
                ], dim=-1)
                action_seq = model.fm_policy.sample(
                    dummy_cond, num_steps=cfg.fm_num_steps_infer
                )
                action = action_seq[:, 0, :]

            step_out = env.step(action)
            # HIMLoco 返回 7 个值
            if len(step_out) >= 7:
                _, _, _, dones, infos, _, _ = step_out[:7]
            else:
                _, _, _, dones, infos = step_out[:5]

            if dones[0]:
                ep_info = infos.get('episode', {})
                ep_r = ep_info.get('r', [0])[0] if ep_info and 'r' in ep_info else '?'
                ep_l = ep_info.get('l', [0])[0] if ep_info and 'l' in ep_info else '?'
                print(f"Episode done: steps={ep_l}, reward={ep_r}")
                env.reset()

    except KeyboardInterrupt:
        print("\n[Play] Stopped")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='go2')
    parser.add_argument('--teacher_path', type=str, default=None,
                        help='教师权重路径 (用于初始化 HIM 估计器)')
    parser.add_argument('--checkpoint', type=str,
                        default='logs/go2_distill/student_ckpt_001350.pt')
    args_l = parser.parse_args()

    play(args_l, teacher_path=args_l.teacher_path,
         checkpoint_path=args_l.checkpoint)
