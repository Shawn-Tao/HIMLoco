"""
学生推理脚本: 加载蒸馏训练好的学生权重，在仿真中回放。

学生模型 = HIMEstimator(冻结) + DepthEncoder + FMDiffusionPolicy

用法:
    python legged_gym/legged_gym/scripts/play_student.py \
        --task=go2 \
        --checkpoint=./logs/go2_distill/student_final.pt
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


def play(args, checkpoint_path: str = None):
    """
    加载学生模型并在仿真中回放。
    """

    # 1. 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    if hasattr(env.cfg, 'depth'):
        env.cfg.depth.use_camera = True

    device = env.device if hasattr(env, 'device') else 'cuda:0'

    # 2. 创建蒸馏模型结构
    cfg = DistillModelCfg()

    model = HIMDistillModel(
        teacher_ckpt_path=None,
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

    # 3. 加载学生权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.depth_encoder.load_state_dict(ckpt['depth_encoder'])
        model.fm_policy.load_state_dict(ckpt['fm_policy'])
        # 也尝试加载 HIM Estimator (如果 checkpoint 中有)
        if 'him_estimator' in ckpt:
            model.him_estimator.load_state_dict(ckpt['him_estimator'])
        print(f"[Play] Student weights loaded from {checkpoint_path}")
    else:
        print(f"[Play] WARNING: No checkpoint at {checkpoint_path}")
        print(f"[Play] Using random weights — robot WILL fall!")

    model.eval_all()

    # 4. 主循环
    env.reset()
    step = 0

    print("\n[Play] Running... Press Ctrl+C to stop.\n")

    try:
        while True:
            obs_history = env.obs_buf                # [num_envs, 270]
            depth = env.extras.get('depth', None)    # [num_envs, 2, 58, 98]

            if depth is not None and depth.shape[0] > 0:
                action = model.act_student(
                    obs_history[:1], depth[:1], num_steps=5
                )
            else:
                # 深度未就绪时用盲态回退
                _, him_z = model.him_estimator(obs_history[:1])
                dummy_cond = torch.cat([
                    obs_history[:1, :cfg.num_one_step_obs],
                    torch.zeros(1, 3, device=device),
                    him_z,
                    torch.zeros(1, cfg.depth_latent_dim, device=device),
                ], dim=-1)
                action = model.fm_policy.act_inference(dummy_cond, num_steps=5)

            step_out = env.step(action.unsqueeze(0))
            if len(step_out) == 7:
                obs, _, _, dones, infos, _, _ = step_out
            else:
                obs, _, _, dones, infos = step_out

            step += 1

            if dones[0]:
                ep_info = infos.get('episode', {})
                ep_r = ep_info.get('r', [0])[0]
                ep_l = ep_info.get('l', [0])[0]
                print(f"Episode done: steps={ep_l}, reward={ep_r:.1f}")
                env.reset()
                step = 0

    except KeyboardInterrupt:
        print(f"\n[Play] Stopped by user")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='go2')
    parser.add_argument('--checkpoint', type=str,
                        default='./logs/go2_distill/student_final.pt')
    parser.add_argument('--headless', action='store_true', default=False)

    args = parser.parse_args()
    play(args, checkpoint_path=args.checkpoint)
