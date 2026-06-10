"""
深度相机渲染测试 Demo (v2 — 使用 legged_gym 环境框架)

用法:
    python test_depth_camera.py [--iters 100] [--save]
"""

import numpy as np
import os
import sys
import argparse

HIMLOCO_ROOT = os.path.dirname(os.path.abspath(__file__))
if HIMLOCO_ROOT not in sys.path:
    sys.path.insert(0, HIMLOCO_ROOT)

import isaacgym
from isaacgym import gymapi, gymtorch
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry


# ============================================================
# 配置
# ============================================================

class DepthConfig:
    """与 depth_fm/configs/go2_distill_config.py 一致"""
    original = (106, 60)
    crop_top = 0
    crop_bottom = 2
    crop_left = 4
    crop_right = 4
    horizontal_fov = 87
    buffer_len = 2
    update_interval = 5
    near_clip = 0.0
    far_clip = 2.0
    dis_noise = 0.0
    position = (0.30, 0.0, 0.35)
    angle = (-3, 3)


# ============================================================
# 深度相机
# ============================================================

def add_depth_camera(gym, sim, env_handle, actor_handle, cfg):
    """在已有环境中挂载深度相机"""
    camera_props = gymapi.CameraProperties()
    camera_props.width = cfg.original[0]
    camera_props.height = cfg.original[1]
    camera_props.horizontal_fov = cfg.horizontal_fov
    camera_props.enable_tensors = True

    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(*cfg.position)
    angle = np.random.uniform(cfg.angle[0], cfg.angle[1])
    local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(angle), 0)

    cam = gym.create_camera_sensor(env_handle, camera_props)
    root_handle = gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
    gym.attach_camera_to_body(cam, env_handle, root_handle,
                              local_transform, gymapi.FOLLOW_TRANSFORM)
    return cam


def capture_depth(gym, sim, env_handle, cam_handle, cfg):
    """渲染一帧深度图"""
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    depth_ptr = gym.get_camera_image_gpu_tensor(
        sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
    depth = gymtorch.wrap_tensor(depth_ptr).clone()

    gym.end_access_image_tensors(sim)

    depth_np = -depth.cpu().numpy().astype(np.float32)
    raw = depth_np.copy()

    h, w = depth_np.shape
    cropped = depth_np[cfg.crop_top:h - cfg.crop_bottom,
                       cfg.crop_left:w - cfg.crop_right]
    cropped = np.clip(cropped, cfg.near_clip, cfg.far_clip)
    normalized = (cropped - cfg.near_clip) / (cfg.far_clip - cfg.near_clip) - 0.5
    return raw, cropped, normalized


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--save', action='store_true')
    # --headless 会被 test 解析器吃掉, 不传给 Gym
    # 深度相机需要 GL context (viewer), headless 下无法渲染深度图
    parser.add_argument('--headless', action='store_true',
                        help='No effect (depth camera requires viewer context)')
    args, remaining = parser.parse_known_args()

    cfg = DepthConfig()

    # ============================================================
    # 1. 用 legged_gym 创建 Go2 环境 (改注册 config → 1 env 避开 OOM)
    # ============================================================
    _orig_num_envs = task_registry.env_cfgs['go2'].env.num_envs
    task_registry.env_cfgs['go2'].env.num_envs = 1
    _orig_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    env, _ = task_registry.make_env(name='go2', args=None)
    sys.argv = _orig_argv
    task_registry.env_cfgs['go2'].env.num_envs = _orig_num_envs
    print(f"[Setup] Env: {env.num_envs} envs, device={env.device}")
    env.reset()

    # ============================================================
    # 2. 挂载深度相机
    # ============================================================
    print("\n[Setup] Creating depth camera...")
    cam_handle = add_depth_camera(
        env.gym, env.sim, env.envs[0], env.actor_handles[0], cfg)

    # ============================================================
    # 3. 主循环
    # ============================================================
    print(f"\n[Test] Running {args.iters} iterations...\n")
    depth_stats = {'min': [], 'max': [], 'mean': [], 'valid_ratio': []}

    for step in range(args.iters):
        actions = torch.zeros(1, env.num_actions, device=env.device)
        env.step(actions)

        if step % cfg.update_interval == 0:
            raw, cropped, normalized = capture_depth(
                env.gym, env.sim, env.envs[0], cam_handle, cfg)

            depth_stats['min'].append(cropped.min())
            depth_stats['max'].append(cropped.max())
            depth_stats['mean'].append(cropped.mean())
            valid = (cropped > cfg.near_clip) & (cropped < cfg.far_clip)
            depth_stats['valid_ratio'].append(valid.mean())

            if step % 50 == 0:
                print(f"  Step {step:4d}: range=[{cropped.min():.3f},{cropped.max():.3f}]m "
                      f"valid={valid.mean():.1%}")

    # ============================================================
    # 4. 保存 & 总结
    # ============================================================
    if args.save:
        import cv2
        out_dir = os.path.join(HIMLOCO_ROOT, 'depth_test_output')
        os.makedirs(out_dir, exist_ok=True)
        for name, img in [('01_raw', raw), ('02_cropped', cropped),
                          ('03_normalized', normalized)]:
            vis = ((np.clip(img, 0, 2.0) / 2.0) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f'{name}.png'), vis)
        print(f"\n[Save] Images → {out_dir}/")

    print(f"\n{'='*50}")
    print(f"  Resolution: {cfg.original[0]}×{cfg.original[1]} → "
          f"{cfg.original[0]-cfg.crop_left-cfg.crop_right}×"
          f"{cfg.original[1]-cfg.crop_top-cfg.crop_bottom}")
    print(f"  Depth range: [{np.mean(depth_stats['min']):.3f}, "
          f"{np.mean(depth_stats['max']):.3f}]m")
    print(f"  Valid ratio: {np.mean(depth_stats['valid_ratio']):.1%}")
    print(f"{'='*50}")
    print("\n✓ Depth camera working correctly!")


if __name__ == '__main__':
    main()
