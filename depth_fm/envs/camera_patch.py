"""
给已有的 LeggedRobot 环境打深度相机补丁 (monkey-patch)。

用法:
    from depth_fm.envs.camera_patch import patch_depth_camera
    env = task_registry.make_env(name='go2', args=args)
    env = patch_depth_camera(env)
"""

import torch
import torchvision
import numpy as np
from isaacgym import gymapi, gymtorch


def patch_depth_camera(env, show_depth=False):
    """
    给 env 附加深度相机功能。

    在 env 对象上添加:
      - env.cam_handles           相机句柄列表
      - env.depth_buffer          [N, 2, H, W] 深度图缓冲区
      - env._depth_cfg            深度配置引用
      - env.update_depth_buffer()  渲染 + 更新 buffer
      - env.extras["depth"]        当前可用的 2 帧深度图

    Args:
        show_depth: True 时每 update_interval 步弹 OpenCV 窗口显示深度图

    调用后需设置 env.cfg.depth.use_camera = True
    """
    cfg = env.cfg

    if not hasattr(cfg, 'depth') or not cfg.depth.use_camera:
        return env

    env._show_depth = show_depth

    depth_cfg = cfg.depth
    env._depth_cfg = depth_cfg

    # ===== 深度缓冲区 =====
    H = depth_cfg.original[1] - depth_cfg.crop_bottom   # 58
    W = depth_cfg.original[0] - depth_cfg.crop_left - depth_cfg.crop_right  # 98
    env.depth_buffer = torch.zeros(
        env.num_envs, depth_cfg.buffer_len, H, W,
        device=env.device
    )
    env.extras["depth"] = None

    # ===== 为每个环境创建相机 =====
    camera_props = gymapi.CameraProperties()
    camera_props.width = depth_cfg.original[0]
    camera_props.height = depth_cfg.original[1]
    camera_props.horizontal_fov = depth_cfg.horizontal_fov
    camera_props.enable_tensors = True

    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(*depth_cfg.position)

    env.cam_handles = []
    for i, env_handle in enumerate(env.envs):
        angle = np.random.uniform(depth_cfg.angle[0], depth_cfg.angle[1])
        local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(angle), 0)

        cam = env.gym.create_camera_sensor(env_handle, camera_props)
        root_handle = env.gym.get_actor_root_rigid_body_handle(
            env_handle, env.actor_handles[i]
        )
        env.gym.attach_camera_to_body(
            cam, env_handle, root_handle,
            local_transform, gymapi.FOLLOW_TRANSFORM,
        )
        env.cam_handles.append(cam)

    print(f"[CameraPatch] {len(env.cam_handles)} cameras created "
          f"({camera_props.width}x{camera_props.height})")

    # ===== 核心方法: 更新深度 buffer =====
    def update_depth_buffer():
        if env.common_step_counter % depth_cfg.update_interval != 0:
            return

        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_tensors(env.sim)

        for i in range(env.num_envs):
            depth_ptr = env.gym.get_camera_image_gpu_tensor(
                env.sim, env.envs[i], env.cam_handles[i], gymapi.IMAGE_DEPTH
            )
            depth = gymtorch.wrap_tensor(depth_ptr)

            # 预处理: 裁剪 + 裁剪深度值 + 归一化
            depth = depth[
                depth_cfg.crop_top : camera_props.height - depth_cfg.crop_bottom,
                depth_cfg.crop_left : camera_props.width - depth_cfg.crop_right,
            ]
            depth += depth_cfg.dis_noise * 2 * (torch.rand(1, device=env.device) - 0.5)[0]
            depth = torch.clip(depth, -depth_cfg.far_clip, -depth_cfg.near_clip)
            # 归一化: [near, far] → [-0.5, 0.5]
            depth = -depth  # IsaacGym 深度为负
            depth = (depth - depth_cfg.near_clip) / (depth_cfg.far_clip - depth_cfg.near_clip) - 0.5

            init_flag = env.episode_length_buf[i] <= 1
            if init_flag:
                env.depth_buffer[i] = torch.stack([depth] * depth_cfg.buffer_len, dim=0)
            else:
                env.depth_buffer[i] = torch.cat(
                    [env.depth_buffer[i, 1:], depth.unsqueeze(0)], dim=0
                )

        env.gym.end_access_image_tensors(env.sim)

        # 调试: 显示首 env 的深度图
        if env._show_depth:
            try:
                import cv2
                vis = env.depth_buffer[0, -1].cpu().numpy()  # 首 env 最新帧
                vis = ((vis + 0.5) * 255).clip(0, 255).astype('uint8')
                cv2.imshow("Depth Camera [env 0]", vis)
                cv2.waitKey(1)
            except Exception:
                pass

    env._update_depth_buffer = update_depth_buffer

    # ===== 覆盖 post_physics_step =====
    _orig_post_physics_step = env.post_physics_step

    def post_physics_step_with_depth():
        update_depth_buffer()
        if env.depth_buffer is not None:
            env.extras["depth"] = env.depth_buffer[:, -2:]  # [N, 2, 58, 98]
        return _orig_post_physics_step()

    env.post_physics_step = post_physics_step_with_depth

    return env
