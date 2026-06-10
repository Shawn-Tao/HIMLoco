"""
深度相机渲染测试 Demo

独立脚本，不依赖 depth_fm 模块，直接测试:
  1. 仿真相机创建  (106×60, HFOV=87°)
  2. 深度图渲染
  3. 裁剪 (底部2px, 左右各4px → 98×58)
  4. 归一化 [-0.5, 0.5]
  5. 保存 sample 图像供检查

用法:
    python test_depth_camera.py [--iters 100] [--save]
"""

import numpy as np
import os
import sys
import argparse

# ============================================================
# 配置 (与 depth_fm/configs/go2_distill_config.py 一致)
# ============================================================

class DepthConfig:
    original = (106, 60)          # 仿真相机分辨率
    crop_top = 0
    crop_bottom = 2
    crop_left = 4
    crop_right = 4
    # 裁剪后: 98×58

    horizontal_fov = 87           # D435i HFOV
    buffer_len = 2                # 帧堆叠
    update_interval = 5           # 更新间隔
    near_clip = 0.0               # 最近深度 (m)
    far_clip = 2.0                # 最远深度 (m)
    dis_noise = 0.0               # 深度噪声 (测试时关闭)
    position = (0.30, 0.0, 0.35)  # 相机安装位置 (Go2 base link)
    angle = (-3, 3)               # 下倾角范围
    scale = 1
    invert = True


# ============================================================
# 核心: 深度相机创建 + 渲染 + 处理
# ============================================================

def setup_depth_camera(gym, sim, env_handle, actor_handle, cfg):
    """
    在一个环境中创建并挂载深度相机。

    Returns:
        cam_handle: 相机句柄
    """
    camera_props = gymapi.CameraProperties()
    camera_props.width = cfg.original[0]
    camera_props.height = cfg.original[1]
    camera_props.horizontal_fov = cfg.horizontal_fov
    camera_props.enable_tensors = True

    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(*cfg.position)
    # 下倾角 (正 = 向下看)
    angle = np.random.uniform(cfg.angle[0], cfg.angle[1])
    local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(angle), 0)

    cam_handle = gym.create_camera_sensor(env_handle, camera_props)
    gym.attach_camera_to_body(
        cam_handle, env_handle, actor_handle,
        local_transform, gymapi.FOLLOW_TRANSFORM,
    )

    print(f"  Camera: {camera_props.width}×{camera_props.height}, "
          f"HFOV={camera_props.horizontal_fov}°, "
          f"position={cfg.position}, tilt={angle:.1f}°")

    return cam_handle


def capture_and_process(gym, sim, env_handle, cam_handle, cfg):
    """
    渲染一帧深度图并完成预处理管线。

    Returns:
        raw:      原始深度图 numpy [H, W]
        cropped:  裁剪后 numpy [H_crop, W_crop]
        normalized: 归一化后 numpy [H_crop, W_crop]  (范围 [-0.5, 0.5])
    """
    import torch
    from isaacgym import gymtorch

    # 1. 渲染
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    # 2. 获取深度图 tensor
    depth_ptr = gym.get_camera_image_gpu_tensor(
        sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
    )
    depth = gymtorch.wrap_tensor(depth_ptr).clone()

    gym.end_access_image_tensors(sim)

    # 3. 转 numpy 并处理负号 (IsaacGym 深度是负的)
    depth_np = -depth.cpu().numpy().astype(np.float32)

    # 4. 记录原始
    raw = depth_np.copy()

    # 5. 裁剪
    h, w = depth_np.shape
    cropped = depth_np[
        cfg.crop_top : h - cfg.crop_bottom,
        cfg.crop_left : w - cfg.crop_right,
    ]

    # 6. 裁剪深度值
    cropped = np.clip(cropped, cfg.near_clip, cfg.far_clip)

    # 7. 归一化
    normalized = (cropped - cfg.near_clip) / (cfg.far_clip - cfg.near_clip) - 0.5

    return raw, cropped, normalized


# ============================================================
# 可视化 (无 headless 时用 OpenCV 显示)
# ============================================================

def visualize(raw, cropped, normalized, step):
    """用 OpenCV 显示三张深度图"""
    try:
        import cv2

        # 转可视范围 [0, 255]
        def to_vis(img, vmin=0, vmax=2.0):
            vis = np.clip(img, vmin, vmax)
            vis = (vis - vmin) / (vmax - vmin) * 255
            return vis.astype(np.uint8)

        # 叠加裁剪线
        raw_vis = cv2.cvtColor(to_vis(raw), cv2.COLOR_GRAY2BGR)
        cfg = DepthConfig
        # 画裁剪线 (红色)
        h, w = raw.shape
        cv2.rectangle(raw_vis,
                      (cfg.crop_left, cfg.crop_top),
                      (w - cfg.crop_right - 1, h - cfg.crop_bottom - 1),
                      (0, 0, 255), 1)

        # 标注
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = f"{raw.shape[1]}x{raw.shape[0]}"
        crp = f"{cropped.shape[1]}x{cropped.shape[0]}"
        cv2.putText(raw_vis, f"Raw {org}", (5, 15), font, 0.4, (0, 255, 0), 1)

        cropped_vis = to_vis(cropped)
        cv2.putText(cropped_vis, f"Cropped {crp}", (5, 15), font, 0.4, 255, 1)

        # 归一化图: [-0.5, 0.5] → [0, 255]
        norm_vis = ((normalized + 0.5) * 255).astype(np.uint8)
        cv2.putText(norm_vis, f"Norm [-0.5,0.5]", (5, 15), font, 0.4, 255, 1)

        # 拼接
        row1 = np.hstack([raw_vis, cv2.cvtColor(cropped_vis, cv2.COLOR_GRAY2BGR)])
        row2 = np.hstack([cv2.cvtColor(norm_vis, cv2.COLOR_GRAY2BGR),
                          np.zeros_like(cv2.cvtColor(norm_vis, cv2.COLOR_GRAY2BGR))])
        display = np.vstack([row1, row2])

        cv2.imshow("Depth Camera Test | Raw → Crop → Normalize", display)
        cv2.waitKey(1)

    except ImportError:
        pass  # 无 OpenCV 时跳过可视化


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=200,
                        help='测试步数')
    parser.add_argument('--save', action='store_true',
                        help='保存样本深度图到文件')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='无头模式 (不显示窗口)')
    parser.add_argument('--gui', action='store_false', dest='headless',
                        help='GUI 模式 (显示 IsaacGym 窗口)')
    args = parser.parse_args()

    from isaacgym import gymapi, gymutil

    cfg = DepthConfig()

    # ============================================================
    # 1. 初始化 Isaac Gym
    # ============================================================
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 0.005
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
    sim_params.physx.num_threads = 4
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0

    sim_params.use_gpu_pipeline = True
    sim_device = 'cuda:0'
    graphics_device = 0 if not args.headless else -1
    physics_engine = gymapi.SIM_PHYSX

    sim = gym.create_sim(
        0, 0, physics_engine, sim_params
    )
    if sim is None:
        print("ERROR: Failed to create sim")
        return

    # ============================================================
    # 2. 加载地面
    # ============================================================
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    gym.add_ground(sim, plane_params)

    # ============================================================
    # 3. 加载 Go2 机器人
    # ============================================================
    # URDF 路径 (相对于 HIMLoco 根目录)
    himloco_root = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(
        himloco_root, 'legged_gym', 'resources', 'robots', 'go2', 'urdf', 'go2.urdf'
    )
    print(f"Loading URDF: {urdf_path}")
    assert os.path.exists(urdf_path), f"URDF not found: {urdf_path}"

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = True
    asset_options.fix_base_link = False
    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01
    asset_options.disable_gravity = False

    robot_asset = gym.load_asset(sim, '', urdf_path, asset_options)

    # ============================================================
    # 4. 创建环境 + 放置机器人
    # ============================================================
    env_spacing = 3.0
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    env_handle = gym.create_env(sim, env_lower, env_upper, 1)

    # 机器人初始位姿
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.42)  # Go2 站立高度

    # PD 增益
    dof_props = gym.get_asset_dof_properties(robot_asset)
    dof_props['driveMode'] = gymapi.DOF_MODE_POS
    stiffness = 40.0
    damping = 1.0
    for i in range(len(dof_props)):
        dof_props['stiffness'][i] = stiffness
        dof_props['damping'][i] = damping

    actor_handle = gym.create_actor(
        env_handle, robot_asset, pose, 'go2', 0, 1
    )
    gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

    # 设置默认关节角度 (站立姿态)
    default_angles = np.array([
        0.1, 0.8, -1.5,    # FL: hip, thigh, calf
        -0.1, 0.8, -1.5,   # FR
        0.1, 1.0, -1.5,    # RL
        -0.1, 1.0, -1.5,   # RR
    ], dtype=np.float32)
    dof_states = np.zeros(len(default_angles) * 2, dtype=np.float32)
    dof_states[::2] = default_angles  # 位置
    gym.set_actor_dof_states(env_handle, actor_handle, dof_states, gymapi.STATE_ALL)

    # ============================================================
    # 5. 创建深度相机
    # ============================================================
    print("\n[Setup] Creating depth camera...")
    cam_handle = setup_depth_camera(gym, sim, env_handle, actor_handle, cfg)

    # ============================================================
    # 6. 主循环: 仿真 + 渲染深度图
    # ============================================================
    print(f"\n[Test] Running {args.iters} iterations...\n")

    depth_stats = {'min': [], 'max': [], 'mean': [], 'valid_ratio': []}

    for step in range(args.iters):
        # 仿真一步
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)

        # 每 5 步渲染一次深度 (与配置一致)
        if step % cfg.update_interval == 0:
            raw, cropped, normalized = capture_and_process(
                gym, sim, env_handle, cam_handle, cfg
            )

            # 统计
            depth_stats['min'].append(cropped.min())
            depth_stats['max'].append(cropped.max())
            depth_stats['mean'].append(cropped.mean())
            valid = (cropped > cfg.near_clip) & (cropped < cfg.far_clip)
            depth_stats['valid_ratio'].append(valid.mean())

            # 可视化
            if not args.headless:
                visualize(raw, cropped, normalized, step)

            # 日志
            if step % 50 == 0:
                print(f"  Step {step:4d}: depth range=[{cropped.min():.3f}, "
                      f"{cropped.max():.3f}]m, valid={valid.mean():.1%}, "
                      f"shape raw={raw.shape}, crop={cropped.shape}")

    # ============================================================
    # 7. 保存样本深度图
    # ============================================================
    if args.save:
        import cv2
        out_dir = os.path.join(himloco_root, 'depth_test_output')
        os.makedirs(out_dir, exist_ok=True)

        # 最后一帧
        raw_vis = ((np.clip(raw, 0, 2.0) / 2.0) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, '01_raw.png'), raw_vis)

        crop_vis = ((np.clip(cropped, 0, 2.0) / 2.0) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, '02_cropped.png'), crop_vis)

        norm_vis = ((normalized + 0.5) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, '03_normalized.png'), norm_vis)

        print(f"\n[Save] Images saved to {out_dir}/")
        print(f"       01_raw.png       — 原始 106×60")
        print(f"       02_cropped.png   — 裁剪后 98×58")
        print(f"       03_normalized.png — 归一化 [-0.5, 0.5]")

    # ============================================================
    # 8. 统计摘要
    # ============================================================
    print(f"\n{'='*50}")
    print(f"[Summary] Depth Camera Test Results")
    print(f"{'='*50}")
    print(f"  Resolution:      {cfg.original[0]}×{cfg.original[1]} → "
          f"{cfg.original[0]-cfg.crop_left-cfg.crop_right}×"
          f"{cfg.original[1]-cfg.crop_top-cfg.crop_bottom}")
    print(f"  HFOV:            {cfg.horizontal_fov}°")
    print(f"  Depth range:     [{np.mean(depth_stats['min']):.3f}, "
          f"{np.mean(depth_stats['max']):.3f}]m")
    print(f"  Valid ratio:     {np.mean(depth_stats['valid_ratio']):.1%}")
    print(f"  Camera position: {cfg.position}")
    print(f"  Test iterations: {args.iters}")
    print(f"{'='*50}")

    # 清理
    gym.destroy_sim(sim)

    # 检查: 深度图是否全黑或全白
    if np.mean(depth_stats['valid_ratio']) < 0.1:
        print("\n⚠️  WARNING: Very few valid depth pixels. Check:")
        print("   - Camera FOV and position")
        print("   - near_clip / far_clip range")
    elif np.mean(depth_stats['max']) < 0.1:
        print("\n⚠️  WARNING: Depth values too small. "
              "Camera might be pointing at empty space.")
    else:
        print("\n✓ Depth camera working correctly!")


if __name__ == '__main__':
    main()
