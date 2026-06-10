"""
DepthCameraRobot: 继承 LeggedRobot，添加深度相机 + scandot 导出。

用法:
  在 train_student.py 中:
    from depth_fm.envs.depth_robot import DepthCameraRobot
    env = DepthCameraRobot(cfg, sim_params, physics_engine, sim_device, headless)
"""

import torch
import torchvision
import numpy as np
from isaacgym import gymapi, gymtorch
from legged_gym.envs.base.legged_robot import LeggedRobot


class DepthCameraRobot(LeggedRobot):
    """
    扩展 LeggedRobot，支持:
      - 仿真深度相机 (106×60 → 裁剪 98×58)
      - 深度图 buffer (2帧堆叠)
      - Scandot 高程采样导出 (187点)
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # ===== 深度配置解析 =====
        self._has_depth = getattr(cfg, 'depth', None) is not None

        if self._has_depth:
            self.depth_cfg = cfg.depth
            self.cam_handles = []

        # 调用父类 __init__ (其中会调用 _init_buffers, 重置等)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        """扩展父类 buffer 初始化，添加深度 buffer"""
        super()._init_buffers()

        if self._has_depth:
            self.resize_transform = torchvision.transforms.Resize(
                (self.depth_cfg.original[1], self.depth_cfg.original[0]),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )

            # Depth buffer: [num_envs, buffer_len, H, W]
            self.depth_buffer = torch.zeros(
                self.num_envs,
                self.depth_cfg.buffer_len,
                self.depth_cfg.original[1] - self.depth_cfg.crop_bottom,
                self.depth_cfg.original[0] - self.depth_cfg.crop_left - self.depth_cfg.crop_right,
                device=self.device,
            )

            self.extras["depth"] = None

    # ================================================================
    # 深度相机创建
    # ================================================================

    def _create_envs(self):
        """覆盖父类，在创建每个环境时附加深度相机"""
        super()._create_envs()

        if self._has_depth:
            self._create_depth_cameras()

    def _create_depth_cameras(self):
        """为所有环境创建深度相机传感器"""
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.depth_cfg.original[0]     # 106
        camera_props.height = self.depth_cfg.original[1]    # 60
        camera_props.horizontal_fov = self.depth_cfg.horizontal_fov
        camera_props.enable_tensors = True

        # 相机安装在 base link 前方
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*self.depth_cfg.position)

        self.cam_handles = []
        for i, env_handle in enumerate(self.envs):
            # 域随机化: 每环境随机下倾角
            angle = np.random.uniform(
                self.depth_cfg.angle[0], self.depth_cfg.angle[1]
            )
            local_transform.r = gymapi.Quat.from_euler_zyx(
                0, np.radians(angle), 0
            )

            cam = self.gym.create_camera_sensor(env_handle, camera_props)
            # attach_camera_to_body 需要 rigid body handle，而非 actor handle
            root_handle = self.gym.get_actor_root_rigid_body_handle(
                env_handle, self.actor_handles[i]
            )
            self.gym.attach_camera_to_body(
                cam, env_handle,
                root_handle,
                local_transform,
                gymapi.FOLLOW_TRANSFORM,
            )
            self.cam_handles.append(cam)

        print(f"[DepthCameraRobot] {len(self.cam_handles)} cameras created "
              f"({camera_props.width}×{camera_props.height})")

    # ================================================================
    # 深度图处理
    # ================================================================

    def crop_depth_image(self, depth_image: torch.Tensor) -> torch.Tensor:
        """裁剪深度图: 底部 crop_bottom px, 左右各 crop_left/crop_right px"""
        cfg = self.depth_cfg
        h, w = depth_image.shape
        return depth_image[
            cfg.crop_top : h - cfg.crop_bottom,
            cfg.crop_left : w - cfg.crop_right,
        ]

    def normalize_depth_image(self, depth_image: torch.Tensor) -> torch.Tensor:
        """归一化到 [-0.5, 0.5]"""
        depth_image = depth_image * -1   # IsaacGym 深度是负的
        depth_image = (
            (depth_image - self.depth_cfg.near_clip)
            / (self.depth_cfg.far_clip - self.depth_cfg.near_clip)
            - 0.5
        )
        return depth_image

    def process_depth_image(
        self, depth_image: torch.Tensor, env_id: int
    ) -> torch.Tensor:
        """完整的深度图预处理管线"""
        # 裁剪
        depth_image = self.crop_depth_image(depth_image)
        # 噪声 (域随机化)
        depth_image += (
            self.depth_cfg.dis_noise * 2 * (torch.rand(1) - 0.5)[0]
        )
        # 裁剪深度值
        depth_image = torch.clip(
            depth_image, -self.depth_cfg.far_clip, -self.depth_cfg.near_clip
        )
        # 归一化
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    # ================================================================
    # 深度缓冲区更新
    # ================================================================

    def update_depth_buffer(self):
        """渲染深度相机并更新深度缓冲区 (每 update_interval 步调用一次)"""
        if not self._has_depth:
            return

        if self.common_step_counter % self.depth_cfg.update_interval != 0:
            return

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_img_ptr = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_DEPTH
            )
            depth_img = gymtorch.wrap_tensor(depth_img_ptr)
            depth_img = self.process_depth_image(depth_img, i)

            # 首帧: 复制填满 buffer
            init_flag = self.episode_length_buf[i] <= 1
            if init_flag:
                self.depth_buffer[i] = torch.stack(
                    [depth_img] * self.depth_cfg.buffer_len, dim=0
                )
            else:
                self.depth_buffer[i] = torch.cat(
                    [self.depth_buffer[i, 1:], depth_img.unsqueeze(0)], dim=0
                )

        self.gym.end_access_image_tensors(self.sim)

    # ================================================================
    # Scandot 导出
    # ================================================================

    def get_scandot_observations(self) -> torch.Tensor:
        """
        返回 scandots: [num_envs, 187]

        这些是 Critic 特权观测中的高程采样点部分。
        格式与 legged_robot_config 中 measured_points 一致:
          187 = 17 (x方向) × 11 (y方向)
        """
        if self.privileged_obs_buf is None:
            return torch.zeros(self.num_envs, 187, device=self.device)

        # 特权 obs 结构:
        # [one_step_obs(45) | base_lin_vel(3) | external_forces(3) | scandots(187)]
        scandot_start = 45 + 3 + 3   # 51
        return self.privileged_obs_buf[:, scandot_start:scandot_start + 187]

    # ================================================================
    # 生命周期钩子
    # ================================================================

    def post_physics_step(self):
        """覆盖: 在物理步后更新深度 buffer"""
        # 更新深度
        self.update_depth_buffer()

        # 更新 extras
        if self._has_depth and self.depth_buffer is not None:
            # 取 buffer 中最后一帧 (最新) 的 2-frame 堆叠
            self.extras["depth"] = self.depth_buffer[:, -2:]
        else:
            self.extras["depth"] = None

        # 调用父类 post_physics_step
        return super().post_physics_step()
