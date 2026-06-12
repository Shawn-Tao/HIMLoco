"""
Go2 蒸馏配置: 覆盖 HIMLoco 基础配置，添加深度相机 + 蒸馏参数。

Phase 1 (教师训练):  使用 go2_config.py + train_teacher.py
Phase 2 (学生蒸馏):  使用本配置 + train_student.py
"""

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


class GO2DistillCfg(GO2RoughCfg):
    """Go2 蒸馏环境配置"""

    class depth:
        """深度相机配置"""
        use_camera = True                       # 开启深度相机

        # 仿真渲染
        original = (106, 60)                    # IsaacGym 仿真相机分辨率
        # 裁剪: 底部2px, 左右各4px
        crop_top = 0
        crop_bottom = 2
        crop_left = 4
        crop_right = 4
        # 裁剪后: 98×58 — 直接输入 CNN (无需 resize)

        horizontal_fov = 87                     # D435i HFOV
        buffer_len = 2                          # 2帧堆叠
        update_interval = 5                     # 每5个控制步更新(10Hz)

        # 深度范围
        near_clip = 0.0                         # 0m
        far_clip = 2.0                          # 2m

        # 域随机化
        dis_noise = 0.005                       # 深度噪声
        latency_depth = 0.08                    # 深度延迟(s)
        latency_prop = 0.016                    # 本体延迟(s)

        # 相机安装 (Go2)
        position = [0.30, 0.0, 0.35]            # 你的实际安装位置
        angle = [-3, 3]                         # 下倾角随机化范围(度)

        scale = 1
        invert = True

    class scandot:
        """Scandot 采样点配置 (用于教师特权信息)"""
        num_points = 187                        # 17×11 网格
        grid_h = 11                             # 行数 (前后方向)
        grid_w = 17                             # 列数 (左右方向)
        # 采样范围 (米, 相对机器人中心)
        x_range = [-0.8, 0.8]                   # 前后
        y_range = [-0.5, 0.5]                   # 左右
        latent_dim = 32                         # 输出隐变量维度

    class terrain(GO2RoughCfg.terrain):
        measure_heights = True
        # Scandot 采样网格 (与 legged_robot_config.py 中的 measured_points 对齐)
        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
        ]  # 17 points
        measured_points_y = [
            -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5
        ]  # 11 points
        # 地形类型 (蒸馏时可以只用简单地形)
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]

    class env(GO2RoughCfg.env):
        num_envs = 4096                        # RTX Titan 24GB 可以跑 4096
        num_one_step_observations = 45
        num_observations = 270                 # 45 × 6
        # 特权 obs 包含 scandots
        num_one_step_privileged_obs = 45 + 3 + 3 + 187  # 238
        num_privileged_obs = num_one_step_privileged_obs

    class noise(GO2RoughCfg.noise):
        add_noise = True
        noise_level = 1.0


class GO2DistillCfgPPO(GO2RoughCfgPPO):
    """蒸馏算法配置 (虽然不是 PPO，但沿用了 legged_gym 的配置结构)"""

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 64
        learning_rate = 1e-3
        max_grad_norm = 1.0

    class runner:
        policy_class_name = 'HIMDistillModel'
        algorithm_class_name = 'FMDistillation'
        num_steps_per_env = 8     # 蒸馏用短 rollout
        max_iterations = 20000

        experiment_name = 'go2_distill'
        run_name = ''

        save_interval = 50
        resume = False


# ============================================================
# 蒸馏模型超参数
# ============================================================

class DistillModelCfg:
    """蒸馏模型架构超参数"""

    # Teacher
    # Phase 1 训练的输出 (相对于 HIMLoco 根目录)
    teacher_ckpt_path = 'logs/rough_go2'  # 目录，自动找最新的 model_*.pt

    # HIM
    num_actor_obs = 270
    num_critic_obs = 238
    num_one_step_obs = 45
    num_actions = 12
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]

    # Scandot
    num_scandots = 187
    scandot_grid_h = 11
    scandot_grid_w = 17
    scandot_latent_dim = 32

    # Depth
    depth_num_frames = 2
    depth_height = 58
    depth_width = 98
    depth_latent_dim = 32

    # FM Diffusion
    # NOTE: 蒸馏阶段 H=1（单步动作蒸馏），后续 RL 微调时可用 H=10
    fm_horizon = 1
    fm_hidden_dim = 256
    fm_num_steps_infer = 5          # 推理时欧拉步数
    fm_num_steps_train = 5

    # Distillation
    learning_rate = 1e-3
    max_grad_norm = 1.0
    latent_loss_coef = 0.1           # 隐空间对齐系数
    action_loss_coef = 0.0           # 行为克隆系数 (先用0，纯隐空间对齐)
    num_epochs_per_update = 5
    num_mini_batches = 64        # 2048×24=49152 样本, 每批~768 样本, 避免 OOM
