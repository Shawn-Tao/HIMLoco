"""
HIM Distillation Model: 教师(冻结) → 学生(可训练) 蒸馏框架。

教师:
  - HIMEstimator (冻结, 共享)
  - ScandotEncoder (冻结) → scandot_latent [32]
  - Actor MLP (冻结) → teacher_action [12]

学生:
  - HIMEstimator (冻结, 与教师共享)
  - DepthEncoder (可训练) → depth_latent [32]
  - FMDiffusionPolicy (可训练) → student_action [12]

蒸馏损失:
  L = L_FM(action, condition)                    ← Flow Matching 损失（主）
    + λ_latent * MSE(depth_latent, scandot_latent) ← 隐空间对齐
    + λ_action * MSE(student_action, teacher_action) ← 行为克隆（可选）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from rsl_rl.modules.him_actor_critic import HIMActorCritic
from rsl_rl.modules.him_estimator import HIMEstimator
from depth_fm.modules.scandot_encoder import ScandotEncoder
from depth_fm.modules.depth_encoder import DepthEncoder
from depth_fm.modules.fm_diffusion import FMDiffusionPolicy


class HIMDistillModel(nn.Module):
    """
    教师-学生蒸馏模型。

    训练时:
      - 教师全部冻结
      - HIM 估计器冻结（师生共享）
      - 只训练 DepthEncoder + FMDiffusionPolicy

    推理时:
      - 只用学生分支: HIM(冻结) + DepthEncoder + FMDiffusionPolicy
    """

    def __init__(
        self,
        # 教师参数
        teacher_ckpt_path: str,
        num_actor_obs: int = 270,
        num_critic_obs: int = 232,
        num_one_step_obs: int = 45,
        num_actions: int = 12,
        actor_hidden_dims: list = None,
        critic_hidden_dims: list = None,
        # Scandot
        num_scandots: int = 187,
        scandot_grid_h: int = 11,
        scandot_grid_w: int = 17,
        scandot_latent_dim: int = 32,
        # Depth
        depth_num_frames: int = 2,
        depth_height: int = 58,
        depth_width: int = 98,
        depth_latent_dim: int = 32,
        # FM Diffusion
        fm_horizon: int = 10,
        fm_hidden_dim: int = 256,
        fm_num_steps_infer: int = 5,
        # Distillation
        latent_loss_coef: float = 0.1,
        action_loss_coef: float = 0.0,
        device: str = 'cuda:0',
    ):
        super().__init__()
        self.device = device
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs
        self.latent_loss_coef = latent_loss_coef
        self.action_loss_coef = action_loss_coef

        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]

        # ================================================================
        # 教师组件 (全部冻结)
        # ================================================================

        # HIM Estimator (共享，冻结)
        self.him_estimator = HIMEstimator(
            temporal_steps=int(num_actor_obs / num_one_step_obs),
            num_one_step_obs=num_one_step_obs,
        )
        self.him_latent_dim = self.him_estimator.num_latent  # 16

        # Scandot Encoder (冻结)
        self.scandot_encoder = ScandotEncoder(
            num_points=num_scandots,
            grid_h=scandot_grid_h,
            grid_w=scandot_grid_w,
            latent_dim=scandot_latent_dim,
        )

        # Teacher Actor MLP — 与原 HIMLoco 一致，盲态输入
        # 输入: obs_current(45) + HIM_vel(3) + him_latent(16) = 64
        teacher_actor_input = num_one_step_obs + 3 + self.him_latent_dim

        teacher_actor_layers = []
        in_dim = teacher_actor_input
        for h in actor_hidden_dims:
            teacher_actor_layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        teacher_actor_layers.append(nn.Linear(in_dim, num_actions))
        self.teacher_actor = nn.Sequential(*teacher_actor_layers)

        # Teacher Critic
        teacher_critic_layers = []
        in_dim = num_critic_obs
        for h in critic_hidden_dims:
            teacher_critic_layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        teacher_critic_layers.append(nn.Linear(in_dim, 1))
        self.teacher_critic = nn.Sequential(*teacher_critic_layers)

        # 加载教师权重
        if teacher_ckpt_path:
            self._load_teacher(teacher_ckpt_path)

        # 冻结教师
        self._freeze_teacher()

        # ================================================================
        # 学生组件 (可训练)
        # ================================================================

        # Depth Encoder
        self.depth_encoder = DepthEncoder(
            num_frames=depth_num_frames,
            input_height=depth_height,
            input_width=depth_width,
            latent_dim=depth_latent_dim,
        )

        # FM Diffusion Policy 的条件维度:
        #   obs_current(45) + HIM_vel(3) + HIM_latent(16) + depth_latent(32)
        # = 96
        fm_cond_dim = num_one_step_obs + 3 + self.him_latent_dim + depth_latent_dim

        self.fm_policy = FMDiffusionPolicy(
            action_dim=num_actions,
            horizon=fm_horizon,
            cond_dim=fm_cond_dim,
            hidden_dim=fm_hidden_dim,
            num_steps_infer=fm_num_steps_infer,
        )

        # 将所有参数移到目标设备
        self.to(device)

    def _load_teacher(self, ckpt_path: str):
        """
        加载 HIMLoco 预训练教师权重。

        HIMLoco checkpoint 结构 (HIMActorCritic):
          estimator.encoder.*   →  him_estimator.encoder.*
          estimator.target.*    →  him_estimator.target.*
          estimator.proto.*     →  him_estimator.proto.*
          actor.*               →  teacher_actor.*
          critic.*              →  teacher_critic.*
          std                   →  (忽略)

        Scandot_encoder 不在原 checkpoint 中，随机初始化后由 latent_loss
        驱动 depth_encoder 去匹配它产生的表示。
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # 处理不同的 checkpoint 格式
        if 'model_state_dict' in ckpt:
            src = ckpt['model_state_dict']
        elif 'actor_critic_state_dict' in ckpt:
            src = ckpt['actor_critic_state_dict']
        else:
            src = ckpt

        # Key 映射: HIMActorCritic → HIMDistillModel
        key_map = {
            'estimator.': 'him_estimator.',
            'actor.':    'teacher_actor.',
            'critic.':   'teacher_critic.',
        }

        dst = {}
        matched = 0
        for old_key, val in src.items():
            new_key = old_key
            for prefix, replacement in key_map.items():
                if old_key.startswith(prefix):
                    new_key = replacement + old_key[len(prefix):]
                    matched += 1
                    break
            dst[new_key] = val

        # 加载 (strict=False, scandot_encoder 等新组件随机初始化)
        missing, unexpected = self.load_state_dict(dst, strict=False)

        # 筛选有意义的 missing keys (只关注教师组件)
        teacher_missing = [k for k in missing
                           if any(p in k for p in ['him_estimator', 'teacher_actor', 'teacher_critic'])
                           and 'scandot' not in k and 'depth' not in k and 'fm_' not in k]

        print(f"[HIMDistillModel] Loaded {matched}/{len(src)} teacher keys from {ckpt_path}")
        if teacher_missing:
            print(f"[HIMDistillModel] ⚠ Teacher keys NOT loaded ({len(teacher_missing)}):")
            for k in teacher_missing[:8]:
                print(f"    - {k}")
        if unexpected:
            print(f"[HIMDistillModel] ⚠ Unexpected keys in checkpoint ({len(unexpected)}), ignored")
        if not teacher_missing:
            print(f"[HIMDistillModel] ✓ Teacher fully loaded")

    def _freeze_teacher(self):
        """冻结所有教师参数"""
        for p in self.him_estimator.parameters():
            p.requires_grad = False
        for p in self.scandot_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_actor.parameters():
            p.requires_grad = False
        for p in self.teacher_critic.parameters():
            p.requires_grad = False

        self.him_estimator.eval()
        self.scandot_encoder.eval()
        self.teacher_actor.eval()
        self.teacher_critic.eval()

    def get_student_condition(
        self,
        obs_history: torch.Tensor,       # [B, 270]
        depth_images: torch.Tensor,      # [B, 2, 58, 98]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构造学生 FM 扩散策略的条件向量。

        condition = [obs_current(45) | him_vel(3) | him_latent(16) | depth_latent(32)]
                  = [B, 96]

        Returns:
            condition:    [B, 96]  FM 条件向量
            depth_latent: [B, 32]  深度隐变量 (复用，避免重复计算)
        """
        with torch.no_grad():
            vel, him_z = self.him_estimator(obs_history)
        depth_latent = self.depth_encoder(depth_images)
        obs_current = obs_history[:, :self.num_one_step_obs]
        condition = torch.cat([obs_current, vel, him_z, depth_latent], dim=-1)
        return condition, depth_latent

    @torch.no_grad()
    def get_teacher_action(
        self,
        obs_history: torch.Tensor,       # [B, 270]
        scandots: torch.Tensor,          # [B, 187]
    ) -> Dict[str, torch.Tensor]:
        """
        教师前向（无梯度）。

        Actor 输入 = obs_current(45) + him_vel(3) + him_z(16) = 64
        (与原 HIMLoco 盲态教师完全一致)

        Scandot 隐变量仅作为蒸馏目标，不参与教师动作计算。

        Returns:
            {
                'action':           [B, 12]  教师盲态动作
                'scandot_latent':   [B, 32]  scandot 隐变量（蒸馏目标）
            }
        """
        # HIM
        vel, him_z = self.him_estimator(obs_history)
        # Scandot → latent (仅用于蒸馏对齐)
        scandot_latent = self.scandot_encoder(scandots)
        # Actor（盲态，与原 HIMLoco 一致）
        obs_current = obs_history[:, :self.num_one_step_obs]
        actor_input = torch.cat([obs_current, vel, him_z], dim=-1)
        action = self.teacher_actor(actor_input)

        return {
            'action': action,
            'scandot_latent': scandot_latent,
        }

    def compute_distill_loss(
        self,
        obs_history: torch.Tensor,       # [B, 270]
        depth_images: torch.Tensor,      # [B, 2, 58, 98]
        scandots: torch.Tensor,          # [B, 187]
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失。

        Returns:
            {
                'fm_loss':          FM 扩散策略训练损失
                'latent_loss':      隐空间对齐损失 (MSE)
                'action_loss':      行为克隆损失 (MSE, 可选)
                'total_loss':       总损失
            }
        """
        # 教师前向（无梯度）
        teacher_out = self.get_teacher_action(obs_history, scandots)
        teacher_action = teacher_out['action']
        scandot_latent = teacher_out['scandot_latent']

        condition, depth_latent = self.get_student_condition(
            obs_history, depth_images
        )

        # ===== FM 扩散损失（主损失） =====
        fm_loss = self.fm_policy.compute_loss(teacher_action.unsqueeze(1), condition)
        # teacher_action: [B, 12] → unsqueeze → [B, 1, 12]
        # FM 内部 operate on [B, H=1, 12]

        # ===== 隐空间对齐损失 =====
        latent_loss = F.mse_loss(depth_latent, scandot_latent.detach())

        # ===== 行为克隆损失（可选，默认关闭） =====
        if self.action_loss_coef > 0:
            student_action = self.fm_policy.sample(condition, num_steps=1)
            action_loss = F.mse_loss(student_action.squeeze(1), teacher_action.detach())
        else:
            action_loss = torch.tensor(0.0, device=self.device)

        # ===== 总损失 =====
        total_loss = fm_loss + self.latent_loss_coef * latent_loss
        total_loss = total_loss + self.action_loss_coef * action_loss

        return {
            'fm_loss': fm_loss,
            'latent_loss': latent_loss,
            'action_loss': action_loss,
            'total_loss': total_loss,
        }

    @torch.no_grad()
    def act_student(
        self,
        obs_history: torch.Tensor,       # [B, 270]
        depth_images: torch.Tensor,      # [B, 2, 58, 98]
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        学生推理: 输出关节动作。

        Returns:
            action: [B, 12] (B=1 时 [12])
        """
        condition, _ = self.get_student_condition(obs_history, depth_images)
        action_seq = self.fm_policy.sample(condition, num_steps)
        return action_seq[:, 0, :]  # [B, 12] 取第一步

    def get_trainable_parameters(self):
        """返回学生可训练参数（给优化器）"""
        return list(self.depth_encoder.parameters()) + list(self.fm_policy.parameters())

    def train_student(self):
        """设置学生为训练模式，教师保持 eval"""
        self.depth_encoder.train()
        self.fm_policy.train()
        self.him_estimator.eval()
        self.scandot_encoder.eval()
        self.teacher_actor.eval()

    def eval_all(self):
        """全设为 eval 模式"""
        self.depth_encoder.eval()
        self.fm_policy.eval()
        self.him_estimator.eval()
        self.scandot_encoder.eval()
        self.teacher_actor.eval()
