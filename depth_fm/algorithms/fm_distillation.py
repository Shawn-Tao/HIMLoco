"""
FM Distillation Algorithm: 教师→学生 蒸馏训练主循环。

训练流程:
  1. 环境 step: 学生输出动作，环境推进
  2. 收集 rollout: (obs_history, depth_images, scandots, teacher_action)
  3. 蒸馏更新: FM loss + latent alignment loss
  4. 迭代至收敛

与标准 RL 不同: 这是 DAgger-style 蒸馏——学生在环境中执行动作，
但同时学习模仿教师的输出和隐空间。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np
import os
import time


class FMDistillation:
    """
    Flow Matching 蒸馏训练器。

    损失 = FM_loss + λ_latent * latent_loss + λ_action * action_loss

    优化目标:
      - FM_loss: 学生扩散策略学习条件动作分布 p(a|c)
      - latent_loss: 学生 depth 隐空间对齐教师 scandot 隐空间
      - action_loss: (可选) 学生输出 直接模仿教师动作
    """

    def __init__(
        self,
        distill_model,             # HIMDistillModel
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        num_epochs_per_update: int = 5,
        num_mini_batches: int = 4,
        latent_loss_coef: float = 0.1,
        action_loss_coef: float = 0.0,
        device: str = 'cuda:0',
        log_dir: str = None,
    ):
        self.model = distill_model
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.num_epochs_per_update = num_epochs_per_update
        self.num_mini_batches = num_mini_batches

        # 更新模型中的系数
        self.model.latent_loss_coef = latent_loss_coef
        self.model.action_loss_coef = action_loss_coef

        # 只优化学生参数
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate)

        # Logging
        self.writer = None
        if log_dir:
            self.writer = SummaryWriter(log_dir)

        # Metrics
        self.metrics = {
            'fm_loss': deque(maxlen=100),
            'latent_loss': deque(maxlen=100),
            'action_loss': deque(maxlen=100),
            'total_loss': deque(maxlen=100),
            'latent_cosine_sim': deque(maxlen=100),  # 隐空间余弦相似度（越高越好）
        }
        self.global_step = 0

    def update(
        self,
        obs_history: torch.Tensor,       # [N, 270]
        depth_images: torch.Tensor,      # [N, 2, 58, 98]
        scandots: torch.Tensor,          # [N, 187]
    ) -> Dict[str, float]:
        """
        一次蒸馏更新。

        在收集的 rollout 数据上做多个 epoch 的监督学习。

        Args:
            obs_history:  本体感知历史
            depth_images: 深度图（2帧堆叠）
            scandots:     教师 scandots（高程采样点）

        Returns:
            metrics: 各项损失的均值
        """
        N = obs_history.shape[0]
        indices = torch.randperm(N, device=self.device)

        epoch_metrics = {
            'fm_loss': 0.0, 'latent_loss': 0.0,
            'action_loss': 0.0, 'total_loss': 0.0,
        }

        for epoch in range(self.num_epochs_per_update):
            # Mini-batch
            for mb in range(self.num_mini_batches):
                mb_idx = indices[mb * N // self.num_mini_batches :
                                 (mb + 1) * N // self.num_mini_batches]

                batch_obs = obs_history[mb_idx]
                batch_depth = depth_images[mb_idx]
                batch_scandots = scandots[mb_idx]

                # 前向 + 损失
                self.model.train_student()
                loss_dict = self.model.compute_distill_loss(
                    batch_obs, batch_depth, batch_scandots
                )

                loss = loss_dict['total_loss']

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # 累积指标
                for k in epoch_metrics:
                    epoch_metrics[k] += loss_dict[k].item()

                self.global_step += 1

        # 平均
        n_updates = self.num_epochs_per_update * self.num_mini_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= n_updates
            self.metrics[k].append(epoch_metrics[k])

        # Log
        if self.writer and self.global_step % 50 == 0:
            for k, v in epoch_metrics.items():
                self.writer.add_scalar(f'distill/{k}', v, self.global_step)

        return epoch_metrics

    def get_metrics(self) -> Dict[str, float]:
        """获取最近 100 步的平均指标"""
        return {k: np.mean(list(v)) if v else 0.0
                for k, v in self.metrics.items()
                if k != 'total_loss'}

    def save(self, path: str):
        """保存学生权重 (含 HIM 估计器，play 时无需单独加载教师)"""
        torch.save({
            'depth_encoder': self.model.depth_encoder.state_dict(),
            'fm_policy': self.model.fm_policy.state_dict(),
            'him_estimator': self.model.him_estimator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, path)
        print(f"[FMDistillation] Saved to {path}")

    def load_student(self, path: str):
        """加载学生权重（用于继续训练或推理）"""
        ckpt = torch.load(path, map_location=self.device)
        self.model.depth_encoder.load_state_dict(ckpt['depth_encoder'])
        self.model.fm_policy.load_state_dict(ckpt['fm_policy'])
        if 'him_estimator' in ckpt:
            self.model.him_estimator.load_state_dict(ckpt['him_estimator'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'global_step' in ckpt:
            self.global_step = ckpt['global_step']
        print(f"[FMDistillation] Student loaded from {path}")
