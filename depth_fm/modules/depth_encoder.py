"""
Depth Encoder: 将深度图编码为紧凑的 32 维地形隐变量。

学生策略使用此模块替代教师的 ScandotEncoder。
输入尺寸 98×58，2 帧堆叠，归一化到 [-0.5, 0.5]。

蒸馏时: MSE(depth_latent, scandot_latent) 作为 latent alignment loss。
"""

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """
    轻量 Depth CNN: [B, 2, 58, 98] → [B, 32]

    设计参考:
      - Extreme Parkour DepthOnlyFCBackbone 的架构思想
      - 输入尺寸改为 98×58（匹配 106×60 裁剪后的尺寸）
      - 3 层 stride-2 降采样 + AdaptiveAvgPool
    """

    def __init__(
        self,
        num_frames: int = 2,       # 深度图帧数（2 = 当前帧 + 上一帧）
        input_height: int = 58,     # 裁剪后高度
        input_width: int = 98,      # 裁剪后宽度
        latent_dim: int = 32,       # 输出隐变量维度（需与 scandot_latent_dim 匹配）
    ):
        super().__init__()
        self.num_frames = num_frames
        self.input_dims = (input_height, input_width)
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            # 输入: [B, 2, 58, 98]
            nn.Conv2d(num_frames, 32, kernel_size=5, stride=2, padding=2),
            # → [B, 32, 29, 49]
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # → [B, 64, 15, 25]
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            # → [B, 64, 8, 13]
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
            # → [B, 64, 1, 1]
        )

        self.head = nn.Sequential(
            nn.Flatten(),                     # [B, 64]
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),       # [B, 32]
        )

        # 参数量 ~0.15M，FLOPs ~0.04G
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, depth_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_img: [B, num_frames, 58, 98]  归一化深度图 (-0.5 ~ 0.5)

        Returns:
            latent: [B, latent_dim]  地形隐变量
        """
        x = self.conv(depth_img)
        return self.head(x)


class DepthEncoderLarger(nn.Module):
    """
    备用: 深一点的版本，用于复杂地形。
    参数量 ~0.3M
    """

    def __init__(self, num_frames=2, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_frames, 32, 5, stride=2, padding=2),   # [32, 29, 49]
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),            # [64, 15, 25]
            nn.ELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),           # [128, 8, 13]
            nn.ELU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),          # [128, 8, 13]
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, depth_img):
        return self.head(self.conv(depth_img))
