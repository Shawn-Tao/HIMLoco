"""
Scandot Encoder: 将仿真中以机器人中心为原点的高程扫描点(scan_dots)编码为紧凑隐变量。

教师策略使用此模块获取特权地形信息。
学生策略则使用 DepthEncoder + 蒸馏来模仿此隐变量。

输入: scandots [B, num_points]  典型值 187 = 17×11 网格
输出: scandot_latent [B, 32]
"""

import torch
import torch.nn as nn


class ScandotEncoder(nn.Module):
    """
    将展平的 scandots 编码为 32 维地形隐变量。

    内部将 scandots reshape 为 (grid_h × grid_w) 的网格，
    用 2D CNN 提取空间特征，再压缩为潜变量。
    """

    def __init__(
        self,
        num_points: int = 187,
        grid_h: int = 11,
        grid_w: int = 17,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.num_points = num_points
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.latent_dim = latent_dim

        # 验证
        assert grid_h * grid_w == num_points, (
            f"grid_h × grid_w ({grid_h}×{grid_w}) != num_points ({num_points})"
        )

        self.conv = nn.Sequential(
            # [B, 1, 11, 17]
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # [16, 11, 17]
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [32, 6, 9]
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # [32, 3, 5]
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),                                # [32, 1, 1]
            nn.Flatten(),                                           # [32]
            nn.Linear(32, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),                             # [32]
        )

    def forward(self, scandots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scandots: [B, num_points]  展平的高程采样值 (归一化后建议在 [-1, 1])

        Returns:
            latent: [B, latent_dim]    地形隐变量 (默认 32 维)
        """
        B = scandots.shape[0]
        x = scandots.view(B, 1, self.grid_h, self.grid_w)  # [B, 1, H, W]
        return self.conv(x)


class ScandotEncoderMLP(nn.Module):
    """
    备用: 纯 MLP 版本的 scandot encoder（不依赖 2D 网格假设）。
    如果 scandot 采样点不是规则网格，用这个。
    """

    def __init__(self, num_points: int = 187, latent_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_points, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, scandots: torch.Tensor) -> torch.Tensor:
        return self.mlp(scandots)
