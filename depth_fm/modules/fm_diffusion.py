"""
Flow Matching Diffusion Policy: 替代传统 MLP 高斯策略头。

训练: 从教师收集的 (obs, action) 数据中学习条件分布 p(action | obs)
推理: 从噪声中沿直线速度场积分生成动作序列

数学:
  - 概率路径:  a_t = (1-t)·a_0 + t·ε,  t ∈ [0,1]
  - 速度场:    v = ε - a_0
  - 损失:      L = MSE(v_pred, ε - a_0)
  - 采样:      a_{k+1} = a_k - Δt·v_pred(a_k, t_k, c)

参考:
  - Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
  - ReinFlow, NeurIPS 2025 (FM + RL fine-tuning for locomotion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 速度场网络 (去噪器)
# ============================================================

class VelocityField1D(nn.Module):
    """
    1D Temporal U-Net: 预测速度场 v = ε - a_0

    输入: 含噪动作序列 + 时间步 + 条件
    输出: 速度场 v_pred（与 a_t 同维度）

    设计要点:
      - 1D Conv 沿 horizon 维度（时序），不是 2D 图像 CNN
      - FiLM conditioning: 条件 c 调制每层特征
      - Time embedding: 正弦位置编码 + MLP
    """

    def __init__(
        self,
        action_dim: int = 12,
        horizon: int = 10,
        cond_dim: int = 128,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # ===== Time Embedding =====
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ===== Condition Projection → FiLM params =====
        self.cond_proj = nn.Sequential(           # 保留用于全局条件投影 (可选)
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ===== Input projection =====
        self.input_proj = nn.Linear(action_dim, hidden_dim)

        # ===== Encoder blocks =====
        self.enc_blocks = nn.ModuleList()
        ch = hidden_dim
        for i in range(num_res_blocks):
            out_ch = hidden_dim * min(2, i + 1) if i < 2 else hidden_dim * 2
            self.enc_blocks.append(FiLMResBlock1D(ch, out_ch, hidden_dim, cond_dim))
            ch = out_ch

        # ===== Middle: Self-Attention along time =====
        self.mid_attn = nn.MultiheadAttention(
            ch, num_heads=4, batch_first=True
        )

        # ===== Decoder blocks =====
        self.dec_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            skip_ch = self.enc_blocks[num_res_blocks - 1 - i].out_channels
            in_ch = ch + skip_ch
            out_ch = hidden_dim if i == num_res_blocks - 1 else ch // 2
            self.dec_blocks.append(FiLMResBlock1D(in_ch, out_ch, hidden_dim, cond_dim))
            ch = out_ch

        # ===== Output projection =====
        self.output_proj = nn.Linear(ch, action_dim)

        # 参数量 ~1.5M

    def forward(
        self,
        a_t: torch.Tensor,         # [B, H, action_dim]  含噪动作序列
        t: torch.Tensor,           # [B] 或 [B, 1]  时间步 t ∈ [0,1]
        condition: torch.Tensor,   # [B, cond_dim]  条件向量
    ) -> torch.Tensor:
        """
        Returns:
            v_pred: [B, H, action_dim]  预测的速度场
        """
        B = a_t.shape[0]

        # Time embedding
        t_emb = self.time_mlp(t.float().squeeze(-1))   # [B, hidden_dim]

        # Condition embedding
        cond_emb = self.cond_proj(condition)            # [B, hidden_dim]

        # Input projection: [B, H, action_dim] → [B, H, hidden_dim]
        x = self.input_proj(a_t)
        x = x.transpose(1, 2)  # [B, hidden_dim, H] for Conv1d

        # Encoder
        skips = []
        for blk in self.enc_blocks:
            x = blk(x, t_emb, cond_emb)
            skips.append(x)

        # Middle Attention
        x_t = x.transpose(1, 2)        # [B, H, ch]
        x_t, _ = self.mid_attn(x_t, x_t, x_t)
        x = x_t.transpose(1, 2)        # [B, ch, H]

        # Decoder
        for blk, skip in zip(self.dec_blocks, reversed(skips)):
            x = torch.cat([x, skip], dim=1)   # skip connection
            x = blk(x, t_emb, cond_emb)

        # Output
        x = x.transpose(1, 2)                  # [B, H, ch]
        v_pred = self.output_proj(x)           # [B, H, action_dim]

        return v_pred


# ============================================================
# 残差块 + FiLM
# ============================================================

class FiLMResBlock1D(nn.Module):
    """1D 残差卷积块 + 内部 FiLM 条件调制"""

    def __init__(self, in_channels, out_channels, time_dim, cond_dim):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.time_proj = nn.Linear(time_dim, out_channels)
        # 内部 FiLM projection: condition → per-channel γ, β
        self.film_proj = nn.Linear(cond_dim, out_channels * 2)

    def forward(self, x, t_emb, condition):
        # FiLM params from condition
        film = self.film_proj(condition)           # [B, out_channels*2]
        gamma = film[:, :self.out_channels]         # [B, out_channels]
        beta  = film[:, self.out_channels:]         # [B, out_channels]

        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        # Second FiLM uses same gamma/beta (or can use a second set)
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        h = F.silu(h)

        return h + self.skip(x)

    # 兼容旧接口 (返回 tuple)
    def forward_with_out_ch(self, x, t_emb, condition):
        return self.forward(x, t_emb, condition), self.out_channels


# ============================================================
# 正弦位置编码
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ============================================================
# FM Diffusion Policy (完整 policy wrapper)
# ============================================================

class FMDiffusionPolicy(nn.Module):
    """
    Flow Matching 扩散策略: 学生用。

    训练:  velocity_net 学习 v = ε - a_0
    推理:  从 N(0,I) 出发，K 步欧拉积分生成动作序列

    用法:
        # 训练
        policy = FMDiffusionPolicy(action_dim=12, horizon=10, cond_dim=128)
        loss = policy.compute_loss(a_0, condition)

        # 推理
        actions = policy.sample(condition, num_steps=5)
    """

    def __init__(
        self,
        action_dim: int = 12,
        horizon: int = 10,
        cond_dim: int = 128,
        hidden_dim: int = 256,
        num_steps_train: int = 5,   # 训练时采样步数（监控用）
        num_steps_infer: int = 5,   # 推理时欧拉步数
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.cond_dim = cond_dim
        self.num_steps_infer = num_steps_infer
        self.num_steps_train = num_steps_train

        self.velocity_net = VelocityField1D(
            action_dim=action_dim,
            horizon=horizon,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
        )

    def compute_loss(
        self,
        a_0: torch.Tensor,           # [B, H, action_dim] 干净动作序列（来自教师）
        condition: torch.Tensor,     # [B, cond_dim] 条件向量
    ) -> torch.Tensor:
        """
        Flow Matching 训练损失。

        1. 随机采样时间 t ∈ [0,1]
        2. 随机采样噪声 ε
        3. 构造 a_t = (1-t)·a_0 + t·ε
        4. 预测速度场 v_pred，目标 = ε - a_0
        5. MSE loss
        """
        B = a_0.shape[0]

        # 随机时间
        t = torch.rand(B, device=a_0.device)
        # 随机噪声
        epsilon = torch.randn_like(a_0)

        # 含噪动作: 直线插值路径
        t_expanded = t.view(B, 1, 1)
        a_t = (1 - t_expanded) * a_0 + t_expanded * epsilon

        # 预测速度场
        v_pred = self.velocity_net(a_t, t, condition)

        # 目标速度场: ε - a_0
        v_target = epsilon - a_0

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,     # [B, cond_dim]
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        FM 欧拉积分采样: 从纯噪声恢复动作序列。

        Args:
            condition: 条件向量
            num_steps: 欧拉步数 (None 则用 self.num_steps_infer)

        Returns:
            actions: [B, H, action_dim] 去噪后的动作序列
        """
        if num_steps is None:
            num_steps = self.num_steps_infer

        B = condition.shape[0]
        device = condition.device

        # 从纯噪声开始 (t=1)
        a = torch.randn(B, self.horizon, self.action_dim, device=device)

        # K 步欧拉积分
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_val = 1.0 - k * dt
            t = torch.full((B,), t_val, device=device)
            v = self.velocity_net(a, t, condition)
            a = a - dt * v   # 沿速度场逆向移动

        return a

    @torch.no_grad()
    def act_inference(
        self,
        condition: torch.Tensor,     # [1, cond_dim] 或 [B, cond_dim]
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        推理入口: 采样完整动作序列，取第一步执行。

        Returns:
            action: [action_dim]  当前应执行的关节目标
        """
        action_seq = self.sample(condition, num_steps)    # [B, H, action_dim]
        return action_seq[0, 0, :]                        # [action_dim] — 取第一步
