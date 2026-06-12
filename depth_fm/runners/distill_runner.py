"""
Distillation Runner: 教师→学生蒸馏的训练 + 推理主循环。

DAgger-style: 学生在环境中执行动作，收集教师标签，蒸馏更新。

环境变量约定 (与 HIMLoco LeggedRobot 一致):
  - env.obs_buf:              [num_envs, num_obs] 本体感知历史 (270维)
  - env.privileged_obs_buf:   [num_envs, privileged_obs] 特权观测 (含 scandots)
  - env.extras["depth"]:      [num_envs, 2, 58, 98] 深度图 (2帧堆叠)
  - env.num_obs:              int  观测维度 (270)
  - env.num_one_step_obs:     int  单步观测维度 (45)
  - env.num_actions:          int  动作维度 (12)
  - env.num_envs:             int  环境数量
"""

import torch
import numpy as np
import os
from collections import deque
from depth_fm.algorithms.fm_distillation import FMDistillation

# Scandots 在特权观测中的位置: [one_step_obs(45) | base_lin_vel(3) | external_forces(3) | scandots(187)]
SCANDOT_START = 45 + 3 + 3   # = 51
SCANDOT_LEN = 187


class DistillRunner:
    """
    蒸馏训练 Runner。

    训练循环:
      for iteration in range(max_iterations):
          1. 环境 rollout (学生执行动作)
          2. 蒸馏更新 (FM loss + latent alignment)
          3. Log & save
    """

    def __init__(
        self,
        env,
        distill_model,
        distill_algo: FMDistillation,
        num_steps_per_env: int = 24,
        save_interval: int = 50,
        log_dir: str = './logs/distill',
        device: str = 'cuda:0',
    ):
        self.env = env
        self.model = distill_model
        self.algo = distill_algo
        self.num_steps_per_env = num_steps_per_env
        self.save_interval = save_interval
        self.log_dir = log_dir
        self.device = device

        os.makedirs(log_dir, exist_ok=True)

        # 从环境读取维度
        self.num_envs = env.num_envs
        self.num_actor_obs = env.num_obs                       # 270 = 45 × 6
        self.num_one_step_obs = env.num_one_step_obs           # 45
        self.num_actions = env.num_actions                     # 12

        # Rollout buffers
        self.obs_history_buf = torch.zeros(
            self.num_envs, num_steps_per_env,
            self.num_actor_obs, device=device
        )
        self.depth_buf = torch.zeros(
            self.num_envs, num_steps_per_env,
            2, 58, 98, device=device
        )
        self.scandot_buf = torch.zeros(
            self.num_envs, num_steps_per_env,
            SCANDOT_LEN, device=device
        )

        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def _get_scandots(self) -> torch.Tensor:
        """从特权观测中提取 scandots"""
        pbuf = self.env.privileged_obs_buf
        if pbuf is None:
            return torch.zeros(self.num_envs, SCANDOT_LEN, device=self.device)
        return pbuf[:, SCANDOT_START : SCANDOT_START + SCANDOT_LEN]

    def collect_rollout(self):
        """收集一个 rollout 周期的数据"""
        self.obs_history_buf.zero_()
        self.depth_buf.zero_()
        self.scandot_buf.zero_()

        for step in range(self.num_steps_per_env):
            # 获取观测
            obs_history = self.env.obs_buf                     # [N, 270]
            depth = self.env.extras.get('depth', None)         # [N, 2, 58, 98]
            scandots = self._get_scandots()                    # [N, 187]

            # 学生推理 (批量 — 避免逐 env 循环)
            if depth is not None:
                condition, _ = self.model.get_student_condition(
                    obs_history, depth
                )  # condition: [N, 96]
                action_seq = self.model.fm_policy.sample(
                    condition, num_steps=self.model.fm_policy.num_steps_infer
                )  # [N, H, 12]
                actions = action_seq[:, 0, :]  # 取第一步
            else:
                # 盲态回退 (深度未就绪)
                _, him_z = self.model.him_estimator(obs_history)
                dummy_cond = torch.cat([
                    obs_history[:, :self.num_one_step_obs],
                    torch.zeros(self.num_envs, 3, device=self.device),
                    him_z,
                    torch.zeros(self.num_envs, 32, device=self.device),
                ], dim=-1)
                action_seq = self.model.fm_policy.sample(
                    dummy_cond, num_steps=self.model.fm_policy.num_steps_infer
                )
                actions = action_seq[:, 0, :]

            # 环境 step (HIMLoco 返回 7 个值)
            step_out = self.env.step(actions)
            if len(step_out) == 7:
                obs, privileged_obs, rewards, dones, infos, _, _ = step_out
            else:
                obs, privileged_obs, rewards, dones, infos = step_out

            # 存储
            self.obs_history_buf[:, step, :] = obs_history
            if depth is not None:
                self.depth_buf[:, step, :, :, :] = depth
            self.scandot_buf[:, step, :] = scandots

            # Episode 统计
            for i in range(self.num_envs):
                done_val = dones[i].item() if isinstance(dones, torch.Tensor) else dones[i]
                if done_val:
                    if 'episode' in infos:
                        ep = infos['episode']
                        if 'r' in ep and 'l' in ep:
                            self.episode_rewards.append(ep['r'][i].item())
                            self.episode_lengths.append(ep['l'][i].item())
                        elif hasattr(self.env, 'episode_sums') and hasattr(self.env, 'episode_length_buf'):
                            total = sum(
                                v[i].item() for v in self.env.episode_sums.values()
                            )
                            self.episode_rewards.append(total)
                            self.episode_lengths.append(
                                self.env.episode_length_buf[i].item()
                            )

    def learn(self, num_iterations: int, init_at_random_ep_len: bool = True):
        """蒸馏训练主循环"""
        if init_at_random_ep_len:
            self.env.reset()

        for it in range(num_iterations):
            # 1. 收集 rollout
            self.collect_rollout()

            # 2. 蒸馏更新
            metrics = self.algo.update(
                self.obs_history_buf.flatten(0, 1),
                self.depth_buf.flatten(0, 1),
                self.scandot_buf.flatten(0, 1),
            )

            # 3. Log
            if it % 10 == 0:
                avg_r = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0
                avg_l = np.mean(list(self.episode_lengths)) if self.episode_lengths else 0.0
                print(
                    f"Iter {it:5d} | "
                    f"FM:{metrics['fm_loss']:.4f} "
                    f"Lat:{metrics['latent_loss']:.4f} "
                    f"Tot:{metrics['total_loss']:.4f} | "
                    f"Rew:{avg_r:.0f} Len:{avg_l:.0f}"
                )

            # 4. 保存
            if it % self.save_interval == 0 and it > 0:
                save_path = os.path.join(self.log_dir, f'student_ckpt_{it:06d}.pt')
                self.algo.save(save_path)

        final_path = os.path.join(self.log_dir, 'student_final.pt')
        self.algo.save(final_path)
        print(f"\nTraining complete. Final model → {final_path}")

    def play(self, num_episodes: int = 5, render: bool = True):
        """可视化学生策略"""
        self.model.eval_all()

        for ep in range(num_episodes):
            self.env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                obs_history = self.env.obs_buf
                depth = self.env.extras.get('depth', None)

                action = self.model.act_student(obs_history[:1], depth[:1] if depth is not None else None)

                step_out = self.env.step(action.unsqueeze(0))
                if len(step_out) == 7:
                    _, _, rewards, dones, _ = step_out[:5]
                else:
                    _, _, rewards, dones, _ = step_out
                reward = rewards[0] if isinstance(rewards, torch.Tensor) else rewards
                done = dones[0] if isinstance(dones, torch.Tensor) else dones
                ep_reward += reward.item() if isinstance(reward, torch.Tensor) else reward

                if render:
                    self.env.render()

            print(f"Episode {ep}: reward = {ep_reward:.1f}")
