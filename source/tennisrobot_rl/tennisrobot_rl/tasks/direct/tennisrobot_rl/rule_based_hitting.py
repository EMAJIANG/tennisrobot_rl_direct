#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FINAL RULE (manual lead, no calibration):
1) Lock hitting plane at world-y = 2.2 (paddle y fixed).
2) Only align world x/z (Z_Pris_H and Z_Pris_V) + swing timing.
3) Swing timing uses ONLY ball vy to compute time-to-cross y_hit (2.2).
4) NO calibration at startup. lead_time is fully manually tuned.

Mapping (confirmed by you):
- Robot (0,0) corresponds to world (x=2, y=2)
- X_Pris controls world y, positive -> y-
- Z_Pris_H controls world x, positive -> x-
- action order: [X_Pris, Z_Pris_H, Z_Pris_V, Rot]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
from isaaclab.app import AppLauncher

# -------------------------
# CLI / Isaac App
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ====== 改成你项目里的真实 import 路径（只改这两行）======
from tennisrobot_rl_env import TennisrobotRlDirectEnv
from tennisrobot_rl_env_cfg import TennisrobotRlDirectEnvCfg
# ======================================================


# -------------------------
# Config
# -------------------------
@dataclass
class HitCfg:
    # env action scaling (your env: target = actions * 50)
    action_scale: float = 50.0

    # mapping reference (confirmed)
    base_world_x0: float = 2.0
    base_world_y0: float = 2.0

    # --- fixed hit plane ---
    y_hit: float = 2.2   # world y plane where we always hit

    # --- manual lead time (YOU TUNE THIS) ---
    lead_time: float = 0.18  # seconds: "start swing when ball is lead_time away from y_hit"

    # --- swing timing (ONLY vy) ---
    vy_min: float = 0.8
    t_window: float = 1.5 * (1 / 120)  # ~0.0125; tune if needed
    t_hit_max: float = 1.0

    # --- swing motion ---
    yaw_ready: float = -math.pi / 3
    yaw_swing: float = math.pi / 2
    swing_hold_s: float = 0.10
    recover_s: float = 0.25

    # --- joint clamps (JOINT VALUES) ---
    x_pris_min: float = -10.0
    x_pris_max: float = 10.0
    z_pris_h_min: float = -10.0
    z_pris_h_max: float = 10.0
    z_pris_v_min: float = 0.0
    z_pris_v_max: float = 2.0
    rot_min: float = -math.pi
    rot_max: float = math.pi

    # --- empirical biases for x/z aim ---
    x_bias: float = 0.89   # world x target = ball_x - x_bias
    z_bias: float = 0.45   # world z target = ball_z - z_bias

    # --- optional freeze xyz during swing ---
    freeze_xyz_during_swing: bool = True


class RuleHitter:
    """
    Joint target order: [X_Pris, Z_Pris_H, Z_Pris_V, Rot]

    Fixed y:
      X_Pris = base_world_y0 - y_hit  (constant)

    Timing:
      t_hit = (y_hit - by) / vy  (vy<0, by>y_hit => t_hit>0)
      swing when |t_hit - lead_time| < t_window
    """

    def __init__(self, cfg: HitCfg, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.mode = "ALIGN"  # ALIGN -> SWING -> RECOVER
        self.t_mode = 0.0

        self.q_des = None  # (4,)
        self.swing_armed = True

        # freeze xyz snapshot during swing
        self.frozen_xyz = None  # tuple(x_pris, z_pris_h, z_pris_v)

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _xpris_for_fixed_y(self) -> float:
        """y_world = y0 - X_Pris => X_Pris = y0 - y_hit"""
        cfg = self.cfg
        return self._clamp(cfg.base_world_y0 - cfg.y_hit, cfg.x_pris_min, cfg.x_pris_max)

    def world_to_joint_targets_xz(self, x_w: float, z_w: float) -> tuple[float, float]:
        """
        x_world = x0 - Z_Pris_H => Z_Pris_H = x0 - x_world
        z_world ~= Z_Pris_V
        """
        cfg = self.cfg
        z_pris_h = cfg.base_world_x0 - x_w
        z_pris_v = z_w

        z_pris_h = self._clamp(z_pris_h, cfg.z_pris_h_min, cfg.z_pris_h_max)
        z_pris_v = self._clamp(z_pris_v, cfg.z_pris_v_min, cfg.z_pris_v_max)
        return z_pris_h, z_pris_v

    def step(
        self,
        dt: float,
        ball_pos_w: torch.Tensor,   # (3,)
        ball_vy_w: float,           # scalar
        q_now_4: torch.Tensor,      # (4,)
    ) -> tuple[torch.Tensor, str, float, float]:
        cfg = self.cfg
        self.t_mode += dt

        ball_pos_w = ball_pos_w.to(self.device).float()
        q_now_4 = q_now_4.to(self.device).float()

        if self.q_des is None:
            self.q_des = q_now_4.clone()
            self.q_des[3] = cfg.yaw_ready

        bx, by, bz = [float(v) for v in ball_pos_w.tolist()]
        vy = float(ball_vy_w)

        # --- compute fixed y joint ---
        x_pris_fixed = self._xpris_for_fixed_y()

        # --- compute x/z targets (track ball x,z with bias) ---
        x_target_w = bx - cfg.x_bias
        z_target_w = bz - cfg.z_bias
        z_pris_h_t, z_pris_v_t = self.world_to_joint_targets_xz(x_target_w, z_target_w)

        # --- time-to-cross y_hit using ONLY vy ---
        t_hit = -1.0
        in_time_window = False
        vy_ok = (vy < -cfg.vy_min)

        if vy_ok and abs(vy) > 1e-6:
            t_hit = float((cfg.y_hit - by) / vy)  # vy<0 and by>y_hit => t_hit>0
            t_hit_ok = (t_hit > 0.0) and (t_hit < cfg.t_hit_max)
            in_time_window = t_hit_ok and (abs(t_hit - cfg.lead_time) < cfg.t_window)

        # --- state machine ---
        if self.mode == "ALIGN":
            self.q_des[0] = float(x_pris_fixed)
            self.q_des[1] = float(z_pris_h_t)
            self.q_des[2] = float(z_pris_v_t)
            self.q_des[3] = cfg.yaw_ready

            if self.swing_armed and in_time_window:
                self.mode = "SWING"
                self.t_mode = 0.0
                self.swing_armed = False
                if cfg.freeze_xyz_during_swing:
                    self.frozen_xyz = (float(self.q_des[0]), float(self.q_des[1]), float(self.q_des[2]))

        elif self.mode == "SWING":
            if cfg.freeze_xyz_during_swing and (self.frozen_xyz is not None):
                self.q_des[0], self.q_des[1], self.q_des[2] = self.frozen_xyz
            else:
                self.q_des[0] = float(x_pris_fixed)
                self.q_des[1] = float(z_pris_h_t)
                self.q_des[2] = float(z_pris_v_t)

            self.q_des[3] = cfg.yaw_swing
            if self.t_mode >= cfg.swing_hold_s:
                self.mode = "RECOVER"
                self.t_mode = 0.0

        elif self.mode == "RECOVER":
            if cfg.freeze_xyz_during_swing and (self.frozen_xyz is not None):
                self.q_des[0], self.q_des[1], self.q_des[2] = self.frozen_xyz
            else:
                self.q_des[0] = float(x_pris_fixed)
                self.q_des[1] = float(z_pris_h_t)
                self.q_des[2] = float(z_pris_v_t)

            self.q_des[3] = cfg.yaw_ready
            if self.t_mode >= cfg.recover_s:
                self.mode = "ALIGN"
                self.t_mode = 0.0
                self.swing_armed = True
                self.frozen_xyz = None

        # clamp yaw
        self.q_des[3] = self._clamp(float(self.q_des[3].item()), cfg.rot_min, cfg.rot_max)

        return self.q_des.clone(), self.mode, t_hit, vy


def main():
    env_cfg = TennisrobotRlDirectEnvCfg()
    env_cfg.ball_pos_x_range = (0.3, 0.5)
    env_cfg.ball_speed_y_range = (-7.0, -6.0)
    env_cfg.ball_speed_x_range = (0.0, 0.0)
    env_cfg.scene.num_envs = args.num_envs

    env = TennisrobotRlDirectEnv(env_cfg, render_mode=None)
    device = env.device

    cfg = HitCfg()
    hitter = RuleHitter(cfg, device=device)

    env.reset()
    step = 0

    while simulation_app.is_running():
        dt = float(env.dt)

        ball_pos_w = env._ball.data.root_pos_w[0]
        ball_vy_w = float(env._ball.data.root_lin_vel_w[0, 1].item())
        q_now_4 = env._robot.data.joint_pos[0, :4]

        q_des_4, mode, t_hit, vy = hitter.step(
            dt=dt,
            ball_pos_w=ball_pos_w,
            ball_vy_w=ball_vy_w,
            q_now_4=q_now_4,
        )

        actions = (q_des_4 / cfg.action_scale).unsqueeze(0)
        _, _, terminated, truncated, _ = env.step(actions)

        if terminated.any() or truncated.any():
            env.reset()
            hitter.swing_armed = True
            hitter.mode = "ALIGN"
            hitter.t_mode = 0.0
            hitter.frozen_xyz = None

        step += 1
        # 你想每步都打日志就保留 %1；否则改成 50
        if step % 1 == 0:
            bx, by, bz = [float(v) for v in ball_pos_w.tolist()]
            print(
                f"[{step:05d}] mode={mode:>7s} ball=({bx:.3f},{by:.3f},{bz:.3f}) "
                f"vy={vy:.3f} t_hit={t_hit:.3f} lead={cfg.lead_time:.3f} q_des={q_des_4.tolist()}"
            )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
