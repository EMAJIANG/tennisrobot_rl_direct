# scripts/vis_check_ball_state.py
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass


from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# ---- 直接从文件路径导入 env，避免 import tennisrobot_rl 包触发 tasks 扫描 ----
ENV_DIR = "/home/robotennis2025/tennis_robot_iros/tennisrobot_rl/source/tennisrobot_rl/tennisrobot_rl/tasks/direct/tennisrobot_rl"
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

from tennisrobot_rl.tasks.direct.tennisrobot_rl.tennisrobot_rl_env import TennisrobotRlDirectEnv
from tennisrobot_rl.tasks.direct.tennisrobot_rl.tennisrobot_rl_env_cfg import TennisrobotRlDirectEnvCfg


# -------------------------
# Config
# -------------------------
@dataclass
class Cfg:
    num_envs: int = 1
    device: str = "cuda"
    steps: int = 5000
    print_every: int = 10
    sleep_s: float = 0.01
    render_mode: str | None = "human"
    ckpt_path: str = "checkpoints_ball_vel/ball_vel_mlp_dt/ckpt_final.pt"  # 改成你的ckpt


# -------------------------
# Minimal MLP (match training)
# -------------------------
class BallSpeedMLP(nn.Module):
    def __init__(self, in_dim=10, out_dim=3, hidden=(128, 128), use_layernorm=False, dropout=0.0):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def get_ball_pos_vel(env):
    ball = env.scene.rigid_objects["ball"]
    return ball.data.root_pos_w, ball.data.root_lin_vel_w


def main():
    cfg = Cfg()
    device = torch.device(cfg.device)

    # ---- build env (GUI) ----
    env_cfg = TennisrobotRlDirectEnvCfg()
    env_cfg.scene.num_envs = int(cfg.num_envs)
    env = TennisrobotRlDirectEnv(cfg=env_cfg, render_mode=cfg.render_mode)
    _ = env.reset()

    dt = float(env.dt)
    print(f"[vis] env.num_envs={env.num_envs} dt={dt}")

    # ---- load checkpoint ----
    payload = torch.load(cfg.ckpt_path, map_location="cpu")
    if "model" not in payload or "rms_x" not in payload:
        raise RuntimeError(f"ckpt missing keys. need ['model','rms_x'], got {list(payload.keys())}")

    train_cfg = payload.get("train_cfg", {})
    hidden = train_cfg.get("hidden", (128, 128))
    dropout = float(train_cfg.get("dropout", 0.0))
    use_ln = bool(train_cfg.get("use_layernorm", False))

    model = BallSpeedMLP(in_dim=10, out_dim=3, hidden=tuple(hidden), use_layernorm=use_ln, dropout=dropout)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()

    # ---- load normalizer stats (rms_x) ----
    rms = payload["rms_x"]
    mean_x = rms["mean"].to(device=device, dtype=torch.float32)          # (10,)
    var_x = rms["var"].to(device=device, dtype=torch.float32)            # (10,)
    eps = float(rms.get("cfg", {}).get("eps", 1e-8))
    clip = rms.get("cfg", {}).get("clip", 10.0)
    std_x = torch.sqrt(var_x.clamp_min(eps))

    def normalize(x: torch.Tensor) -> torch.Tensor:
        x_n = (x - mean_x) / std_x
        if clip is not None:
            x_n = torch.clamp(x_n, -float(clip), float(clip))
        return x_n

    # ---- action (keep robot still) ----
    action = torch.zeros((env.num_envs, 4), device=device, dtype=torch.float32)

    # ---- history buffer for x = [p_{t-2}, p_{t-1}, p_t, dt] ----
    pos_hist = torch.zeros((env.num_envs, 3, 3), device=device, dtype=torch.float32)
    count = torch.zeros((env.num_envs,), device=device, dtype=torch.int32)
    dt_col = torch.tensor(dt, device=device, dtype=torch.float32).view(1, 1).expand(env.num_envs, 1)

    for step in range(1, cfg.steps + 1):
        out = env.step(action)
        _, _, terminated, truncated, _ = out[:5]
        done = terminated | truncated
        reset_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)

        if reset_ids.numel() > 0:
            pos_hist[reset_ids] = 0.0
            count[reset_ids] = 0

        pos_w, vel_gt = get_ball_pos_vel(env)
        pos_w = pos_w.to(device=device, dtype=torch.float32)

        # push history
        pos_hist = torch.roll(pos_hist, shifts=-1, dims=1)
        pos_hist[:, -1, :].copy_(pos_w)
        count = torch.clamp(count + 1, max=3)

        ready = count >= 3
        v_hat = torch.zeros_like(vel_gt)

        if ready.any():
            x_pos = pos_hist.reshape(env.num_envs, 9)
            x = torch.cat([x_pos, dt_col], dim=1)          # (N,10)
            x_n = normalize(x)
            v_hat_ready = model(x_n[ready])
            v_hat[ready] = v_hat_ready

        if step % cfg.print_every == 0:
            i = 0
            e = torch.norm(v_hat[i] - vel_gt[i]).item()
            gt_spd = torch.norm(vel_gt[i]).item()
            hat_spd = torch.norm(v_hat[i]).item()
            print(
                f"[{step:05d}] ready={bool(ready[i].item())} "
                f"gt_v={vel_gt[i].tolist()} "
                f"hat_v={v_hat[i].tolist()} "
                f"|e|={e:.4f} gt|v|={gt_spd:.3f} hat|v|={hat_spd:.3f}"
            )

        if cfg.sleep_s > 0:
            time.sleep(cfg.sleep_s)


if __name__ == "__main__":
    main()
