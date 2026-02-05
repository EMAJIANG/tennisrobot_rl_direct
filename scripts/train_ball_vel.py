# tennisrobot_rl/velocity_estimator/scripts/train_ball_vel.py
from __future__ import annotations

import os
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))  # velocity_estimator/
if PROJ_DIR not in sys.path:
    sys.path.insert(0, PROJ_DIR)

import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# ============= your env =============
from tennisrobot_rl.tasks.direct.tennisrobot_rl.tennisrobot_rl_env import TennisrobotRlDirectEnv
from tennisrobot_rl.tasks.direct.tennisrobot_rl.tennisrobot_rl_env_cfg import TennisrobotRlDirectEnvCfg

from tennisrobot_rl.velocity_estimator.models.mlp_speed import BallSpeedMLP, MLPConfig
from tennisrobot_rl.velocity_estimator.modules.normalizer import RunningMeanStd, NormalizerConfig
from tennisrobot_rl.velocity_estimator.data.replay_buffer import SupervisedReplayBuffer, ReplayBufferConfig
from tennisrobot_rl.velocity_estimator.envs.observation_buffer import PositionHistoryBuffer, PositionHistoryConfig


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    # sim/env
    seed: int = 0
    device: str = "cuda"
    num_envs: int = 1

    # NOTE: dt will be overridden by env.dt after env is created.
    dt: float = 1.0 / 120.0

    # ---- history feature ----
    history_len: int = 8          # <<< 8帧历史
    use_dt_input: bool = True     # append dt as last feature

    # data/buffer
    capacity: int = 2_000_000
    batch_size: int = 4096
    warmup_steps: int = 2000
    train_every: int = 2
    gradient_steps: int = 1

    # model
    hidden: tuple[int, int] = (128, 128)
    dropout: float = 0.0
    use_layernorm: bool = False

    # optimizer
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0

    # run
    total_steps: int = 200_000
    log_every: int = 200
    ckpt_every: int = 20_000
    out_dir: str = "checkpoints_ball_vel"
    run_name: str = "ball_vel_mlp_hist8_dt"

    # normalizer
    norm_clip: float = 10.0
    norm_update_every: int = 1  # update RMS every N sim steps


def compute_x_dim(history_len: int, use_dt_input: bool) -> int:
    return 3 * int(history_len) + (1 if use_dt_input else 0)


# -------------------------
# IsaacLab integration
# -------------------------
def make_env(cfg: TrainConfig) -> TennisrobotRlDirectEnv:
    env_cfg = TennisrobotRlDirectEnvCfg()
    env_cfg.scene.num_envs = int(cfg.num_envs)
    env = TennisrobotRlDirectEnv(cfg=env_cfg, render_mode=None)
    cfg.dt = float(env.dt)
    return env


@torch.no_grad()
def get_ball_pos_vel(env: TennisrobotRlDirectEnv) -> Tuple[torch.Tensor, torch.Tensor]:
    ball = env.scene.rigid_objects["ball"]
    # pos in env-local frame to avoid env_origin offsets across envs
    pos_w = (ball.data.root_pos_w - env.scene.env_origins)  # (N,3)
    vel_w = ball.data.root_lin_vel_w                        # (N,3)
    return pos_w, vel_w


def get_reset_ids(terminated, truncated, device: torch.device) -> torch.Tensor:
    if terminated is None and truncated is None:
        return torch.empty((0,), device=device, dtype=torch.long)

    if isinstance(terminated, torch.Tensor):
        done = terminated
        if truncated is not None:
            done = done | truncated
        if done.numel() == 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        return torch.nonzero(done, as_tuple=False).squeeze(-1).to(device=device, dtype=torch.long)

    t = torch.as_tensor(terminated, dtype=torch.bool, device=device)
    if truncated is not None:
        t = t | torch.as_tensor(truncated, dtype=torch.bool, device=device)
    return torch.nonzero(t, as_tuple=False).squeeze(-1).to(device=device, dtype=torch.long)


# -------------------------
# Checkpoint utils
# -------------------------
def save_checkpoint(path: str, model: torch.nn.Module, rms: RunningMeanStd, train_cfg: TrainConfig) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "rms_x": rms.state_dict(),
        "train_cfg": asdict(train_cfg),
    }
    torch.save(payload, path)


# -------------------------
# Main training loop
# -------------------------
def main(train_cfg: TrainConfig) -> None:
    torch.manual_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    # --- build env ---
    env = make_env(train_cfg)
    if int(env.num_envs) != int(train_cfg.num_envs):
        raise RuntimeError(f"env.num_envs={env.num_envs} != cfg.num_envs={train_cfg.num_envs}")

    # --- feature dims ---
    x_dim = compute_x_dim(train_cfg.history_len, train_cfg.use_dt_input)

    # --- model ---
    mlp_cfg = MLPConfig(
        in_dim=x_dim,
        out_dim=3,
        hidden=tuple(train_cfg.hidden),
        activation="relu",
        use_layernorm=train_cfg.use_layernorm,
        dropout=train_cfg.dropout,
    )
    model = BallSpeedMLP(mlp_cfg).to(device)
    model.train()

    # --- normalizer (input x) ---
    rms = RunningMeanStd(
        dim=x_dim,
        device=device,
        dtype=torch.float32,
        cfg=NormalizerConfig(eps=1e-8, clip=train_cfg.norm_clip),
    )

    # --- replay buffer ---
    rb = SupervisedReplayBuffer(
        ReplayBufferConfig(
            capacity=train_cfg.capacity,
            x_dim=x_dim,
            y_dim=3,
            device=str(device),
            dtype=torch.float32,
        )
    )

    # --- position history buffer ---
    hist = PositionHistoryBuffer(
        num_envs=train_cfg.num_envs,
        dt=train_cfg.dt,
        device=device,
        dtype=torch.float32,
        cfg=PositionHistoryConfig(hist_len=train_cfg.history_len, use_dt_input=train_cfg.use_dt_input, x_dim=x_dim),
    )

    # --- optimizer ---
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    # --- reset env ---
    _ = env.reset()
    hist.reset(torch.arange(train_cfg.num_envs, device=device, dtype=torch.long))

    # action: your env action space is 4
    action = torch.zeros((train_cfg.num_envs, 4), device=device, dtype=torch.float32)

    # --- stats ---
    t0 = time.time()
    loss_ema: Optional[float] = None
    num_updates = 0

    for step in range(1, train_cfg.total_steps + 1):
        # 1) step sim
        out = env.step(action)
        if not (isinstance(out, tuple) and len(out) >= 5):
            raise RuntimeError(
                "env.step(action) must return at least (obs, rew, terminated, truncated, info). "
                f"Got type={type(out)} len={len(out) if isinstance(out, tuple) else 'NA'}"
            )
        _, _, terminated, truncated, _ = out[:5]

        # 2) read ball GT state
        pos_w, vel_w = get_ball_pos_vel(env)

        # 3) update history and build x
        hist.push(pos_w)
        x_all, mask = hist.get_features()  # (N, x_dim), (N,)

        if mask.any():
            x = x_all[mask]
            y = vel_w[mask]

            # update RMS using raw x
            if train_cfg.norm_update_every > 0 and (step % train_cfg.norm_update_every == 0):
                rms.update(x)

            # store normalized x and raw y
            x_n = rms.normalize(x)
            rb.add(x_n, y)

        # 4) handle resets
        reset_ids = get_reset_ids(terminated, truncated, device=device)
        if reset_ids.numel() > 0:
            hist.reset(reset_ids)

        # 5) train
        can_train = (len(rb) >= train_cfg.batch_size) and (step >= train_cfg.warmup_steps)
        if can_train and (step % train_cfg.train_every == 0):
            for _ in range(train_cfg.gradient_steps):
                xb, yb = rb.sample(train_cfg.batch_size)
                pred = model(xb)
                loss = F.mse_loss(pred, yb)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if train_cfg.grad_clip_norm and train_cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip_norm)
                optim.step()

                num_updates += 1
                l = float(loss.detach().item())
                loss_ema = l if loss_ema is None else (0.98 * loss_ema + 0.02 * l)

        # 6) logging / checkpoint
        if step % train_cfg.log_every == 0:
            dt_s = time.time() - t0
            sps = step / max(dt_s, 1e-6)
            print(
                f"[step {step:>8d}] "
                f"K={train_cfg.history_len} x_dim={x_dim} "
                f"dt={train_cfg.dt:.6f} "
                f"buf={len(rb):>7d} updates={num_updates:>7d} "
                f"loss_ema={loss_ema if loss_ema is not None else float('nan'):.6f} "
                f"sps={sps:.1f}"
            )

        if step % train_cfg.ckpt_every == 0:
            ckpt_dir = os.path.join(train_cfg.out_dir, train_cfg.run_name)
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{step}.pt")
            save_checkpoint(ckpt_path, model, rms, train_cfg)
            print(f"Saved checkpoint: {ckpt_path}")

    # final save
    ckpt_dir = os.path.join(train_cfg.out_dir, train_cfg.run_name)
    ckpt_path = os.path.join(ckpt_dir, "ckpt_final.pt")
    save_checkpoint(ckpt_path, model, rms, train_cfg)
    print(f"Training done. Saved: {ckpt_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)
