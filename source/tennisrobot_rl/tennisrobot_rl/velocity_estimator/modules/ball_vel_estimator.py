from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BallVelEstimatorCfg:
    ckpt_path: str
    device: str = "cuda"

    # fallback if ckpt lacks train_cfg
    history_len: int = 8
    use_dt_input: bool = True


class BallSpeedMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 3,
        hidden: tuple[int, ...] = (128, 128),
        use_layernorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        last = int(in_dim)
        for h in hidden:
            h = int(h)
            layers.append(nn.Linear(last, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and float(dropout) > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            last = h
        layers.append(nn.Linear(last, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BallVelEstimator:
    """Online ball velocity estimator.

    Inputs (deploy):
      - ball position p_t (N,3) each step
      - dt (float)
      - reset_ids (optional) to clear history for selected envs

    Output:
      - v_hat (N,3)

    This module:
      - loads ckpt (model + RMS stats)
      - maintains per-env position history of length K
      - normalizes x and predicts v_hat when ready (count>=K)
    """

    def __init__(self, cfg: BallVelEstimatorCfg, num_envs: int):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(cfg.device)

        if not cfg.ckpt_path or not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(f"BallVelEstimator ckpt_path not found: {cfg.ckpt_path}")

        payload = torch.load(cfg.ckpt_path, map_location="cpu")
        if "model" not in payload or "rms_x" not in payload:
            raise RuntimeError(f"ckpt missing keys ['model','rms_x'], got {list(payload.keys())}")

        train_cfg = payload.get("train_cfg", {})

        # prefer ckpt config if present
        self.K = int(train_cfg.get("history_len", cfg.history_len))
        self.use_dt = bool(train_cfg.get("use_dt_input", cfg.use_dt_input))
        self.x_dim = 3 * self.K + (1 if self.use_dt else 0)

        hidden = tuple(train_cfg.get("hidden", (128, 128)))
        dropout = float(train_cfg.get("dropout", 0.0))
        use_ln = bool(train_cfg.get("use_layernorm", False))

        # build model
        self.model = BallSpeedMLP(
            in_dim=self.x_dim, out_dim=3, hidden=hidden, use_layernorm=use_ln, dropout=dropout
        )
        self.model.load_state_dict(payload["model"], strict=True)
        self.model = self.model.to(self.device).eval()

        # load RMS
        rms = payload["rms_x"]
        mean_x = rms["mean"].to(device=self.device, dtype=torch.float32)
        var_x = rms["var"].to(device=self.device, dtype=torch.float32)
        rms_cfg = rms.get("cfg", {})

        self.eps = float(rms_cfg.get("eps", 1e-8))
        self.clip = float(rms_cfg.get("clip", 10.0))
        std_x = torch.sqrt(var_x.clamp_min(self.eps))

        if mean_x.numel() != self.x_dim or std_x.numel() != self.x_dim:
            raise RuntimeError(
                f"RMS dim mismatch: got mean/std dim {mean_x.numel()} but x_dim={self.x_dim}. "
                f"Check ckpt vs K/use_dt."
            )

        self.mean_x = mean_x
        self.std_x = std_x

        # online state
        self.pos_hist = torch.zeros((self.num_envs, self.K, 3), device=self.device, dtype=torch.float32)
        self.count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        self.v_hat = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)


        print(
            f"[BallVelEstimator] Loaded ckpt={cfg.ckpt_path} "
            f"K={self.K} use_dt={self.use_dt} x_dim={self.x_dim} "
            f"hidden={hidden} ln={use_ln} dropout={dropout}"
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset history for selected envs."""
        if env_ids is None or env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.pos_hist[env_ids] = 0.0
        self.count[env_ids] = 0
        self.v_hat[env_ids] = 0.0

    @torch.no_grad()
    def step(self, pos: torch.Tensor, dt: float, reset_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update estimator with current position.

        Args:
            pos: (N,3) current ball position (suggest: env-local)
            dt:  scalar float (env.dt)
            reset_ids: (k,) env indices to reset

        Returns:
            v_hat: (N,3)
        """
        if reset_ids is not None and reset_ids.numel() > 0:
            self.reset(reset_ids)

        if pos.ndim != 2 or pos.shape != (self.num_envs, 3):
            raise ValueError(f"pos must be shape ({self.num_envs}, 3), got {tuple(pos.shape)}")

        pos = pos.to(device=self.device, dtype=torch.float32, non_blocking=True)

        # push history
        self.pos_hist = torch.roll(self.pos_hist, shifts=-1, dims=1)
        self.pos_hist[:, -1, :].copy_(pos)
        self.count = torch.clamp(self.count + 1, max=self.K)

        ready = self.count >= self.K
        self.v_hat.zero_()

        if ready.any():
            x_pos = self.pos_hist.reshape(self.num_envs, 3 * self.K)
            if self.use_dt:
                dt_col = torch.full((self.num_envs, 1), float(dt), device=self.device, dtype=torch.float32)
                x = torch.cat([x_pos, dt_col], dim=1)
            else:
                x = x_pos

            # normalize
            x_n = (x - self.mean_x) / self.std_x
            x_n = torch.clamp(x_n, -self.clip, self.clip)

            v = self.model(x_n[ready])
            self.v_hat[ready] = v

        return self.v_hat
