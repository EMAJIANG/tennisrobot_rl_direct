# tennisrobot_rl/velocity_estimator/envs/observation_buffer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class PositionHistoryConfig:
    # number of historical frames to keep: K
    hist_len: int = 3
    # whether to append dt as the last feature
    use_dt_input: bool = True
    # if set (>0), validate against computed dim; if 0, auto compute
    x_dim: int = 0


class PositionHistoryBuffer:
    """Per-env position history buffer for building supervised features.

    Stores:
      pos_hist: (N, K, 3) with oldest->newest along dim=1
      count:    (N,) number of frames seen since last reset (clamped to K)

    Features:
      x = flatten(pos_hist) -> (N, 3K)
      if use_dt_input: append dt -> (N, 3K+1)
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        cfg: PositionHistoryConfig = PositionHistoryConfig(),
    ):
        if cfg.hist_len <= 0:
            raise ValueError(f"hist_len must be > 0, got {cfg.hist_len}")

        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype

        self.dt_value = float(dt)
        self._dt = torch.tensor(self.dt_value, device=self.device, dtype=self.dtype)

        self.K = int(cfg.hist_len)
        self.x_dim = 3 * self.K + (1 if cfg.use_dt_input else 0)
        if cfg.x_dim and int(cfg.x_dim) != self.x_dim:
            raise ValueError(
                f"x_dim mismatch: cfg.x_dim={cfg.x_dim} but computed={self.x_dim} "
                f"(hist_len={cfg.hist_len}, use_dt_input={cfg.use_dt_input})"
            )

        self.pos_hist = torch.zeros((self.num_envs, self.K, 3), device=self.device, dtype=self.dtype)
        self._count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

    @property
    def dt(self) -> torch.Tensor:
        return self._dt

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset history for given env_ids."""
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.pos_hist[env_ids] = 0.0
        self._count[env_ids] = 0

    def push(self, pos_t: torch.Tensor) -> None:
        """Push current position for all envs. pos_t: (N,3)."""
        if pos_t.ndim != 2 or pos_t.shape != (self.num_envs, 3):
            raise ValueError(f"pos_t must be shape ({self.num_envs}, 3), got {tuple(pos_t.shape)}")

        pos_t = pos_t.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # roll left, newest goes to last
        self.pos_hist = torch.roll(self.pos_hist, shifts=-1, dims=1)
        self.pos_hist[:, -1, :].copy_(pos_t)

        self._count = torch.clamp(self._count + 1, max=self.K)

    def ready_mask(self) -> torch.Tensor:
        return self._count >= self.K

    def get_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return x: (N, x_dim), mask: (N,) bool."""
        x_pos = self.pos_hist.reshape(self.num_envs, 3 * self.K)  # (N, 3K)

        if self.cfg.use_dt_input:
            dt_col = self._dt.view(1, 1).expand(self.num_envs, 1)  # (N,1)
            x = torch.cat([x_pos, dt_col], dim=1)                  # (N, 3K+1)
        else:
            x = x_pos

        return x, self.ready_mask()
