# envs/observation_buffer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class PositionHistoryConfig:
    hist_len: int = 3          # we need [t-2, t-1, t]
    use_dt_input: bool = True  # append dt as the last feature
    x_dim: int = 10            # 3*3 + 1


class PositionHistoryBuffer:
    """Maintain per-env position history of length hist_len.

    For our task:
      - hist_len = 3
      - pos_hist: (N, 3, 3) time-major [t-2, t-1, t]
      - When not enough frames collected after reset, mask is False.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        cfg: PositionHistoryConfig = PositionHistoryConfig(),
    ):
        if cfg.hist_len != 3:
            raise ValueError("This buffer is designed for hist_len=3 (t-2,t-1,t).")
        if cfg.use_dt_input and cfg.x_dim != 10:
            raise ValueError("If use_dt_input=True, x_dim must be 10.")
        if (not cfg.use_dt_input) and cfg.x_dim != 9:
            raise ValueError("If use_dt_input=False, x_dim must be 9.")

        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.dt_value = float(dt)

        self.device = torch.device(device)
        self.dtype = dtype

        self.pos_hist = torch.zeros((self.num_envs, cfg.hist_len, 3), device=self.device, dtype=self.dtype)
        # how many frames already pushed since last reset (0..hist_len)
        self._count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        # dt tensor cached
        self._dt = torch.tensor(self.dt_value, device=self.device, dtype=self.dtype)

    @property
    def dt(self) -> torch.Tensor:
        return self._dt

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset history for env_ids. env_ids: (k,) long/int."""
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device)
        self.pos_hist[env_ids] = 0.0
        self._count[env_ids] = 0

    def push(self, pos_t: torch.Tensor) -> None:
        """Push current position for all envs.

        Args:
            pos_t: (N,3) in world frame
        """
        if pos_t.ndim != 2 or pos_t.shape != (self.num_envs, 3):
            raise ValueError(f"Expected pos_t shape {(self.num_envs,3)}, got {tuple(pos_t.shape)}")

        pos_t = pos_t.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # roll time axis left: [t-2,t-1,t] -> [t-1,t,t] then write last as pos_t
        self.pos_hist = torch.roll(self.pos_hist, shifts=-1, dims=1)
        self.pos_hist[:, -1, :].copy_(pos_t)

        # update counts up to hist_len
        self._count = torch.clamp(self._count + 1, max=self.cfg.hist_len)

    def ready_mask(self) -> torch.Tensor:
        """Return (N,) bool mask indicating whether we have full history (>=hist_len)."""
        return self._count >= self.cfg.hist_len

    def get_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return:
        - x: (N, x_dim)
        - mask: (N,) bool
        """
        # flatten positions
        x_pos = self.pos_hist.reshape(self.num_envs, 9)

        if self.cfg.use_dt_input:
            dt_col = self._dt.view(1, 1).expand(self.num_envs, 1)
            x = torch.cat([x_pos, dt_col], dim=1)
        else:
            x = x_pos

        mask = self.ready_mask()
        return x, mask
