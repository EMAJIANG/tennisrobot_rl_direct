# data/replay_buffer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class ReplayBufferConfig:
    capacity: int = 2_000_000
    x_dim: int = 10
    y_dim: int = 3
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


class SupervisedReplayBuffer:
    """Ring buffer for supervised pairs (x,y).

    - x: (N, x_dim)
    - y: (N, y_dim)

    Supports:
      - add(x, y) with x,y shaped (B, dim)
      - sample(batch_size) -> (x_b, y_b)
    """

    def __init__(self, cfg: ReplayBufferConfig):
        self.cfg = cfg
        self.capacity = int(cfg.capacity)
        self.x_dim = int(cfg.x_dim)
        self.y_dim = int(cfg.y_dim)

        device = torch.device(cfg.device)
        dtype = cfg.dtype

        self._x = torch.empty((self.capacity, self.x_dim), device=device, dtype=dtype)
        self._y = torch.empty((self.capacity, self.y_dim), device=device, dtype=dtype)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return int(self._size)

    @property
    def device(self) -> torch.device:
        return self._x.device

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a batch of samples.

        Args:
            x: (B, x_dim)
            y: (B, y_dim)
        """
        if x.ndim != 2 or x.shape[1] != self.x_dim:
            raise ValueError(f"x must be (B,{self.x_dim}), got {tuple(x.shape)}")
        if y.ndim != 2 or y.shape[1] != self.y_dim:
            raise ValueError(f"y must be (B,{self.y_dim}), got {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: x {x.shape[0]} vs y {y.shape[0]}")

        B = x.shape[0]
        if B == 0:
            return

        # ensure on buffer device/dtype
        x = x.to(device=self.device, dtype=self._x.dtype, non_blocking=True)
        y = y.to(device=self.device, dtype=self._y.dtype, non_blocking=True)

        # If incoming batch larger than capacity, keep only the last 'capacity' samples.
        if B >= self.capacity:
            x = x[-self.capacity :]
            y = y[-self.capacity :]
            B = self.capacity

        end = self._ptr + B
        if end <= self.capacity:
            self._x[self._ptr:end].copy_(x)
            self._y[self._ptr:end].copy_(y)
        else:
            first = self.capacity - self._ptr
            second = end - self.capacity
            self._x[self._ptr:].copy_(x[:first])
            self._y[self._ptr:].copy_(y[:first])
            self._x[:second].copy_(x[first:])
            self._y[:second].copy_(y[first:])

        self._ptr = end % self.capacity
        self._size = min(self.capacity, self._size + B)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform random sampling."""
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        bs = int(batch_size)
        if bs <= 0:
            raise ValueError(f"batch_size must be > 0, got {bs}")

        # sample indices in [0, size)
        idx = torch.randint(0, self._size, (bs,), device=self.device)
        return self._x[idx], self._y[idx]

    def state_dict(self) -> dict:
        """Optional: save buffer state (usually you don't need this)."""
        return {
            "cfg": {
                "capacity": self.capacity,
                "x_dim": self.x_dim,
                "y_dim": self.y_dim,
            },
            "ptr": self._ptr,
            "size": self._size,
            "x": self._x[: self._size].clone(),
            "y": self._y[: self._size].clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Optional: restore buffer (rarely used)."""
        size = int(state["size"])
        self._ptr = int(state["ptr"])
        self._size = 0

        x = state["x"].to(self.device, dtype=self._x.dtype)
        y = state["y"].to(self.device, dtype=self._y.dtype)

        self.add(x[:size], y[:size])
