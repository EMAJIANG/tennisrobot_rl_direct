# tennisrobot_rl/velocity_estimator/data/replay_buffer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch


@dataclass
class ReplayBufferConfig:
    capacity: int = 2_000_000
    # If x_dim == 0, buffer will be lazily initialized on first add()
    x_dim: int = 0
    # If y_dim == 0, buffer will be lazily initialized on first add()
    y_dim: int = 0
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


class SupervisedReplayBuffer:
    """Ring buffer for supervised pairs (x, y).

    - x: (B, x_dim)
    - y: (B, y_dim)

    Supports:
      - add(x, y) where x,y are (B,dim) or single sample (dim,)
      - sample(batch_size) -> (x_b, y_b)
    """

    def __init__(self, cfg: ReplayBufferConfig):
        self.cfg = cfg
        self.capacity = int(cfg.capacity)
        if self.capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {self.capacity}")

        self.x_dim = int(cfg.x_dim) if cfg.x_dim else 0
        self.y_dim = int(cfg.y_dim) if cfg.y_dim else 0

        self._device = torch.device(cfg.device)
        self._dtype = cfg.dtype

        # storage (lazy if dims unknown)
        self._x: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None

        if self.x_dim > 0 and self.y_dim > 0:
            self._alloc(self.x_dim, self.y_dim)

        self._ptr = 0
        self._size = 0

    def _alloc(self, x_dim: int, y_dim: int) -> None:
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        if self.x_dim <= 0 or self.y_dim <= 0:
            raise ValueError(f"Invalid dims: x_dim={self.x_dim}, y_dim={self.y_dim}")

        self._x = torch.empty((self.capacity, self.x_dim), device=self._device, dtype=self._dtype)
        self._y = torch.empty((self.capacity, self.y_dim), device=self._device, dtype=self._dtype)

    def __len__(self) -> int:
        return int(self._size)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _ensure_alloc(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self._x is not None and self._y is not None:
            return
        # lazy init dims
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Lazy init expects x,y to be 2D tensors, got x{tuple(x.shape)} y{tuple(y.shape)}")
        self._alloc(x.shape[1], y.shape[1])

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a batch of samples.

        Args:
            x: (B, x_dim) or (x_dim,)
            y: (B, y_dim) or (y_dim,)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"x,y must be 2D after unsqueeze, got x{tuple(x.shape)} y{tuple(y.shape)}")

        # allocate if needed
        self._ensure_alloc(x, y)
        assert self._x is not None and self._y is not None  # for type checker

        if x.shape[1] != self.x_dim:
            raise ValueError(f"x must be (B,{self.x_dim}), got {tuple(x.shape)}")
        if y.shape[1] != self.y_dim:
            raise ValueError(f"y must be (B,{self.y_dim}), got {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: x {x.shape[0]} vs y {y.shape[0]}")

        B = int(x.shape[0])
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

        if self._x is None or self._y is None:
            raise RuntimeError("Buffer is not initialized (no data added yet).")

        bs = int(batch_size)
        if bs <= 0:
            raise ValueError(f"batch_size must be > 0, got {bs}")

        idx = torch.randint(0, self._size, (bs,), device=self.device)
        return self._x[idx], self._y[idx]

    def state_dict(self) -> dict:
        """Optional: save buffer state (usually you don't need this)."""
        if self._x is None or self._y is None:
            x = torch.empty((0, 0), device=self.device, dtype=self.dtype)
            y = torch.empty((0, 0), device=self.device, dtype=self.dtype)
        else:
            x = self._x[: self._size].clone()
            y = self._y[: self._size].clone()

        return {
            "cfg": {
                "capacity": self.capacity,
                "x_dim": self.x_dim,
                "y_dim": self.y_dim,
                "device": str(self.device),
            },
            "ptr": self._ptr,
            "size": self._size,
            "x": x,
            "y": y,
        }

    def load_state_dict(self, state: dict) -> None:
        """Optional: restore buffer (rarely used)."""
        size = int(state["size"])
        self._ptr = int(state["ptr"])
        self._size = 0

        x = state["x"].to(self.device, dtype=self.dtype)
        y = state["y"].to(self.device, dtype=self.dtype)

        # allocate using loaded dims
        if x.ndim == 2 and y.ndim == 2 and x.shape[1] > 0 and y.shape[1] > 0:
            self._alloc(x.shape[1], y.shape[1])

        self.add(x[:size], y[:size])
