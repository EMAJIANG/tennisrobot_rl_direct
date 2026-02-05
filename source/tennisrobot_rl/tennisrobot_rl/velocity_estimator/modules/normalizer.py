# modules/normalizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NormalizerConfig:
    eps: float = 1e-8
    clip: Optional[float] = 10.0  # None means no clipping


class RunningMeanStd:
    """Running mean/std for vector features with numerically stable updates.

    Keeps:
      - count (scalar)
      - mean  (D,)
      - var   (D,)  (population variance)
    """

    def __init__(
        self,
        dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        cfg: NormalizerConfig = NormalizerConfig(),
    ):
        self.dim = int(dim)
        self.cfg = cfg

        self.count = torch.tensor(0.0, device=device, dtype=dtype)
        self.mean = torch.zeros(self.dim, device=device, dtype=dtype)
        self.var = torch.ones(self.dim, device=device, dtype=dtype)  # start with 1 to avoid div0

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var.clamp_min(self.cfg.eps))

    def to(self, device: torch.device | str) -> "RunningMeanStd":
        self.count = self.count.to(device)
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running stats with a batch x of shape (B, D)."""
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x shape (B,{self.dim}), got {tuple(x.shape)}")

        x = x.detach()
        batch_count = x.shape[0]
        if batch_count == 0:
            return

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)  # population var

        # Parallel update (Chan et al.)
        delta = batch_mean - self.mean
        total_count = self.count + float(batch_count)

        new_mean = self.mean + delta * (float(batch_count) / total_count)
        m_a = self.var * self.count
        m_b = batch_var * float(batch_count)
        M2 = m_a + m_b + (delta * delta) * (self.count * float(batch_count) / total_count)
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x using (x-mean)/std. x can be (..., D)."""
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")
        x_n = (x - self.mean) / self.std
        if self.cfg.clip is not None:
            x_n = torch.clamp(x_n, -float(self.cfg.clip), float(self.cfg.clip))
        return x_n

    def denormalize(self, x_n: torch.Tensor) -> torch.Tensor:
        """Inverse transform: x = x_n*std + mean. x_n can be (..., D)."""
        if x_n.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x_n.shape[-1]}")
        return x_n * self.std + self.mean

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "cfg": {
                "eps": float(self.cfg.eps),
                "clip": None if self.cfg.clip is None else float(self.cfg.clip),
            },
            "count": self.count,
            "mean": self.mean,
            "var": self.var,
        }

    def load_state_dict(self, state: dict) -> None:
        dim = int(state.get("dim", self.dim))
        if dim != self.dim:
            raise ValueError(f"RMS dim mismatch: state dim={dim}, self dim={self.dim}")

        cfg = state.get("cfg", {})
        self.cfg = NormalizerConfig(
            eps=float(cfg.get("eps", self.cfg.eps)),
            clip=cfg.get("clip", self.cfg.clip),
        )

        self.count = state["count"]
        self.mean = state["mean"]
        self.var = state["var"]


@torch.no_grad()
def compute_mean_std_offline(x: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean/std from a dataset tensor x of shape (N, D)."""
    if x.ndim != 2:
        raise ValueError(f"Expected x shape (N,D), got {tuple(x.shape)}")
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(eps)
    return mean, std
