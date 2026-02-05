# models/mlp_speed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    in_dim: int = 10          # [p_{t-2}(3), p_{t-1}(3), p_t(3), dt(1)] => 10
    out_dim: int = 3          # v vector (vx, vy, vz)
    hidden: tuple[int, ...] = (128, 128)
    activation: str = "relu"  # "relu" | "elu" | "gelu" | "tanh"
    use_layernorm: bool = False
    dropout: float = 0.0      # 0.0 means no dropout


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class BallSpeedMLP(nn.Module):
    """MLP for estimating ball velocity vector from (pos_{t-2}, pos_{t-1}, pos_t, dt).

    Input:  x (B, 10)
    Output: v_hat (B, 3)
    """

    def __init__(self, cfg: MLPConfig = MLPConfig()):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        act = _make_activation(cfg.activation)

        last_dim = cfg.in_dim
        for h in cfg.hidden:
            layers.append(nn.Linear(last_dim, h))
            if cfg.use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(act)
            if cfg.dropout and cfg.dropout > 0.0:
                layers.append(nn.Dropout(p=float(cfg.dropout)))
            last_dim = h

        layers.append(nn.Linear(last_dim, cfg.out_dim))
        self.net = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Reasonable defaults for MLP regression
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Last layer: smaller init helps stabilize early training
        last = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is not None:
            nn.init.uniform_(last.weight, -1e-3, 1e-3)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.cfg.in_dim:
            raise ValueError(f"Expected x shape (B,{self.cfg.in_dim}), got {tuple(x.shape)}")
        return self.net(x)


@torch.no_grad()
def build_input_from_pos_hist(pos_hist: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Utility to build model input from position history and dt.

    Args:
        pos_hist: (B, 3, 3) where dim1 is time [t-2, t-1, t], dim2 is xyz
        dt: either scalar (), (1,), (B,), or (B,1)

    Returns:
        x: (B, 10) = [p_{t-2}, p_{t-1}, p_t, dt]
    """
    if pos_hist.ndim != 3 or pos_hist.shape[1:] != (3, 3):
        raise ValueError(f"Expected pos_hist shape (B,3,3), got {tuple(pos_hist.shape)}")

    B = pos_hist.shape[0]
    x_pos = pos_hist.reshape(B, 9)

    if dt.ndim == 0:
        dt_b = dt.expand(B, 1)
    elif dt.ndim == 1:
        if dt.numel() == 1:
            dt_b = dt.view(1, 1).expand(B, 1)
        elif dt.numel() == B:
            dt_b = dt.view(B, 1)
        else:
            raise ValueError(f"dt has shape {tuple(dt.shape)} not compatible with batch {B}")
    elif dt.ndim == 2:
        if dt.shape == (B, 1):
            dt_b = dt
        elif dt.shape == (1, 1):
            dt_b = dt.expand(B, 1)
        else:
            raise ValueError(f"dt has shape {tuple(dt.shape)} not compatible with batch {B}")
    else:
        raise ValueError(f"dt must be scalar or 1D/2D tensor, got ndim={dt.ndim}")

    return torch.cat([x_pos, dt_b.to(dtype=x_pos.dtype, device=x_pos.device)], dim=1)
