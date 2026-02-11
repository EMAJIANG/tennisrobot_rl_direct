# scripts/rsl_rl_estimator/modules/ball_vel_estimator.py
from __future__ import annotations

import torch
import torch.nn as nn


class BallVelEstimator(nn.Module):
    """Simple MLP that predicts 3D ball velocity from the first num_prop dims of policy obs."""

    def __init__(
        self,
        num_prop: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: str = "elu",
    ):
        super().__init__()
        self.num_prop = int(num_prop)

        act = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}[activation.lower()]
        layers: list[nn.Module] = []
        in_dim = self.num_prop
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act()]
            in_dim = h
        layers += [nn.Linear(in_dim, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs[:, : self.num_prop]
        return self.net(x)
