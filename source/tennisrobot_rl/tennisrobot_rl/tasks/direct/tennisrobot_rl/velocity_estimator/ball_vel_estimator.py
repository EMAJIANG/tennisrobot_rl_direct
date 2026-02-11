# ball_vel_estimator.py
from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CkptCfg:
    root_dir: str                                # e.g. "/.../checkpoints_ball_vel"
    run_id: Optional[str] = None                 # None -> auto timestamp
    save_latest_every: int = 200                 # steps
    save_snapshot_every: int = 5000              # steps (0 to disable)
    save_best: bool = True
    keep_last_k_snapshots: int = 5               # delete older snapshots
    filename_latest: str = "ckpt_latest.pt"
    filename_best: str = "ckpt_best.pt"

class BallVelEstimator(nn.Module):
    def __init__(
        self,
        hist_len: int = 8,          # 8 -> 24 pos dims
        hidden_dim: int = 128,
        lr: float = 3e-4,
        device: str | torch.device = "cuda",
        eps_dt: float = 1e-8,
        ckpt_cfg: Optional[CkptCfg] = None,
    ):
        super().__init__()
        self.hist_len = int(hist_len)
        self.hidden_dim = int(hidden_dim)
        self.device = torch.device(device)
        self.eps_dt = float(eps_dt)

        in_dim = self.hist_len * 3 + 1  # 25 when hist_len=8

        # two-layer MLP
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 3),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self._pos_hist: torch.Tensor | None = None

        # ckpt state
        self.ckpt_cfg = ckpt_cfg
        self.run_dir: Optional[str] = None
        self.global_step: int = 0
        self.best_loss: float = float("inf")
        self.loss_ema: Optional[float] = None

        if self.ckpt_cfg is not None:
            self._init_run_dir()

    # ---------------- history ----------------
    def reset(self, env_ids: torch.Tensor | None = None):
        if self._pos_hist is None:
            return
        if env_ids is None:
            self._pos_hist.zero_()
        else:
            env_ids = env_ids.to(dtype=torch.long, device=self.device).flatten()
            if env_ids.numel() > 0:
                self._pos_hist[env_ids] = 0.0

    @torch.no_grad()
    def _ensure_hist(self, N: int, dtype: torch.dtype):
        if self._pos_hist is None or self._pos_hist.shape[0] != N or self._pos_hist.shape[1] != self.hist_len:
            # 用0初始化；你要更稳也可改成“用当前pos填充”
            self._pos_hist = torch.zeros((N, self.hist_len, 3), device=self.device, dtype=dtype)

    # ---------------- ckpt dir ----------------
    def _init_run_dir(self):
        assert self.ckpt_cfg is not None
        root = self.ckpt_cfg.root_dir
        os.makedirs(root, exist_ok=True)

        run_id = self.ckpt_cfg.run_id
        if run_id is None:
            run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.ckpt_cfg.run_id = run_id

        self.run_dir = os.path.join(root, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

    def _ckpt_path_latest(self) -> str:
        assert self.run_dir is not None
        return os.path.join(self.run_dir, self.ckpt_cfg.filename_latest)  # type: ignore

    def _ckpt_path_best(self) -> str:
        assert self.run_dir is not None
        return os.path.join(self.run_dir, self.ckpt_cfg.filename_best)  # type: ignore

    def _ckpt_path_snapshot(self, step: int) -> str:
        assert self.run_dir is not None
        return os.path.join(self.run_dir, f"ckpt_step_{step:08d}.pt")

    def _cleanup_snapshots(self):
        assert self.run_dir is not None
        k = int(self.ckpt_cfg.keep_last_k_snapshots)  # type: ignore
        if k <= 0:
            return
        files = [f for f in os.listdir(self.run_dir) if f.startswith("ckpt_step_") and f.endswith(".pt")]
        files.sort()  # name includes step with zero padding
        if len(files) <= k:
            return
        for f in files[:-k]:
            try:
                os.remove(os.path.join(self.run_dir, f))
            except Exception:
                pass

    # ---------------- save/load ----------------
    def save(self, path: str, extra: Optional[Dict[str, Any]] = None):
        payload = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hist_len": self.hist_len,
            "hidden_dim": self.hidden_dim,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "loss_ema": self.loss_ema,
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    def load(self, path: str, load_optimizer: bool = False, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)

        # 结构检查：防止 hist_len 不一致导致输入维度不对
        if int(ckpt.get("hist_len", self.hist_len)) != self.hist_len:
            raise ValueError(f"hist_len mismatch: ckpt={ckpt.get('hist_len')} vs current={self.hist_len}")
        if int(ckpt.get("hidden_dim", self.hidden_dim)) != self.hidden_dim:
            raise ValueError(f"hidden_dim mismatch: ckpt={ckpt.get('hidden_dim')} vs current={self.hidden_dim}")

        self.net.load_state_dict(ckpt["model"], strict=strict)
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        self.global_step = int(ckpt.get("global_step", 0))
        self.best_loss = float(ckpt.get("best_loss", float("inf")))
        self.loss_ema = ckpt.get("loss_ema", None)

        # 恢复后 history 不可信，建议 reset
        self.reset(None)

    def maybe_checkpoint(self, loss_value: Optional[float] = None):
        """
        Call this each env step (or each estimator step).
        It will save latest/snapshot/best based on ckpt_cfg.
        """
        if self.ckpt_cfg is None:
            return

        if self.run_dir is None:
            self._init_run_dir()

        # latest
        if self.ckpt_cfg.save_latest_every > 0 and (self.global_step % self.ckpt_cfg.save_latest_every == 0):
            self.save(self._ckpt_path_latest())

        # snapshot
        if self.ckpt_cfg.save_snapshot_every and self.ckpt_cfg.save_snapshot_every > 0:
            if self.global_step % self.ckpt_cfg.save_snapshot_every == 0:
                self.save(self._ckpt_path_snapshot(self.global_step))
                self._cleanup_snapshots()

        # best
        if self.ckpt_cfg.save_best and (loss_value is not None):
            if loss_value < self.best_loss:
                self.best_loss = float(loss_value)
                self.save(self._ckpt_path_best())

    # ---------------- core step ----------------
    def step(
        self,
        pos: torch.Tensor,                    # (N,3)
        dt: float | torch.Tensor,             # scalar or (N,) or (N,1)
        vel_gt: torch.Tensor | None = None,   # (N,3)
        reset_ids: torch.Tensor | None = None,
        online_train: bool = True,
    ):
        pos = pos.to(self.device)
        N = pos.shape[0]
        self._ensure_hist(N, pos.dtype)

        if reset_ids is not None and reset_ids.numel() > 0:
            self.reset(reset_ids)

        self._pos_hist = torch.roll(self._pos_hist, shifts=-1, dims=1)
        self._pos_hist[:, -1, :] = pos

        x_pos = self._pos_hist.reshape(N, self.hist_len * 3)

        # dt -> (N,1)
        if not torch.is_tensor(dt):
            dt_t = torch.tensor(dt, device=self.device, dtype=pos.dtype)
        else:
            dt_t = dt.to(device=self.device, dtype=pos.dtype)

        if dt_t.ndim == 0:
            dt_t = dt_t.expand(N).reshape(N, 1)
        elif dt_t.ndim == 1:
            dt_t = dt_t.reshape(N, 1)
        else:
            dt_t = dt_t.reshape(N, 1)

        dt_t = dt_t.clamp_min(self.eps_dt)
        x = torch.cat([x_pos, dt_t], dim=1)  # (N, 25) when hist_len=8

        loss = None

        if online_train and (vel_gt is not None):
            vel_gt = vel_gt.to(self.device)

            # 训练分支：必须在 enable_grad 里重新 forward（不能复用外面算的 vel_hat）
            with torch.enable_grad():
                vel_hat = self.net(x)
                loss = F.mse_loss(vel_hat, vel_gt)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            # （可选）update EMA：如果你 env 里要打印 ema
            loss_val = float(loss.detach().item())
            if getattr(self, "loss_ema", None) is None:
                self.loss_ema = loss_val
            else:
                beta = getattr(self, "ema_beta", 0.98)
                self.loss_ema = beta * self.loss_ema + (1.0 - beta) * loss_val

        else:
            # 推理分支：不需要梯度
            with torch.no_grad():
                vel_hat = self.net(x)

        # update step + checkpoint
        self.global_step += 1
        loss_value = float(loss.detach().item()) if loss is not None else None
        self.maybe_checkpoint(loss_value=loss_value)

        return vel_hat, loss
