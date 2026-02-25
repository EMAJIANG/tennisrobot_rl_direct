# serve_hit_logger.py
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ServeHitLoggerCfg:
    out_dir: str
    filename: str = "serve_hit_pairs.csv"
    flush_every: int = 2048
    write_header: bool = True
    fsync: bool = False


class ServeHitLogger:
    """
    One row per serve per env.
    - start_serve(): create/overwrite a pending serve record for env_ids
    - mark_hit(): fill hit fields (first hit only)
    - finalize(env_ids): write pending rows for env_ids (hit or no-hit). no-hit => hit_pos=(0,0,0)
    - close(): finalize all pending and flush
    """

    def __init__(self, cfg: ServeHitLoggerCfg, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)

        os.makedirs(self.cfg.out_dir, exist_ok=True)
        self.path = os.path.join(self.cfg.out_dir, self.cfg.filename)

        self._buf: List[List[Any]] = []
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._serve_id: Optional[torch.Tensor] = None
        self._num_envs: Optional[int] = None

        if self.cfg.write_header:
            self._maybe_write_header()

    def attach_num_envs(self, num_envs: int):
        if self._num_envs is not None:
            return
        self._num_envs = int(num_envs)
        self._serve_id = torch.zeros(self._num_envs, dtype=torch.long)

    def _maybe_write_header(self):
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            return
        header = [
            "wall_time", "global_step",
            "env_id", "serve_id", "episode_id",

            # serve (ENV-LOCAL !!!)
            "serve_x", "serve_y", "serve_z",
            "serve_vx", "serve_vy", "serve_vz",

            # hit (ENV-LOCAL !!!)
            "hit_flag", "t_hit",
            "hit_x", "hit_y", "hit_z",
            "paddle_x", "paddle_y", "paddle_z",
        ]
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            if self.cfg.fsync:
                f.flush()
                os.fsync(f.fileno())

    def flush(self, force: bool = False):
        if (not force) and (len(self._buf) < self.cfg.flush_every):
            return
        if not self._buf:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerows(self._buf)
            if self.cfg.fsync:
                f.flush()
                os.fsync(f.fileno())
        self._buf.clear()

    def close(self):
        if self._pending:
            env_ids = torch.tensor(list(self._pending.keys()), dtype=torch.long, device=self.device)
            self.finalize(env_ids)
        self.flush(force=True)

    @torch.no_grad()
    def start_serve(
        self,
        env_ids: torch.Tensor,               # (M,)
        episode_ids: Optional[torch.Tensor], # (M,) or None
        serve_pos_local: torch.Tensor,       # (M,3)  ENV-LOCAL
        serve_vel_local: torch.Tensor,       # (M,3)  ENV-LOCAL
        global_step: int = -1,
    ):
        if env_ids.numel() == 0:
            return
        if self._serve_id is None:
            raise RuntimeError("Call attach_num_envs(num_envs) once before start_serve().")

        env_ids_cpu = env_ids.detach().to("cpu", non_blocking=True).long().numpy()
        sp = serve_pos_local.detach().to("cpu", non_blocking=True).float().numpy()
        sv = serve_vel_local.detach().to("cpu", non_blocking=True).float().numpy()

        ep_cpu = None
        if episode_ids is not None:
            ep_cpu = episode_ids.detach().to("cpu", non_blocking=True).long().numpy()

        wt = time.time()
        gs = int(global_step)

        for i in range(env_ids_cpu.shape[0]):
            eid = int(env_ids_cpu[i])

            self._serve_id[eid] += 1
            sid = int(self._serve_id[eid].item())
            ep = int(ep_cpu[i]) if ep_cpu is not None else -1

            # overwrite pending (if any)
            self._pending[eid] = {
                "wall_time": wt,
                "global_step": gs,
                "env_id": eid,
                "serve_id": sid,
                "episode_id": ep,

                "serve": (float(sp[i, 0]), float(sp[i, 1]), float(sp[i, 2])),
                "serve_v": (float(sv[i, 0]), float(sv[i, 1]), float(sv[i, 2])),

                "hit_flag": 0,
                "t_hit": -1,
                "hit": (0.0, 0.0, 0.0),
                "paddle": (0.0, 0.0, 0.0),
            }

    @torch.no_grad()
    def mark_hit(
        self,
        env_ids: torch.Tensor,              # (K,)
        t_hit: torch.Tensor,                # (K,)
        hit_pos_local: torch.Tensor,        # (K,3) ENV-LOCAL
        paddle_pos_local: torch.Tensor,     # (K,3) ENV-LOCAL
    ):
        if env_ids.numel() == 0:
            return

        env_ids_cpu = env_ids.detach().to("cpu", non_blocking=True).long().numpy()
        t_cpu = t_hit.detach().to("cpu", non_blocking=True).long().numpy()
        hp = hit_pos_local.detach().to("cpu", non_blocking=True).float().numpy()
        pp = paddle_pos_local.detach().to("cpu", non_blocking=True).float().numpy()

        for i in range(env_ids_cpu.shape[0]):
            eid = int(env_ids_cpu[i])
            rec = self._pending.get(eid, None)
            if rec is None:
                continue
            if rec["hit_flag"] == 1:
                continue  # first hit only

            rec["hit_flag"] = 1
            rec["t_hit"] = int(t_cpu[i])
            rec["hit"] = (float(hp[i, 0]), float(hp[i, 1]), float(hp[i, 2]))
            rec["paddle"] = (float(pp[i, 0]), float(pp[i, 1]), float(pp[i, 2]))

    @torch.no_grad()
    def finalize(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        env_ids_cpu = env_ids.detach().to("cpu", non_blocking=True).long().numpy()

        for eid_np in env_ids_cpu:
            eid = int(eid_np)
            rec = self._pending.pop(eid, None)
            if rec is None:
                continue
            self._buf.append(self._row(rec))

        self.flush(force=False)

    def _row(self, rec: Dict[str, Any]) -> List[Any]:
        sx, sy, sz = rec["serve"]
        svx, svy, svz = rec["serve_v"]
        hflag = int(rec["hit_flag"])
        thit = int(rec["t_hit"])
        hx, hy, hz = rec["hit"]
        px, py, pz = rec["paddle"]

        return [
            rec["wall_time"], rec["global_step"],
            rec["env_id"], rec["serve_id"], rec["episode_id"],
            sx, sy, sz,
            svx, svy, svz,
            hflag, thit,
            hx, hy, hz,
            px, py, pz,
        ]
