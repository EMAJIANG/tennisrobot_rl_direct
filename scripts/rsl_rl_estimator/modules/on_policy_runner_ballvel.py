from __future__ import annotations

import os
import time
from collections import deque
from copy import deepcopy

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.modules import ActorCritic, EmpiricalNormalization

from .ppo_with_ballvel import PPOWithBallVelEstimator


def _strip_meta(d: dict) -> dict:
    """Remove hydra/meta keys that should never be passed to constructors."""
    if d is None:
        return {}
    out = deepcopy(d)
    # common hydra / dynamic-loading metadata keys
    for k in ["class_name", "_target_", "name"]:
        out.pop(k, None)
    # also drop any private keys (rare but safe)
    for k in list(out.keys()):
        if isinstance(k, str) and k.startswith("_"):
            out.pop(k, None)
    return out


class OnPolicyRunnerBallVel(OnPolicyRunner):
    """Custom runner for PPO + online ball-velocity estimator.

    Differences vs rsl_rl OnPolicyRunner:
    - Custom learn() loop (IsaacLab VecEnv-style observations/infos)
    - Custom logging (baseline-style) to avoid depending on base runner locals()
    - Custom save() to avoid base runner field dependencies
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        # keep same field names as rsl_rl runner expects
        self.cfg = train_cfg
        self.alg_cfg = _strip_meta(train_cfg.get("algorithm", {}))
        self.policy_cfg = _strip_meta(train_cfg.get("policy", {}))
        self.estimator_cfg = train_cfg.get("ball_vel_estimator", {}) or {}

        # normalize device
        self.device = torch.device(device) if isinstance(device, str) else device

        self.env = env
        self.log_dir = log_dir

        # required by base runner / external code
        self.num_steps_per_env = int(train_cfg["num_steps_per_env"])
        self.save_interval = int(train_cfg.get("save_interval", 1000))
        self.current_learning_iteration = int(train_cfg.get("resume_iteration", 0))

        # ---- Multi-GPU fields expected by some utilities (even if you don't use DDP)
        self.writer = None
        self.disable_logs = False
        self.logger_type = str(self.cfg.get("logger", "tensorboard")).lower()
        try:
            self._configure_multi_gpu()  # sets gpu_world_size, etc.
        except Exception:
            # safe defaults for non-DDP runs
            self.gpu_world_size = 1
            self.gpu_rank = 0

        # totals for logging
        self.tot_timesteps = 0
        self.tot_time = 0.0

        # --- get obs shapes (IsaacLab-style: env.get_observations() returns (policy_obs, extras))
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        critic_obs = extras.get("observations", {}).get("critic", obs)
        num_priv_obs = critic_obs.shape[1]

        # --- normalizers
        self.obs_normalizer = EmpiricalNormalization(shape=[num_obs]).to(self.device)
        self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_priv_obs]).to(self.device)

        # --- build policy
        policy = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_priv_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # --- estimator integration config (safe defaults)
        est_model_cfg = self.estimator_cfg.get("model", {}) or {}
        num_prop = int(self.estimator_cfg.get("num_prop", num_obs))
        ball_vel_policy_offset = int(self.estimator_cfg.get("ball_vel_policy_offset", 0))
        ball_vel_critic_offset = int(self.estimator_cfg.get("ball_vel_critic_offset", 0))

        # --- build algorithm (PPO + estimator)
        # CRITICAL: force PPO device to match runner device (avoid PPO defaulting to CPU)
        self.alg_cfg["device"] = self.device

        self.alg = PPOWithBallVelEstimator(
            policy=policy,
            estimator_cfg=est_model_cfg,
            num_prop=num_prop,
            ball_vel_policy_offset=ball_vel_policy_offset,
            ball_vel_critic_offset=ball_vel_critic_offset,
            train_with_estimated_ball_vel=bool(self.estimator_cfg.get("train_with_estimated_ball_vel", False)),
            estimator_loss_coef=float(self.estimator_cfg.get("estimator_loss_coef", 1.0)),
            estimator_lr=float(self.estimator_cfg.get("lr", 3e-4)),
            **self.alg_cfg,
        )
        self.policy = policy

        # init storage (signature matches your rsl_rl version)
        self.alg.init_storage(
            training_type="rl",
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_obs],
            critic_obs_shape=[num_priv_obs],
            actions_shape=[self.env.num_actions],
        )

    # ---------------------------------------------------------------------
    # Logging (baseline-style, independent from base runner log())
    # ---------------------------------------------------------------------
    def log_ballvel(self, locs: dict, width: int = 80, pad: int = 35):
        """Baseline-style logger for this custom runner (does NOT rely on base OnPolicyRunner.log keys)."""
        gpu_world_size = getattr(self, "gpu_world_size", 1)
        collection_size = int(self.num_steps_per_env * self.env.num_envs * gpu_world_size)

        # update totals
        self.tot_timesteps = int(self.tot_timesteps + collection_size)
        self.tot_time = float(self.tot_time + locs["collection_time"] + locs["learn_time"])

        iter_time = float(locs["collection_time"] + locs["learn_time"])
        fps = int(collection_size / max(1e-8, iter_time))

        # -------- tensorboard --------
        if self.writer is not None:
            self.writer.add_scalar("Perf/fps", fps, locs["it"])
            self.writer.add_scalar("Perf/collection_time", float(locs["collection_time"]), locs["it"])
            self.writer.add_scalar("Perf/learn_time", float(locs["learn_time"]), locs["it"])

            loss_dict = locs.get("loss_dict", {}) or {}
            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f"Loss/{k}", float(v), locs["it"])

            # episode stats
            if len(locs.get("rewbuffer", [])) > 0:
                import statistics
                self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            if len(locs.get("lenbuffer", [])) > 0:
                import statistics
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        # -------- episode infos (TB + terminal) --------
        ep_infos = locs.get("ep_infos", []) or []
        ep_string = ""

        def _flatten_ep(ep: dict) -> dict:
            # support nesting: {"episode": {...}} or {"log": {...}}
            if isinstance(ep, dict):
                if "episode" in ep and isinstance(ep["episode"], dict):
                    return ep["episode"]
                if "log" in ep and isinstance(ep["log"], dict):
                    return ep["log"]
            return ep if isinstance(ep, dict) else {}

        if len(ep_infos) > 0 and isinstance(ep_infos, list) and isinstance(ep_infos[0], dict):
            flat0 = _flatten_ep(ep_infos[0])
            for key in flat0.keys():
                vals = []
                for ep in ep_infos:
                    ep_flat = _flatten_ep(ep)
                    if key not in ep_flat:
                        continue
                    x = ep_flat[key]
                    if torch.is_tensor(x):
                        x = x.detach().float().mean().item()
                    else:
                        try:
                            x = float(x)
                        except Exception:
                            continue
                    vals.append(x)

                if len(vals) > 0:
                    v = sum(vals) / len(vals)
                    if self.writer is not None:
                        self.writer.add_scalar(f"Episode/{key}", float(v), locs["it"])
                    ep_string += f"{f'Episode {key}:':>{pad}} {v:.4f}\n"

        # -------- terminal print --------
        title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"{'#' * width}\n"
            f"{title.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps} steps/s (collection {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
        )

        loss_dict = locs.get("loss_dict", {}) or {}
        if isinstance(loss_dict, dict):
            for k, v in loss_dict.items():
                log_string += f"{f'Mean {k}:':>{pad}} {float(v):.6f}\n"

        rewbuffer = locs.get("rewbuffer", None)
        lenbuffer = locs.get("lenbuffer", None)
        if rewbuffer is not None and len(rewbuffer) > 0:
            import statistics
            log_string += f"{'Mean reward:':>{pad}} {statistics.mean(rewbuffer):.4f}\n"
        if lenbuffer is not None and len(lenbuffer) > 0:
            import statistics
            log_string += f"{'Mean ep length:':>{pad}} {statistics.mean(lenbuffer):.2f}\n"

        # ✅ print the small episode items
        log_string += ep_string

        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iter_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"
        )
        print(log_string)

    # ---------------------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Collect rollouts + update PPO (with estimator) and log."""

        # init writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # initial obs
        obs, extras = self.env.get_observations()
        critic_obs = extras.get("observations", {}).get("critic", obs)

        obs = self.obs_normalizer(obs.to(self.device))
        critic_obs = self.privileged_obs_normalizer(critic_obs.to(self.device))

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + int(num_learning_iterations)

        for it in range(start_iter, tot_iter):
            # -------- Rollout / collection --------
            t_collect0 = time.time()
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    # env likely runs on env.device; actions must be on env.device
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

                    # move to runner device
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # normalize + critic obs
                    obs = self.obs_normalizer(obs)
                    critic_obs = infos.get("observations", {}).get("critic", obs).to(self.device)
                    critic_obs = self.privileged_obs_normalizer(critic_obs)

                    # PPO expects time_outs on alg device
                    if "time_outs" in infos and torch.is_tensor(infos["time_outs"]):
                        infos["time_outs"] = infos["time_outs"].to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)

                    # book-keeping
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if new_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        # store episode info dicts (some envs nest under 'episode' or 'log')
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

            collection_time = time.time() - t_collect0

            # -------- Learn / update --------
            t_learn0 = time.time()
            self.alg.compute_returns(critic_obs)
            loss_dict = self.alg.update()
            learn_time = time.time() - t_learn0

            self.current_learning_iteration = it + 1

            # -------- Logging / saving --------
            if self.log_dir is not None and not self.disable_logs:
                self.log_ballvel(
                    {
                        "it": it,
                        "tot_iter": tot_iter,
                        "collection_time": float(collection_time),
                        "learn_time": float(learn_time),
                        "loss_dict": loss_dict,
                        "rewbuffer": rewbuffer,
                        "lenbuffer": lenbuffer,
                        "ep_infos": ep_infos,
                    }
                )

                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

    # ---------------------------------------------------------------------
    # Custom save (do NOT call base runner save)
    # ---------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "policy_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_state_dict": self.alg.estimator.state_dict(),
            "estimator_optim_state_dict": self.alg.estimator_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
        }
        # save normalizers if present
        if hasattr(self, "obs_normalizer") and hasattr(self.obs_normalizer, "state_dict"):
            ckpt["obs_normalizer"] = self.obs_normalizer.state_dict()
        if hasattr(self, "privileged_obs_normalizer") and hasattr(self.privileged_obs_normalizer, "state_dict"):
            ckpt["privileged_obs_normalizer"] = self.privileged_obs_normalizer.state_dict()

        torch.save(ckpt, path)

        deploy = {
            "actor_state_dict": self.alg.policy.actor.state_dict(),   # 部署只要 actor
            "estimator_state_dict": self.alg.estimator.state_dict(),
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "privileged_obs_normalizer": self.privileged_obs_normalizer.state_dict(),  # 部署如果不用 critic 可不加载
            "meta": {
                "num_prop": self.alg.num_prop,
                "ball_vel_policy_offset": self.alg.ball_vel_policy_offset,
                "ball_vel_critic_offset": self.alg.ball_vel_critic_offset,
                "train_with_estimated_ball_vel": self.alg.train_with_estimated_ball_vel,
            },
        }
        torch.save(deploy, os.path.join(os.path.dirname(path), "deployment.pt"))


    def load(self, path: str, load_optimizer: bool = True):
        
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)

        # -------------------------
        # 1) policy state dict (兼容两种格式)
        # -------------------------
        if "model_state_dict" in loaded_dict:
            policy_sd = loaded_dict["model_state_dict"]          # rsl_rl 原生
        elif "policy_state_dict" in loaded_dict:
            policy_sd = loaded_dict["policy_state_dict"]         # 你自定义 save()
        elif "model" in loaded_dict:
            policy_sd = loaded_dict["model"]                     # 兜底
        else:
            raise KeyError(
                f"Unknown checkpoint format: cannot find policy state dict key. "
                f"Keys={list(loaded_dict.keys())[:50]}"
            )

        # rsl_rl 的 ActorCritic.load_state_dict 有时返回 bool resumed_training（某些分支）
        # 也可能返回 Missing/Unexpected keys 的对象（torch 标准行为）
        try:
            resumed_training = self.alg.policy.load_state_dict(policy_sd)
        except Exception:
            # 有些情况下需要 strict=False
            resumed_training = self.alg.policy.load_state_dict(policy_sd, strict=False)

        # 统一成 bool（保守：能正常load就认为 resumed_training=True）
        resumed_training_flag = True
        if isinstance(resumed_training, bool):
            resumed_training_flag = resumed_training

        # -------------------------
        # 2) estimator（你的 save 格式里一定有）
        # -------------------------
        if hasattr(self.alg, "estimator") and self.alg.estimator is not None:
            if "estimator_state_dict" in loaded_dict:
                try:
                    self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"])
                except Exception:
                    self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"], strict=False)
            else:
                warnings.warn("'estimator_state_dict' not found in checkpoint, estimator not loaded.")

        # -------------------------
        # 3) RND（如果有）
        # -------------------------
        if getattr(self.alg, "rnd", None) is not None:
            if "rnd_state_dict" in loaded_dict:
                self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            else:
                warnings.warn("'rnd_state_dict' not found, RND not loaded.")

        # -------------------------
        # 4) normalizers（兼容两套 key）
        # -------------------------
        # 你自定义 save():
        if "obs_normalizer" in loaded_dict and hasattr(self, "obs_normalizer"):
            try:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_normalizer"])
            except Exception:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_normalizer"], strict=False)

        if "privileged_obs_normalizer" in loaded_dict and hasattr(self, "privileged_obs_normalizer"):
            try:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_normalizer"])
            except Exception:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_normalizer"], strict=False)

        # rsl_rl 原生 save() 的 key（baseline 里那套）
        # 注意：baseline 有“resumed_training 决定加载哪个”的逻辑，这里也尽量保持
        if "obs_norm_state_dict" in loaded_dict and hasattr(self, "obs_normalizer"):
            if resumed_training_flag:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                if "privileged_obs_norm_state_dict" in loaded_dict and hasattr(self, "privileged_obs_normalizer"):
                    self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # 某些情况下 baseline 用 obs_norm_state_dict 去加载 privileged
                if hasattr(self, "privileged_obs_normalizer"):
                    self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])

        # -------------------------
        # 5) optimizers（可选）
        # -------------------------
        if load_optimizer and resumed_training_flag:
            if "optimizer_state_dict" in loaded_dict and hasattr(self.alg, "optimizer"):
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

            if "estimator_optim_state_dict" in loaded_dict and hasattr(self.alg, "estimator_optimizer"):
                self.alg.estimator_optimizer.load_state_dict(loaded_dict["estimator_optim_state_dict"])

            if getattr(self.alg, "rnd", None) is not None and "rnd_optimizer_state_dict" in loaded_dict:
                if hasattr(self.alg, "rnd_optimizer"):
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        # -------------------------
        # 6) iteration + infos
        # -------------------------
        if resumed_training_flag and "iter" in loaded_dict:
            self.current_learning_iteration = int(loaded_dict["iter"])

        # baseline 会 return infos；你的 save 里可能没有 infos
        if "infos" in loaded_dict:
            return loaded_dict["infos"]
        return loaded_dict