# SPDX-License-Identifier: BSD-3-Clause
"""
Train RL agent with RSL-RL (IsaacLab-style entry) + online ball-velocity estimator.

This script follows the official IsaacLab RSL-RL train.py structure:
- parse_known_args + hydra_args
- sys.argv stripping for Hydra
- AppLauncher
- cli_args.update_rsl_rl_cfg(...)
- log_root/log_dir naming
- resume_path logic
- dump params/env/agent yaml+pkl
"""

import argparse
import sys
import os
from datetime import datetime

import numpy as np  # optional: keep parity with upstream
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# ---- local imports (match upstream style)
import cli_args  # <-- your uploaded cli_args.py (same folder) :contentReference[oaicite:3]{index=3}

# -----------------------------------------------------------------------------
# CLI args (copy upstream structure) :contentReference[oaicite:4]{index=4}
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (BallVel Estimator).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")

# append RSL-RL cli arguments (experiment_name/run_name/resume/checkpoint/logger/project)
cli_args.add_rsl_rl_args(parser)  # :contentReference[oaicite:5]{index=5}

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra (match upstream) :contentReference[oaicite:6]{index=6}
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- IsaacLab / tasks
from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# IMPORTANT: ensure your task extension is registered
import tennisrobot_rl.tasks  # noqa: F401

# ---- your estimator runner + wrapper
# adjust these imports to your actual file locations
from vecenv_wrapper import TennisRslRlVecEnvWrapper
from modules.on_policy_runner_ballvel import OnPolicyRunnerBallVel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Train with RSL-RL agent (Parkour-style privileged obs + online estimator)."""

    # -------------------------------------------------------------------------
    # 1) Update agent cfg from CLI (match upstream) :contentReference[oaicite:7]{index=7}
    # -------------------------------------------------------------------------
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # override env_cfg/agent_cfg with CLI
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    # set seed/device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # -------------------------------------------------------------------------
    # 2) Distributed training handling (keep same pattern) :contentReference[oaicite:8]{index=8}
    # -------------------------------------------------------------------------
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # -------------------------------------------------------------------------
    # 3) Logging directory (match upstream format) :contentReference[oaicite:9]{index=9}
    # -------------------------------------------------------------------------
    # NOTE: agent_cfg.experiment_name is expected to exist (same as upstream cfg)
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # -------------------------------------------------------------------------
    # 4) Create Isaac environment (match upstream) :contentReference[oaicite:10]{index=10}
    # -------------------------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating wrappers (match upstream) :contentReference[oaicite:11]{index=11}
    if agent_cfg.resume or getattr(agent_cfg.algorithm, "class_name", "") == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = None

    # -------------------------------------------------------------------------
    # 5) Video wrapper (match upstream) :contentReference[oaicite:12]{index=12}
    # -------------------------------------------------------------------------
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # -------------------------------------------------------------------------
    # 6) Wrap for RSL-RL (IMPORTANT: use Tennis wrapper, not upstream wrapper)
    #    - must preserve extras["observations"]["critic"]
    # -------------------------------------------------------------------------
    env = TennisRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Created environment with action space {env.action_space}.")

    # -------------------------------------------------------------------------
    # 7) Create runner (use your estimator runner) + git log + resume
    # -------------------------------------------------------------------------
    print(f"agent_cfg.device: {agent_cfg.device}")
    runner = OnPolicyRunnerBallVel(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # write git state (same API as upstream runner) :contentReference[oaicite:13]{index=13}
    # runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # -------------------------------------------------------------------------
    # 8) Dump params to log_dir/params (match upstream) :contentReference[oaicite:14]{index=14}
    # -------------------------------------------------------------------------
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # -------------------------------------------------------------------------
    # 9) Run training (match upstream call) :contentReference[oaicite:15]{index=15}
    # -------------------------------------------------------------------------
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
