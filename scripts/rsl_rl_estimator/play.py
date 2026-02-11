# scripts/rsl_rl_estimator/play.py
# SPDX-License-Identifier: BSD-3-Clause
"""Play a trained checkpoint (Parkour-style)."""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

# local imports (lightweight)
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (Parkour-style).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ✅ add clip_actions for wrapper (optional)
parser.add_argument(
    "--clip_actions",
    type=float,
    default=None,
    help="Action clipping magnitude for TennisRslRlVecEnvWrapper. "
         "If not set, fall back to agent_cfg.clip_actions (if present), else 1.0.",
)

# append RSL-RL cli arguments (includes --device, --checkpoint, etc.)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app (everything heavy must be imported after this)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------
# Heavy imports AFTER app launch
# -------------------------
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import tennisrobot_rl  # noqa: F401, E402
import gymnasium as gym  # noqa: E402

from vecenv_wrapper import TennisRslRlVecEnvWrapper  # noqa: E402
from modules.on_policy_runner_ballvel import OnPolicyRunnerBallVel  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _get_clip_actions(agent_cfg) -> float:
    """Resolve clip_actions from CLI -> agent_cfg -> default."""
    if getattr(args_cli, "clip_actions", None) is not None:
        return float(args_cli.clip_actions)

    # agent_cfg could be OmegaConf / DictConfig / object
    try:
        # common: agent_cfg has attribute or dict key
        if hasattr(agent_cfg, "clip_actions") and agent_cfg.clip_actions is not None:
            return float(agent_cfg.clip_actions)
        if hasattr(agent_cfg, "get") and agent_cfg.get("clip_actions", None) is not None:
            return float(agent_cfg.get("clip_actions"))
    except Exception:
        pass

    return 1.0


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # apply env count if set
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)

    clip_actions = _get_clip_actions(agent_cfg)
    env = TennisRslRlVecEnvWrapper(env, clip_actions=clip_actions)

    runner = OnPolicyRunnerBallVel(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)

    obs, extras = env.get_observations()
    obs = obs.to(args_cli.device)
    critic_obs = extras.get("observations", {}).get("critic", obs).to(args_cli.device)

    # ✅ use no_grad (safer than inference_mode with some pipelines)
    with torch.no_grad():
        while simulation_app.is_running():
            actions = runner.alg.act(obs, critic_obs)
            obs, rew, dones, infos = env.step(actions.to(env.device))
            obs = obs.to(args_cli.device)
            critic_obs = infos.get("observations", {}).get("critic", obs).to(args_cli.device)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
