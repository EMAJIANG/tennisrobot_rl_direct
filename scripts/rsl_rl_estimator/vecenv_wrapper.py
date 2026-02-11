# scripts/rsl_rl_estimator/vecenv_wrapper.py
from __future__ import annotations

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv


class TennisRslRlVecEnvWrapper(VecEnv):
    """A VecEnv wrapper that matches Parkour's behavior.

    It guarantees:
      - get_observations() returns (policy_obs, {"observations": obs_dict})
      - step() returns obs=policy_obs, and extras["observations"]=obs_dict
    """

    def __init__(self, env: gym.Env, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # actions dim
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # obs dims (policy)
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])

        # privileged obs dims (critic) if exist
        if hasattr(self.unwrapped, "observation_manager") and "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "single_observation_space") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0

        self._modify_action_space()
        self.env.reset()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def __getattr__(self, name):
        # 关键：把没实现的属性/方法透传给底层 env（比如 spec, render_mode, etc.）
        return getattr(self.env, name)
    
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        dones = (terminated | truncated).to(dtype=torch.long)
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict

        # infinite horizon compatibility
        if hasattr(self.unwrapped, "cfg") and hasattr(self.unwrapped.cfg, "is_finite_horizon"):
            if not self.unwrapped.cfg.is_finite_horizon:
                extras["time_outs"] = truncated

        return obs, rew, dones, extras

    def close(self):
        return self.env.close()

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
