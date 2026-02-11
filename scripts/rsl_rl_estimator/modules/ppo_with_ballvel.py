from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO

from .estimator_adapter import BallVelEstimator


def _strip_meta(d: dict) -> dict:
    if d is None:
        return {}
    out = deepcopy(d)
    for k in ["class_name", "_target_", "name"]:
        out.pop(k, None)
    for k in list(out.keys()):
        if isinstance(k, str) and k.startswith("_"):
            out.pop(k, None)
    return out


class PPOWithBallVelEstimator(PPO):
    """PPO + an auxiliary supervised estimator trained online.

    - critic_obs provides GT ball velocity (privileged).
    - estimator predicts ball velocity from policy obs.
    - (optional) inject predicted velocity into policy obs before action.
    """

    def __init__(
        self,
        policy,
        estimator_cfg: dict,
        # obs layout
        num_prop: int,
        ball_vel_policy_offset: int,
        ball_vel_critic_offset: int,
        # behavior
        train_with_estimated_ball_vel: bool = False,
        estimator_loss_coef: float = 1.0,
        estimator_lr: float = 3e-4,
        **ppo_kwargs,
    ):
        # critical: drop hydra/meta keys before calling base PPO
        # ppo_kwargs = _strip_meta(ppo_kwargs)
        if "device" not in ppo_kwargs or ppo_kwargs["device"] is None:
            ppo_kwargs["device"] = next(policy.parameters()).device  # e.g. cuda:0
        super().__init__(policy=policy, **ppo_kwargs)

        self.num_prop = int(num_prop)
        self.ball_vel_policy_offset = int(ball_vel_policy_offset)
        self.ball_vel_critic_offset = int(ball_vel_critic_offset)

        self.train_with_estimated_ball_vel = bool(train_with_estimated_ball_vel)
        self.estimator_loss_coef = float(estimator_loss_coef)

        estimator_cfg = _strip_meta(estimator_cfg)
        self.estimator = BallVelEstimator(num_prop=self.num_prop, **estimator_cfg).to(self.device)
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=float(estimator_lr))
        self.estimator_loss_fn = nn.MSELoss()

        # --- device sync ---
        # PPO uses self.device to move tensors (e.g., infos['time_outs'].to(self.device)).
        # Keep it consistent with the policy / rollout storage device.
        self.device = next(self.policy.parameters()).device  # torch.device
        # ensure estimator lives on the same device
        self.estimator.to(self.device)


    def _ensure_devices(self):
        """Keep policy/estimator on the algorithm device and return it."""
        dev = self.device
        pol_dev = next(self.policy.parameters()).device
        if pol_dev != dev:
            self.device = pol_dev
            dev = pol_dev
        if next(self.estimator.parameters()).device != dev:
            self.estimator.to(dev)
        return dev
    def _inject_ball_vel(self, obs: torch.Tensor, vel_hat: torch.Tensor) -> torch.Tensor:
        """Replace the 3 dims starting at ball_vel_policy_offset with vel_hat."""
        obs2 = obs.clone()
        s = self.ball_vel_policy_offset
        obs2[:, s : s + 3] = vel_hat
        return obs2

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor, **kwargs):
        # Keep all tensors on the algorithm device (must match PPO's self.device usage in process_env_step).
        dev = self._ensure_devices()

        obs = obs.to(dev)
        critic_obs = critic_obs.to(dev)

        vel_hat = self.estimator(obs)

        if self.train_with_estimated_ball_vel:
            obs_actor = self._inject_ball_vel(obs, vel_hat.detach())
        else:
            obs_actor = obs

        actions = self.policy.act(obs_actor, **kwargs)
        self.transition.actions = actions.detach()

        values = self.policy.evaluate(critic_obs)
        self.transition.values = values.detach()

        logp = self.policy.get_actions_log_prob(self.transition.actions)
        self.transition.actions_log_prob = logp.detach()

        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def update(self):  # noqa: C901
        dev = self._ensure_devices()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimator_loss = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            # ------------------------------------------------------------
            # 1) robust unpack: allow extra returned fields (e.g. rnd_state_batch)
            # ------------------------------------------------------------
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                *rest,  # ignore any extra batches
            ) = batch

            # ------------------------------------------------------------
            # 2) move ALL used tensors to same device ONCE
            # ------------------------------------------------------------
            obs_batch = obs_batch.to(dev)
            critic_obs_batch = critic_obs_batch.to(dev)
            actions_batch = actions_batch.to(dev)

            target_values_batch = target_values_batch.to(dev)
            advantages_batch = advantages_batch.to(dev)
            returns_batch = returns_batch.to(dev)

            old_actions_log_prob_batch = old_actions_log_prob_batch.to(dev)
            # old_mu_batch / old_sigma_batch may be unused, but keep consistent if you later use KL, etc.
            old_mu_batch = old_mu_batch.to(dev) if torch.is_tensor(old_mu_batch) else old_mu_batch
            old_sigma_batch = old_sigma_batch.to(dev) if torch.is_tensor(old_sigma_batch) else old_sigma_batch

            masks_batch = masks_batch.to(dev) if torch.is_tensor(masks_batch) else masks_batch

            # recurrent hidden state could be Tensor or tuple/list of Tensors (e.g., LSTM (h,c))
            if isinstance(hid_states_batch, (tuple, list)):
                hid_states_batch = tuple(h.to(dev) if torch.is_tensor(h) else h for h in hid_states_batch)
            elif torch.is_tensor(hid_states_batch):
                hid_states_batch = hid_states_batch.to(dev)

            # ------------------------------------------------------------
            # 3) refresh distribution + compute PPO losses
            # ------------------------------------------------------------
            if self.policy.is_recurrent:
                # rsl_rl recurrent ActorCritic typically supports these args
                self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
                value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
            else:
                self.policy.act(obs_batch)
                value_batch = self.policy.evaluate(critic_obs_batch)

            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            entropy_batch = self.policy.entropy  # usually (B,) or scalar

            # ensure shapes are (B,)
            old_logp = old_actions_log_prob_batch.squeeze(-1)
            adv = advantages_batch.squeeze(-1)

            ratio = torch.exp(actions_log_prob_batch - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
            surrogate_loss = -torch.mean(torch.min(surr1, surr2))

            # value loss expects (B,1) or (B,)
            # keep consistent with rsl_rl: value_batch is usually (B,1)
            if value_batch.ndim == 1:
                value_batch = value_batch.unsqueeze(-1)

            if target_values_batch.ndim == 1:
                target_values_batch = target_values_batch.unsqueeze(-1)
            if returns_batch.ndim == 1:
                returns_batch = returns_batch.unsqueeze(-1)

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(value_losses, value_losses_clipped))
            else:
                value_loss = 0.5 * torch.mean((returns_batch - value_batch).pow(2))

            # entropy loss (PPO convention uses minus entropy to maximize entropy)
            entropy_loss = -torch.mean(entropy_batch) if torch.is_tensor(entropy_batch) else -entropy_batch

            loss = surrogate_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

            # ------------------------------------------------------------
            # 4) estimator supervised loss (GT from critic obs)
            # ------------------------------------------------------------
            gt = critic_obs_batch[:, self.ball_vel_critic_offset : self.ball_vel_critic_offset + 3].detach()
            pred = self.estimator(obs_batch)
            est_loss = self.estimator_loss_fn(pred, gt)

            loss = loss + self.estimator_loss_coef * est_loss

            # ------------------------------------------------------------
            # 5) step (one backward for both optimizers)
            # ------------------------------------------------------------
            self.optimizer.zero_grad(set_to_none=True)
            self.estimator_optimizer.zero_grad(set_to_none=True)

            loss.backward()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            # optional: also clip estimator grads if needed
            # nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.estimator_optimizer.step()

            # ------------------------------------------------------------
            # 6) stats
            # ------------------------------------------------------------
            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            if torch.is_tensor(entropy_batch):
                mean_entropy += float(entropy_batch.mean().item())
            else:
                mean_entropy += float(entropy_batch)
            mean_estimator_loss += float(est_loss.item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= max(1, num_updates)
        mean_surrogate_loss /= max(1, num_updates)
        mean_entropy /= max(1, num_updates)
        mean_estimator_loss /= max(1, num_updates)

        self.storage.clear()

        return {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator_loss": mean_estimator_loss,
        }
