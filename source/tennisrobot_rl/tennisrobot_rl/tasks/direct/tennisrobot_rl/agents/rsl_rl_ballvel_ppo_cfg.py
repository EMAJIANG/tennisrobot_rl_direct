# source/tennisrobot_rl/.../agents/rsl_rl_ballvel_ppo_cfg.py
from __future__ import annotations
from dataclasses import dataclass, field
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg



@configclass
class TennisBallVelEstimatorCfg:
    # which part of policy obs is used as estimator input
    num_prop: int = 14  # 你现在 policy_obs=14，先默认这样

    # slice offsets:
    # policy_obs = [joint_pos | joint_vel | ball_pos | ball_vel]
    # ball_vel is LAST 3 dims, so offset = policy_dim - 3 = 11
    ball_vel_policy_offset: int = 11

    # critic_obs must contain GT ball_linvel; you need to set this offset correctly once you confirm critic dim layout
    # 暂时给一个占位：你跑一次打印 critic_obs_dim 后再改
    ball_vel_critic_offset: int = 0

    # training behavior
    train_with_estimated_ball_vel: bool = False  # 先保持用GT训（你说效果很好）
    estimator_loss_coef: float = 1.0
    lr: float = 3e-4

    # model cfg forwarded into BallVelEstimator(...)
    model: dict = field(default_factory=lambda: dict(
        hidden_dims=(256, 256),
        activation="elu",
    ))


@configclass
class TennisRslRlPpoCfg:
    class_name: str = "PPOWithBallVelEstimator"
    # PPO hyperparams (match rsl_rl PPO kwargs)
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True


@configclass
class TennisRslRlPolicyCfg:
    # rsl_rl.modules.ActorCritic kwargs
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (1024, 512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0


@configclass
class TennisRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    # runner settings
    device: str = "cuda:0"
    experiment_name = "tennisrobot_rl_with_ballvel_estimator"
    empirical_normalization = False

    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 500
    logger: str = "tensorboard"

    algorithm: TennisRslRlPpoCfg = TennisRslRlPpoCfg()
    policy: TennisRslRlPolicyCfg = TennisRslRlPolicyCfg()
    ball_vel_estimator: TennisBallVelEstimatorCfg = TennisBallVelEstimatorCfg()
