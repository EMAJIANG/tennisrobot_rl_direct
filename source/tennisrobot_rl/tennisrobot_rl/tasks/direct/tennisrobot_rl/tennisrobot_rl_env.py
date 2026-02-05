# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import datetime
import os
import math
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.assets import Articulation
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners import materials
from isaaclab.utils.math import quat_apply
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from .tennisrobot_rl_direct_env_cfg import TennisrobotRlDirectEnvCfg
from .kf import BatchedKalmanFilter

class TennisrobotRlDirectEnv(DirectRLEnv):
    cfg: TennisrobotRlDirectEnvCfg

    def __init__(
            self, cfg: TennisrobotRlDirectEnvCfg, render_mode: str | None = None, **kwargs
        ):

        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )
        self.total_reward = torch.zeros(self.num_envs, device=self.device)
        self.has_touch_paddle = torch.zeros(self.num_envs, device=self.device).bool()
        self.has_first_bouce = torch.zeros(self.num_envs, device=self.device).bool()
        self.ball_outside_now = torch.zeros(self.num_envs, device=self.device).bool()
        self.has_first_bouce_prev = torch.zeros(
            self.num_envs, device=self.device
        ).bool()
        self.has_touch_own_court = torch.zeros(self.num_envs, device=self.device).bool()
        self.paddle_touch_point = torch.zeros((self.num_envs, 3), device=self.device)
        self.has_touch_own_court_prev = torch.zeros(
            self.num_envs, device=self.device
        ).bool()
        self.reward_vel_prev = torch.zeros(self.num_envs, device=self.device)
        self.rew_court_success = torch.zeros(self.num_envs, device=self.device)
        self.rew_court_fail = torch.zeros(self.num_envs, device=self.device)
        self.rew_ball_to_outside = torch.zeros(self.num_envs, device=self.device)
        self.smallest_dis = torch.ones(self.num_envs, device=self.device) * 100.0
        self._previous_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self.own_bounce_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.two_bounce_own_court = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.x_algin_dist = torch.zeros(self.num_envs, device=self.device)
        self.spatial_ball_to_racket = torch.zeros((self.num_envs, 3), device=self.device)

        # debug
        self.current_obs = []
        self.current_rew = []
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logdir = os.path.join("logs/plots", now)
        os.makedirs(self.logdir, exist_ok=True)
        self.episode_count = 0

        ##
        # 添加这一行来初始化 extras 字典
        self.extras = {}   

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._court = RigidObject(self.cfg.court)
        self._ball  = RigidObject(self.cfg.ball)
        self._paddle_sensor = ContactSensor(self.cfg.contact_sensor)
        self._hitting_point_visualizer = VisualizationMarkers(self.cfg.hitting_point_visualizer_cfg)
        self.env_camera = self.cfg.tiled_camera
        self.scene.sensors["paddle_sensor"] = self._paddle_sensor
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["court"] = self._court
        self.scene.rigid_objects["ball"] = self._ball
        self.hitting_point_visualizer = self._hitting_point_visualizer

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        spawn_ground_plane(
            prim_path="/World/ground", 
            cfg=GroundPlaneCfg(
                physics_material=materials.RigidBodyMaterialCfg(
                    restitution=0.8),
                # color= (223/255, 255/255, 79/255),
                )
        )
        
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
    
    def _apply_action(self):
        self._robot.set_joint_position_target(self.actions * 50)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        stop = (
            (self.rew_court_success != 0)
            | (self.rew_court_fail != 0)
            | (self.rew_ball_to_outside != 0)
        )
        return stop, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        # --- reward 1: ball vel after hitted ---
        reward_vel = (
            self.ball_linvel[:, 1]
            * self.has_touch_paddle.float()
            * self.cfg.rew_scale_y
            * (self.ball_contact == 0).float()
            * torch.logical_not(self.reward_vel_prev)
        )
        # Add bonus reward if contact happened (binary flag).

        # --- reward 2: hit ball by paddle center ---
        reward_contact = self.cfg.rew_scale_contact * self.ball_contact.float()

        # --- reward 3: ball hit target court ---
        self.rew_court_success = (
            self.has_touch_paddle.float()
            * self.has_touch_opponent_court.float()
            * self.cfg.rew_scale_court_success
        )

        # --- reward 4: ball pos ---
        rew_ball_pos = (
            self.rew_court_success * self.ball_pos[:, 1] * self.cfg.rew_scale_ball_pos
        )

        rew_x_align = torch.exp(-(4 * self.x_algin_dist) ** 2) * self.cfg.rew_scale_rew_x_align
        # print(f"x_align_dist: {self.x_algin_dist[0].item():.4f}, rew_x_align: {rew_x_align[0].item():.4f}")
        # --- penalty 1: ball hit own court ---
        self.rew_court_fail = (
            self.has_touch_paddle.float()
            * self.has_touch_own_court.float()
            * self.cfg.rew_scale_court_fail
        )
        ball_y = self.ball_pos[:, 1]  # (N,)
        ball_x = self.ball_pos[:, 0]  # (N,)
        mask_fail = self.rew_court_fail != 0  # (N,) boolean
        self.rew_court_fail[mask_fail] += ball_y[mask_fail] + 0.1

        # --- penalty 2: ball fall to floor with dis penalty---
        ball_to_outside = self.ball_outside_now
        illegal_situation = ball_to_outside | self.two_bounce_own_court
        self.rew_ball_to_outside = illegal_situation.float() * self.cfg.rew_scale_ball_outside * torch.tanh(5 * self.smallest_dis)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self.actions - self._previous_actions), dim=1)
        rew_torque = joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt
        rew_joint_accel  = joint_accel  * self.cfg.joint_accel_reward_scale  * self.step_dt
        rew_action_rate  = action_rate  * self.cfg.action_rate_reward_scale  * self.step_dt

        # --- sum up all rewards ---
        self.total_reward = (
            reward_contact
            + self.rew_court_success
            + rew_x_align
            - self.rew_court_fail
            - self.rew_ball_to_outside
            + reward_vel
            + rew_ball_pos
            + rew_torque
            + rew_joint_accel
            + rew_action_rate
        )
        # print(
        #     "reward | "
        #     f"contact={reward_contact.mean().item():+.3f} | "
        #     f"x_align={rew_x_align.mean().item():+.3f} | "
        #     f"court_succ={self.rew_court_success.mean().item():+.3f} | "
        #     f"court_fail={-self.rew_court_fail.mean().item():+.9f} | "
        #     f"ball_out={-self.rew_ball_to_outside.mean().item():+.9f} | "
        #     f"vel={reward_vel.mean().item():+.3f} | "
        #     f"ball_pos={rew_ball_pos.mean().item():+.3f} | "
        #     f"torque={rew_torque.mean().item():+.3f} | "
        #     f"joint_acc={rew_joint_accel.mean().item():+.3f} | "
        #     f"action_rate={rew_action_rate.mean().item():+.3f} | "
        #     f"TOTAL={self.total_reward.mean().item():+.3f}"
        # )

        still_false = self.reward_vel_prev == 0
        self.reward_vel_prev[still_false] = reward_vel[still_false]

        e0 = 0
        self.current_rew.append(
            {
                # gates / events
                "has_touch_paddle": float(self.has_touch_paddle[e0].item()),
                "has_touch_opponent_court": float(self.has_touch_opponent_court[e0].item()),
                "has_touch_own_court": float(self.has_touch_own_court[e0].item()),
                "ball_outside_now": float(self.ball_outside_now[e0].item()),
                "reward_vel_prev": float(self.reward_vel_prev[e0].item()),

                # kinematics (helpful to debug semantics)
                "ball_x": float(self.ball_pos[e0, 0].item()),
                "ball_y": float(self.ball_pos[e0, 1].item()),
                "ball_z": float(self.ball_pos[e0, 2].item()),
                "ball_vy": float(self.ball_linvel[e0, 1].item()),
                "smallest_dis": float(self.smallest_dis[e0].item()),

                # main reward terms
                "reward_contact": float(reward_contact[e0].item()),
                "rew_x_align": float(rew_x_align[e0].item()),
                "reward_vel": float(reward_vel[e0].item()),
                "rew_court_success": float(self.rew_court_success[e0].item()),
                "rew_ball_pos": float(rew_ball_pos[e0].item()),
                "rew_court_fail": float(self.rew_court_fail[e0].item()),
                "rew_ball_to_outside": float(self.rew_ball_to_outside[e0].item()),

                # regularizers (already scaled by dt & weight)
                "rew_torque": float(rew_torque[e0].item()),
                "rew_joint_accel": float(rew_joint_accel[e0].item()),
                "rew_action_rate": float(rew_action_rate[e0].item()),

                # total
                "total_reward": float(self.total_reward[e0].item()),
            }
        )

        # 将各个奖励分量存储在 extras 字典中，便于后续分析
        self.extras["episode"] = {
            # gates / events (float for logging)
            "has_touch_paddle": self.has_touch_paddle.float().clone(),
            "has_touch_opponent_court": self.has_touch_opponent_court.float().clone(),
            "has_touch_own_court": self.has_touch_own_court.float().clone(),
            "ball_outside_now": self.ball_outside_now.float().clone(),
            "reward_vel_prev": self.reward_vel_prev.float().clone(),

            # kinematics / diagnosticsrew_x_align
            "ball_x": self.ball_pos[:, 0],
            "ball_y": self.ball_pos[:, 1],
            "ball_z": self.ball_pos[:, 2],
            "ball_vy": self.ball_linvel[:, 1],
            "smallest_dis": self.smallest_dis.clone(),

            # reward terms (note: keep sign consistent with how they enter total_reward)
            "reward_contact": reward_contact,
            "reward_vel": reward_vel,
            "rew_court_success": self.rew_court_success,
            "rew_ball_pos": rew_ball_pos,
            "rew_court_fail": -self.rew_court_fail,
            "rew_ball_to_outside": -self.rew_ball_to_outside,
            "rew_x_align": rew_x_align,

            # regularizers (already weighted & dt-scaled; signs are + in total_reward but typically represent penalties)
            "rew_torque": rew_torque,
            "rew_joint_accel": rew_joint_accel,
            "rew_action_rate": rew_action_rate,

            # total
            "total_reward": self.total_reward,
        }
        return self.total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # if 0 in env_ids and len(self.ball_pos_buf) != 0 and len(self.ball_linvel_buf) != 0:
        # # 只写 env=0（与之前讨论一致）。如需全部 env，每步就直接 vstack 全部行即可。
        #     import numpy as np
        #     pos_seq = np.vstack([step_arr[0:1, :] for step_arr in self.ball_pos_buf])       # (T, 3)
        #     vel_seq = np.vstack([step_arr[0:1, :] for step_arr in self.ball_linvel_buf])    # (T, 3)

        #     # 每个 episode 单独一个文件，带表头，便于对齐与可视化
        #     pos_path = os.path.join(self.logdir, f"ball_pos_ep{self.episode_count:06d}.csv")
        #     vel_path = os.path.join(self.logdir, f"ball_linvel_ep{self.episode_count:06d}.csv")
        #     np.savetxt(pos_path, pos_seq, delimiter=",", header="x,y,z", comments="")
        #     np.savetxt(vel_path, vel_seq, delimiter=",", header="vx,vy,vz", comments="")

        #     # 清空缓冲，进入下一集
        #     self.ball_pos_buf.clear()
        #     self.ball_linvel_buf.clear()
        # save plots
        if 0 in env_ids and len(self.current_obs) != 0 and len(self.current_rew) != 0:
            if self.episode_count % 500 == 0:
                self._plot_last_episode(self.current_obs, self.current_rew)
            self.current_obs = []
            self.current_rew = []
            self.episode_count += 1

        # reset the entries in env_ids
        self.total_reward[env_ids] = 0.0
        self.has_touch_paddle[env_ids] = False
        self.has_first_bouce[env_ids] = False
        self.has_first_bouce_prev[env_ids] = False
        self.has_touch_own_court[env_ids] = False
        self.has_touch_own_court_prev[env_ids] = False
        self.reward_vel_prev[env_ids] = 0.0
        self.rew_court_success[env_ids] = 0.0
        self.rew_court_fail[env_ids] = 0.0
        self.rew_ball_to_outside[env_ids] = 0.0
        self.smallest_dis[env_ids] = 100.0
        self.actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.ball_outside_now[env_ids] = False
        self.two_bounce_own_court[env_ids] = 0
        self.own_bounce_count[env_ids] = 0

        # ball (only env_ids)
        ball_state = self._ball.data.default_root_state.clone()[env_ids]
        ball_state[:, :3] += self.scene.env_origins[env_ids]
        # random x-noise and velocity
        pos_noise_x = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_pos_x_range
        )
        pos_noise_y = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_pos_y_range
        )
        pos_noise_z = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_pos_z_range
        )
        ball_state[:, 0:1] += pos_noise_x
        ball_state[:, 1:2] += pos_noise_y
        ball_state[:, 2:3] += pos_noise_z

        v_x = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_speed_x_range
        )
        v_y = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_speed_y_range
        )
        v_z = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.ball_speed_z_range
        )
        lin_vel = torch.cat((v_x, v_y, v_z), dim=1)
        ang_vel = torch.zeros(len(env_ids), 3, device=self.device)
        ball_state[:, 7:] = torch.cat((lin_vel, ang_vel), dim=1)
        self._ball.write_root_pose_to_sim(ball_state[:, :7], env_ids)
        self._ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        # joint_pos = torch.zeros((1, self._robot.num_joints), device=self.device)
        joint_vel = torch.zeros_like(joint_pos)
        # joint_pos[:,0] = 0.0
        # joint_pos[:,2] = 0.3
        # joint_pos[:,1] = ball_state[:, 0] - 1.0 - self.scene.env_origins[env_ids][:,0]
        # joint_pos[:,1] = torch.clamp(joint_pos[:,1], min=-1.8, max=1.8)
        # print(f"env:{env_ids}, reset pos x:{joint_pos[:,1]}, ball pos x:{ball_state[:, 0]-self.scene.env_origins[env_ids][:,0]}")
        joint_pos[:,3] = math.pi/2
        # self._robot.is_initialized
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # self._robot.write_joint_position_limit_to_sim(
        #     joint_ids=[3],
        #     limits=torch.tensor([-torch.pi/3,torch.pi/6],device=self.device),
        #     env_ids= env_ids
        # )
        # self._robot.reset(env_ids)

        # 重置 extras 字典中对应环境的值
        if "episode" in self.extras:
            # 遍历 "episode" 字典中的所有值（这些值就是我们的奖励张量）
            for value_tensor in self.extras["episode"].values():
                # 检查它确实是一个张量并且有足够的维度
                if isinstance(value_tensor, torch.Tensor) and value_tensor.ndim > 0:
                    # 将需要重置的环境在每个奖励分量张量中对应的值设为0
                    value_tensor[env_ids] = 0.0

        self._compute_intermediate_values()
        self.x_algin_dist[env_ids] = torch.abs(self.ball_pos[env_ids, 0] - (joint_pos[:,1] + self.scene.env_origins[env_ids][:,0]))

    def paddle_contact(self) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        contact_sensor = self.scene.sensors["paddle_sensor"]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        # check if any contact force exceeds the threshold
        return torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, :], dim=-1), dim=1)[0] > 0.0, dim=1
        )

    def get_paddle_touch_point(self):
        # --- Compute Paddle Position and Contact ---
        paddle_index = -1
        paddle_index = 4
        paddle_pos = self._robot.data.body_pos_w[:, paddle_index, :]
        paddle_quat = self._robot.data.body_quat_w[:, paddle_index, :]
        # 1) Normalize the quaternion (just in case):
        paddle_quat = paddle_quat / paddle_quat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        # 2) Build the local offset (0, +0.3, 0) and expand to (N,3):
        local_offset = (
            torch.tensor(
                [-2.44 + 1.72915, 0.0, 0.0925],
                device=paddle_pos.device,
                dtype=paddle_pos.dtype,
            )
            .unsqueeze(0)
            .expand_as(paddle_pos)
        )
        rotated_offset: torch.Tensor = quat_apply(paddle_quat, local_offset)
        # 4) Compute your touch point:
        self.paddle_touch_point = paddle_pos + rotated_offset
        self.hitting_point_visualizer.visualize(self.paddle_touch_point)

    def _compute_intermediate_values(self):
        # For ball: extract global pose and then compute local pose using scene.env_origins
        self.ball_global_pos = (
            self._ball.data.root_pos_w
        )  # Global position in simulation
        self.ball_pos = (
            self._ball.data.root_pos_w - self.scene.env_origins
        )  # Local (offset) position
        self.ball_quat = self._ball.data.root_quat_w
        self.ball_vel = self._ball.data.root_vel_w
        self.ball_linvel = self._ball.data.root_lin_vel_w
        self.ball_angvel = self._ball.data.root_ang_vel_w

        self.get_paddle_touch_point()
        self.spatial_ball_to_racket = self.ball_global_pos - self.paddle_touch_point
        self.x_algin_dist = torch.abs(self.ball_global_pos[:, 0] - self.paddle_touch_point[:, 0])
        distance = torch.norm(self.ball_global_pos - self.paddle_touch_point, dim=1).clamp_min(1e-12)
        # print(f"distance: {distance} ball_pos: {self.ball_global_pos} paddle_pos: {self.paddle_touch_point}")
        contact_score = self.paddle_contact().to(torch.float32)
        self.ball_contact = contact_score
        self.ball_contact = self.ball_contact * ~self.has_touch_paddle
        new_hits = contact_score > 0  # Tensor[N] bool
        still_false = ~self.has_touch_paddle  # Tensor[N] bool
        self.has_touch_paddle[still_false] = new_hits[still_false]

        # --- Tennis semantics: in-bounds / half-court / first bounce ---
        bx, by, bz = self.ball_pos[:, 0], self.ball_pos[:, 1], self.ball_pos[:, 2]
        vz = self.ball_linvel[:, 2]

        # 1) court in-bounds (先复用 cfg.court_contact_x/y 作为 court 边界)
        x_min, x_max = self.cfg.court_contact_x
        y_min, y_max = self.cfg.court_contact_y
        in_bounds_xy = (bx >= x_min) & (bx <= x_max) & (by >= y_min) & (by <= y_max)
        self.ball_outside_now = ~in_bounds_xy
        # 2) half-court split (先假设 net 在 y=0；如果你网不在 0，改这里)
        y_net = 9.05
        own_half = by < y_net
        opponent_half = by > y_net

        # 3) bounce detect (用 cfg.court_contact_z 的下界当 bounce_z)
        bounce_z = float(self.cfg.court_contact_z[1])
        bounce_now = (bz <= bounce_z) & (vz < 0.0)
        has_touch_opponent_court = (
            bounce_now & in_bounds_xy & opponent_half
        )
        self.has_touch_opponent_court = has_touch_opponent_court

        has_touch_own_court_just_now = (
            bounce_now & in_bounds_xy & own_half
        )
        self.own_bounce_count += has_touch_own_court_just_now.to(torch.int32)
        self.two_bounce_own_court = self.own_bounce_count >= 2
        self.has_touch_own_court = has_touch_own_court_just_now
        self.has_first_bouce_prev = self.has_first_bouce.clone()
        self.has_touch_own_court_prev |= has_touch_own_court_just_now
        still_false = ~self.has_first_bouce
        self.has_first_bouce[still_false] = self.has_touch_own_court[still_false]
        self.smallest_dis = torch.minimum(
            self.smallest_dis, distance
        )
         # debug info
        self.current_obs.append(
            {
                "has_touch_own_court": self.has_touch_own_court[0].cpu().numpy(),
                "has_touch_own_court_prev": self.has_touch_own_court_prev[0]
                .cpu()
                .numpy(),
                "has_touch_opponent_court": self.has_touch_opponent_court[0]
                .cpu()
                .numpy(),
                "has_first_bouce": self.has_first_bouce[0].cpu().numpy(),
                "has_first_bouce_prev": self.has_first_bouce_prev[0].cpu().numpy(),
                "ball_contact": self.ball_contact[0].cpu().numpy(),
                "has_touch_paddle": self.has_touch_paddle[0].cpu().numpy(),
                "has_first_bouce_prev": self.has_first_bouce_prev[0].cpu().numpy(),
            }
        )

    def _get_observations(self) -> dict:
        self._previous_actions = self.actions.clone()
        self.get_paddle_touch_point()
        critic_ball_racket_pos_error = torch.norm(self.ball_global_pos - self.paddle_touch_point, dim=1).clamp_min(1e-12).unsqueeze(-1)
        policy_obs = torch.cat(
            (self._robot.data.joint_pos,
             self._robot.data.joint_vel, 
             self.ball_pos, 
             self.ball_linvel),
            dim=-1,
        )

        critic_obs = torch.cat(
            (self._robot.data.joint_pos,
             self._robot.data.joint_vel, 
             self.ball_pos, 
             self.ball_linvel,
             critic_ball_racket_pos_error,
             self.spatial_ball_to_racket,
             ),
            dim=-1,
        )
        return {"policy": policy_obs,
                "critic": critic_obs}


    def _plot_last_episode(self, obs_list, rew_list):
        # keys
        obs_keys = list(obs_list[0].keys())
        rew_keys = list(rew_list[0].keys())
        nrows = max(len(obs_keys), len(rew_keys))

        # time axis
        t_obs = np.arange(len(obs_list))
        t_rew = np.arange(len(rew_list))

        # create subplots: nrows x 2
        fig, axes = plt.subplots(
            nrows,
            2,
            figsize=(12, 2.5 * nrows),
            sharex=True,
            tight_layout=True,
        )

        # ensure axes is always 2D array
        if nrows == 1:
            axes = axes[np.newaxis, :]

        # plot observations in left column
        for i, key in enumerate(obs_keys):
            ax = axes[i, 0]
            series = np.array([s[key] for s in obs_list])
            ax.plot(t_obs, series)
            ax.set_ylabel(key)
        # blank any extra rows if obs < nrows
        for i in range(len(obs_keys), nrows):
            axes[i, 0].axis("off")

        # plot rewards in right column
        for i, key in enumerate(rew_keys):
            ax = axes[i, 1]
            series = np.array([r[key] for r in rew_list])
            ax.plot(t_rew, series)
            ax.set_ylabel(key)
        # blank extra reward rows
        for i in range(len(rew_keys), nrows):
            axes[i, 1].axis("off")

        # common x-label on bottom row
        axes[-1, 0].set_xlabel("Timestep")
        axes[-1, 1].set_xlabel("Timestep")

        fig.suptitle("Last Episode: Observations (left) & Rewards (right)")
        filename = os.path.join(self.logdir, f"episode_{self.episode_count:03d}.png")
        fig.savefig(filename)
        plt.close(fig)