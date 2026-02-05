# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim import (
    SimulationCfg,
    SimulationContext,
    RigidBodyMaterialCfg,
    UsdFileCfg,
    RigidBodyPropertiesCfg,
    PhysxCfg
)
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners import materials
import os
import torch

# Base directory path for the USD files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
urdf_dir_path = os.path.join(BASE_DIR, "TR_V4_URDF/urdf") + os.sep  # 确保路径分隔符正确
robot_urdf = "TR_V4_URDF.urdf"
usd_dir_path = os.path.join(BASE_DIR, "USD") + os.sep  # 确保路径分隔符正确
robot_usd = "TR_V4_URDF.usd"  # USD file for the Tennis Robot (TR)
ground_usd = "GroundPlane.usd"  # USD file for the ground plane
court_usd = "Court.usd"  # USD file for the net

@configclass
class TennisrobotRlDirectEnvCfg(DirectRLEnvCfg):
# env
    decimation = 1
    episode_length_s = 5.0
    # - spaces definition
    action_space = 4
    observation_space = 7
    state_space = 0
    # reset
    ball_speed_x_range = (-0.5, 0.5)
    ball_speed_y_range = (-8.0, -6.0)
    ball_speed_z_range = (0.5, 0.8)
    ball_pos_x_range = (0.3, 1.0)
    ball_pos_y_range = (0.0, 0.3)
    ball_pos_z_range = (-0.1, 0.1)

    court_contact_x = (0.0, 10.39)
    court_contact_y = (0.0, 20.0)
    court_contact_z = (0.0, 0.04)

    court_not_contact_x = (0.0, 10.39)
    court_not_contact_y = (0, 9.05)
    court_not_contact_z = (0.0, 0.04)
    
    contact_threshold=0.06

    rew_scale_y = 0.5
    rew_scale_contact =1
    rew_scale_court_success = 5
    rew_scale_court_fail = 2
    rew_scale_ball_outside = 3.5
    rew_scale_ball_pos = 2
    rew_scale_rew_x_align = 1.0

    joint_torque_reward_scale = -2.5e-10
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -1e-4
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=2,
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=25.0, replicate_physics=True
    )
    
    print("Initialized TennisrobotRlDirectEnvCfg with scene configuration.")

    # court
    court = RigidObjectCfg(
        prim_path="/World/envs/env_.*/court",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(5.4, 9.0, 0.0),  # Initial position
            ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(usd_dir_path, court_usd),  # Court USD
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled= True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )   
    print("Initialized TennisrobotRlDirectEnvCfg with court articulation configuration.")
    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/TRbot",
            spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(usd_dir_path, robot_usd),  # TR robot USD
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True
        ),
        activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "X_Pris": 0.0,
                "Z_Pris_H": 0.0,
                "Z_Pris_V": 0.3,
                "Racket_Pev": -torch.pi/3
            },
            pos=(0.0, 0.0, 0.0),
        ),
        actuators={
            "X_Pris": ImplicitActuatorCfg(
                joint_names_expr=["X_Pris"],  # Corrected joint names
                velocity_limit_sim=3.0,
                # effort_limit_sim=3402823466385288598117041834845,
                stiffness=300.0,
                damping=10.0,
            ),
            "Z_Pris_H": ImplicitActuatorCfg(
                joint_names_expr=["Z_Pris_H"],  # Corrected joint names
                velocity_limit_sim=3.0,
                # effort_limit_sim=3402823466385288598117041834845,
                stiffness=30.0,
                damping=20.0,
            ),
            "Z_Pris_V": ImplicitActuatorCfg(
                joint_names_expr=["Z_Pris_V"],  # Corrected joint names
                velocity_limit_sim=3.0,
                # effort_limit_sim=3402823466385288598117041834845,
                stiffness=5000.0,
                damping=100.0,
            ),
            "Racket_Pev": ImplicitActuatorCfg(
                joint_names_expr=["Racket_Pev"],  # Corrected joint names
                velocity_limit_sim=1145.0,  # Adjusted for the racket's movement
                effort_limit_sim=314,
                stiffness=10000000.0,
                damping=0.0,
            ),
        },
    )

    URDF_TR = ArticulationCfg(
        prim_path="/World/envs/env_.*/TRbot",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=False,
            asset_path=os.path.join(urdf_dir_path, robot_urdf),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
                init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "X_Pris": 0.0,
                    "Z_Pris_H": 0.0,
                    "Z_Pris_V": 0.3,
                    "Racket_Pev": 0.0
                },
                pos=(0.0, 0.0, 0.0),
            ),
            actuators={
                "X_Pris": ImplicitActuatorCfg(
                    joint_names_expr=["X_Pris"],  # Corrected joint names
                    velocity_limit_sim=3.0,
                    # effort_limit_sim=3402823466385288598117041834845,
                    stiffness=300.0,
                    damping=10.0,
                ),
                "Z_Pris_H": ImplicitActuatorCfg(
                    joint_names_expr=["Z_Pris_H"],  # Corrected joint names
                    velocity_limit_sim=3.0,
                    # effort_limit_sim=3402823466385288598117041834845,
                    stiffness=30.0,
                    damping=20.0,
                ),
                "Z_Pris_V": ImplicitActuatorCfg(
                    joint_names_expr=["Z_Pris_V"],  # Corrected joint names
                    velocity_limit_sim=3.0,
                    # effort_limit_sim=3402823466385288598117041834845,
                    stiffness=5000.0,
                    damping=100.0,
                ),
                "Racket_Pev": ImplicitActuatorCfg(
                    joint_names_expr=["Racket_Pev"],  # Corrected joint names
                    velocity_limit_sim=1145.0,  # Adjusted for the racket's movement
                    effort_limit_sim=314,
                    stiffness=10000000.0,
                    damping=0.0,
                ),
            },
    )
    print("Initialized TennisrobotRlDirectEnvCfg with robot articulation configuration.")

    # court tennis
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/tennis_ball",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.32511, 8.2, 1.4),
            # lin_vel=(0.0, -5.7, 1.5), 
            lin_vel=(0.0, 0.0, 0.0),  # 初始速度为0
            ang_vel=(0.0, 0.0, 0.0)
            ),
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.SphereCfg(
                    radius=0.033,
                    physics_material=materials.RigidBodyMaterialCfg(
                        restitution=0.6,
                        ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(255/255, 165/255, 0/255),
                        metallic=0.0,
                        roughness=0.5,
                    )
                )
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.057),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors= True,  # Enable contact sensors for the tennis ball
        ),
    )

    contact_sensor = ContactSensorCfg(prim_path="/World/envs/env_.*/TRbot/TR_V4_URDF/hitting_paddle", update_period=0.0, history_length=6, debug_vis=True)

    hitting_point_visualizer_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/paddle_pose_marker",
                markers={
                    "goal": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_y.usd",
                        scale=(0.3, 0.3, 0.3),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    )
                },
            )
    
    print("Initialized TennisrobotRlDirectEnvCfg with ball rigid object configuration.")

    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )

    def __post_init__(self):
        super().__post_init__()
        self.viewer.resolution = (1920, 1080)
        self.viewer.eye = (-1.34, 40, 15)
        self.viewer.lookat = (3.5, 8.2, 1.4)
