"""
实时可视化SkRL代理的Action和Observation数据
基于提供的skrl play脚本进行修改
"""

import argparse
import sys
import os
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from threading import Thread, Lock
import queue
import math
from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Play and visualize a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--plot-window", type=int, default=100, help="Number of timesteps to display in plots.")
parser.add_argument("--save-plots", action="store_true", help="Save plots to files.")

# 添加AppLauncher参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 如果录制视频，启用摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 清理sys.argv用于Hydra
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入其余模块
import gymnasium as gym
import skrl
from packaging import version

# 检查skrl版本
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class RealTimePlotter:
    """实时数据可视化类 - Action命令 vs 实际关节速度"""
    
    def __init__(self, window_size=100, save_plots=False):
        self.window_size = window_size
        self.save_plots = save_plots
        
        # 数据缓存
        self.action_data = deque(maxlen=window_size)
        self.obs_data = deque(maxlen=window_size)
        self.timesteps = deque(maxlen=window_size)
        
        # 线程安全锁
        self.data_lock = Lock()
        
        # Action限制范围 - 根据ActionsCfg定义
        self.action_limits = {
            "Z_Pris_H": (-2.0, 2.0),      # Y轴对应的关节
            "X_Pris": (-2.0, 2.0),       # X轴对应的关节  
            "Z_Pris_V": (0.0, 1.0),      # Z轴对应的关节
            "Racket_Pev": (-math.pi/3, math.pi/3*4)  # 旋转轴
        }
        
        # 设置matplotlib为交互模式
        plt.ion()
        
        # 创建图形和子图 - 4个关节，每个关节一个子图显示命令vs实际位置
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('SkRL Agent: Action Commands vs Joint Positions (World Coordinates)', fontsize=16)
        
        # 为每个关节创建对比图
        self.action_lines = []
        self.obs_lines = []
        joint_labels = ['Y-axis Joint (Z_Pris_H)', 'X-axis Joint (X_Pris)', 'Z-axis Joint (Z_Pris_V)', 'Rotation Joint (Racket_Pev)']
        joint_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2网格位置
        
        for i in range(4):
            row, col = joint_positions[i]
            ax = self.axes[row, col]
            ax.set_title(f'{joint_labels[i]}: Target vs Actual Position')
            
            # Action目标位置线 (蓝色)
            action_line, = ax.plot([], [], 'b-', linewidth=2, label='Target Position (from Action)')
            self.action_lines.append(action_line)
            
            # 实际关节位置线 (红色)
            obs_line, = ax.plot([], [], 'r-', linewidth=2, label='Actual Joint Position')
            self.obs_lines.append(obs_line)
            
            ax.grid(True)
            ax.set_ylabel('Position (m or rad)')
            ax.legend()
            
        # 设置底部子图的x轴标签
        self.axes[1, 0].set_xlabel('Timesteps')
        self.axes[1, 1].set_xlabel('Timesteps')
        
        plt.tight_layout()
        plt.show(block=False)

    def calculate_racket_center(self, world_move_x, world_move_y, world_move_z, theta):
        """
        计算球拍中心位置（支持批量计算）
        Args:
            world_move_x: tensor, shape (num_envs,)
            world_move_y: tensor, shape (num_envs,)  
            world_move_z: tensor, shape (num_envs,)
            theta: tensor, shape (num_envs,)
        Returns:
            tensor, shape (num_envs, 4) - [x, y, z, theta]
        """
        # 确保输入都是tensor且形状一致
        num_envs = world_move_x.shape[0]
        device = world_move_x.device
        dtype = world_move_x.dtype
    
        
        # 计算击球中心位置 - 批量计算
        hitting_center = torch.stack([
            -world_move_x + 2,     # x方向偏移
            -world_move_y + 2,         # y方向保持
            world_move_z  # z方向偏移
        ], dim=1)  # (num_envs, 3)
        
        # 将theta扩展为列向量并拼接
        theta_expanded = theta.unsqueeze(1)  # (num_envs, 1)
        
        # 返回 (num_envs, 4) 的结果
        return torch.cat([hitting_center, theta_expanded], dim=1)

    def convert_actions_to_world_coords(self, normalized_actions):
        """
        将归一化的actions [-1, 1] 转换为世界坐标下的目标位置
        Args:
            normalized_actions: tensor or array, shape (..., 4) - [y, x, z, r] 顺序
        Returns:
            target positions in world coordinates
        """
        if isinstance(normalized_actions, np.ndarray):
            normalized_actions = torch.from_numpy(normalized_actions)
        
        # 确保形状正确
        if normalized_actions.ndim == 1:
            normalized_actions = normalized_actions.unsqueeze(0)
        
        batch_size = normalized_actions.shape[0]
        device = normalized_actions.device
        dtype = normalized_actions.dtype

        # 将归一化的actions转换为实际的关节限制范围
        # Action顺序: [y, x, z, r] 对应 [Z_Pris_H, X_Pris, Z_Pris_V, Racket_Pev]
        
        # Y轴 (Z_Pris_H): [-1,1] -> [-2.0, 2.0]
        y_target = normalized_actions[:, 0]
        y_target = torch.clamp(y_target,-2.0,2.0)
        # X轴 (X_Pris): [-1,1] -> [-2.0, 2.0] 
        x_target = normalized_actions[:, 1]
        x_target = torch.clamp(x_target,-2.0,2.0)
        # Z轴 (Z_Pris_V): [-1,1] -> [0.0, 1.0]
        z_target = normalized_actions[:, 2]
        z_target = torch.clamp(z_target,0.0,1.0)
        # 旋转轴 (Racket_Pev): [-1,1] -> [-pi/3, 4*pi/3]
        r_min, r_max = -math.pi/3, math.pi/3*4
        r_target = normalized_actions[:, 3]
        r_target = torch.clamp(r_target,r_min,r_max)
        
        # 使用calculate_racket_center函数转换前三个坐标到世界坐标
        world_coords = self.calculate_racket_center(x_target, y_target, z_target, r_target)
        
        return world_coords
        
    def add_data(self, timestep, actions, observations):
        """添加新的数据点"""
        with self.data_lock:
            self.timesteps.append(timestep)
            
            # 处理actions - 将归一化actions转换为世界坐标目标位置
            if isinstance(actions, torch.Tensor):
                actions_cpu = actions.cpu()
            else:
                actions_cpu = torch.from_numpy(np.array(actions).flatten())
            
            if actions_cpu.ndim == 1:
                actions_cpu = actions_cpu.unsqueeze(0)
                
            # 转换为世界坐标目标位置
            world_target_positions = self.convert_actions_to_world_coords(actions_cpu)
            
            # 存储第一个环境的目标位置 [x, y, z, theta]
            target_pos = world_target_positions[0].cpu().numpy()
            self.action_data.append(target_pos)
            
            # 处理observations - 世界坐标下的实际关节位置
            if isinstance(observations, torch.Tensor):
                observations = observations.cpu().numpy()
            if isinstance(observations, np.ndarray) and observations.ndim > 1:
                observations = observations.flatten()
            
            # 假设observations的前4个值是世界坐标下的关节位置 [x, y, z, theta]
            obs_parsed = {
                'joint_positions_world': observations[:4] if len(observations) >= 4 else np.zeros(4)
            }
            self.obs_data.append(obs_parsed)
    
    def update_plots(self):
        """更新所有图表"""
        with self.data_lock:
            if len(self.timesteps) < 2:
                return
            
            timesteps_array = np.array(self.timesteps)
            
            # 更新每个关节的目标位置和实际位置对比图
            if self.action_data and self.obs_data:
                target_positions_array = np.array(self.action_data)  # [x, y, z, theta]
                actual_positions_array = np.array([obs['joint_positions_world'] for obs in self.obs_data])
                
                # 为每个关节更新对应的目标位置和实际位置
                coordinate_labels = ['X', 'Y', 'Z', 'Theta']
                for i in range(min(4, target_positions_array.shape[1], actual_positions_array.shape[1])):
                    # 更新目标位置线 (从action转换得到)
                    self.action_lines[i].set_data(timesteps_array, target_positions_array[:, i])
                    
                    # 更新实际位置线 (从observation获得)
                    self.obs_lines[i].set_data(timesteps_array, actual_positions_array[:, i])
                    
                    # 自动调整坐标轴范围
                    ax = self.action_lines[i].axes
                    ax.relim()
                    ax.autoscale_view()
            
            # 刷新显示
            plt.draw()
            plt.pause(0.01)
    
    def save_current_plots(self, timestep):
        """保存当前图表"""
        if self.save_plots:
            filename = f'skrl_visualization_step_{timestep}.png'
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")


# 配置算法和代理配置入口点
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """运行带可视化的skrl代理"""
    
    # 覆盖配置
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 配置ML框架
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # 设置随机种子
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    task_name = args_cli.task.split(":")[-1]

    # 指定日志目录
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # 获取检查点路径
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # 创建isaac环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 如果需要，转换为单代理实例
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # 获取环境步长
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # 视频录制包装器
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # skrl环境包装器
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # 配置和实例化skrl运行器
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # 初始化可视化器
    plotter = RealTimePlotter(window_size=args_cli.plot_window, save_plots=args_cli.save_plots)
    print("[INFO] Real-time visualization initialized. Close the plot window to stop.")

    # 重置环境
    obs, _ = env.reset()
    timestep = 0

    # 主仿真循环
    try:
        while simulation_app.is_running():
            start_time = time.time()

            # 推理模式下运行
            with torch.inference_mode():
                # 代理步进
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                
                # 多代理确定性动作
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                    # 为可视化提取第一个代理的动作
                    first_agent = list(env.possible_agents)[0]
                    action_for_plot = actions[first_agent]
                    obs_for_plot = obs[first_agent] if isinstance(obs, dict) else obs
                else:
                    # 单代理确定性动作
                    actions = outputs[-1].get("mean_actions", outputs[0])
                    action_for_plot = actions
                    obs_for_plot = obs
                
                # 环境步进
                print(f"Action:{actions}")
                obs, _, _, _, _ = env.step(actions)

            # 添加数据到可视化器
            plotter.add_data(timestep, action_for_plot, obs_for_plot)
            plotter.update_plots()

            timestep += 1

            # 如果录制视频，在录制完成后退出
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # 保存图表
            if args_cli.save_plots and timestep % 50 == 0:  # 每50步保存一次
                plotter.save_current_plots(timestep)

            # 实时评估的时间延迟
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Visualization interrupted by user.")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
    finally:
        # 最终保存图表
        if args_cli.save_plots:
            plotter.save_current_plots(timestep)
        
        # 关闭环境
        env.close()
        plt.close('all')
        print("[INFO] Visualization completed.")


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()