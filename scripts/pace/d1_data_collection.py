# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""Collect a synthetic PACE excitation trajectory for the DDT D1."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect DDT D1 PACE identification data.")
parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 for data collection.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum chirp frequency in Hz.")
parser.add_argument("--max_frequency", type=float, default=4.0, help="Maximum chirp frequency in Hz.")
parser.add_argument("--duration", type=float, default=20.0, help="Chirp duration in seconds.")
parser.add_argument("--plot", action="store_true", help="Plot trajectories after collection.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch import pi

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root


def main():
    if args_cli.num_envs != 1:
        raise ValueError("D1 data collection uses exactly one environment; omit --num_envs or set it to 1.")

    env_cfg = parse_env_cfg("Isaac-Pace-D1-v0", device=args_cli.device, num_envs=1)
    env = gym.make("Isaac-Pace-D1-v0", cfg=env_cfg)
    articulation = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=device)
    env_id = torch.tensor([0], device=device)

    # Synthetic D1 parameters used only to validate the identification pipeline.
    armature = torch.tensor([[0.0535] * 12], device=device)
    damping = torch.tensor([[0.05] * 12], device=device)
    friction = torch.tensor([[0.15] * 12], device=device)
    bias = torch.tensor([[0.01] * 12], device=device)
    time_lag = torch.tensor([3], dtype=torch.int, device=device)

    env.reset()
    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_id)
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=env_id)
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_id)
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_id)
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = friction

    actuator = articulation.actuators["legs"]
    drive_indices = torch.as_tensor(actuator.joint_indices, device=device)
    comparison_matrix = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
    drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
    actuator.update_time_lags(time_lag)
    actuator.update_encoder_bias(bias[:, drive_joint_idx])
    actuator.reset(env_id)

    duration = args_cli.duration
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()
    num_steps = int(duration * sample_rate)
    t = torch.arange(num_steps, device=device) / sample_rate
    phase = 2 * pi * (
        args_cli.min_frequency * t
        + ((args_cli.max_frequency - args_cli.min_frequency) / (2 * duration)) * t**2
    )
    chirp_signal = torch.sin(phase)

    # Match the real collector's folded pose, signed amplitudes, and 400 Hz grid.
    # Hardware adds a per-joint holding offset to its position targets at handoff.
    trajectory_center = torch.tensor(
        [0.3, 1.2, -2.50, -0.3, 1.2, -2.50,
         0.3, 1.2, -2.50, -0.3, 1.2, -2.50],
        device=device,
    )
    trajectory_amplitude = torch.tensor(
        [0.08, 0.12, -0.12, -0.08, 0.12, -0.12,
         -0.08, -0.12, 0.12, 0.08, -0.12, 0.12],
        device=device,
    )
    trajectory = trajectory_center.unsqueeze(0) + chirp_signal.unsqueeze(1) * trajectory_amplitude.unsqueeze(0)

    articulation.write_joint_position_to_sim(trajectory[0].unsqueeze(0) + bias, joint_ids=joint_ids, env_ids=env_id)
    articulation.write_joint_velocity_to_sim(torch.zeros_like(bias), joint_ids=joint_ids, env_ids=env_id)

    dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=device)
    dof_target_pos_buffer = torch.zeros_like(dof_pos_buffer)
    for step in range(num_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            dof_pos_buffer[step] = articulation.data.joint_pos[0, joint_ids] - bias[0]
            actions = torch.zeros(env.action_space.shape, device=device)
            actions[:, joint_ids] = trajectory[step].unsqueeze(0)
            env.step(actions)
            dof_target_pos_buffer[step] = articulation.data.joint_pos_target[0, joint_ids]
        if (step + 1) % 400 == 0:
            print(f"[INFO]: Step {(step + 1) / sample_rate:.1f} seconds")

    env.close()

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time": t.cpu(),
            "dof_pos": dof_pos_buffer.cpu(),
            "des_dof_pos": dof_target_pos_buffer.cpu(),
        },
        data_dir / "chirp_data.pt",
    )

    if args_cli.plot:
        import matplotlib.pyplot as plt

        for index, name in enumerate(joint_order):
            plt.figure()
            plt.plot(t.cpu(), dof_pos_buffer[:, index].cpu(), label=f"{name} pos")
            plt.plot(t.cpu(), dof_target_pos_buffer[:, index].cpu(), "--", label=f"{name} target")
            plt.xlabel("Time [s]")
            plt.ylabel("Joint position [rad]")
            plt.title(name)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
    simulation_app.close()
