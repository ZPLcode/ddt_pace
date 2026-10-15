# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Fit one PACE recording, or replay a saved D1 parameter checkpoint."""

import argparse
import hashlib
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--task", default="Isaac-Pace-Anymal-D-v0")
parser.add_argument(
    "--data-file", type=Path, action="append", help="One PACE recording; relative to the repository root."
)
parser.add_argument("--fixed-zero-bias", action="store_true", help="D1: fix all twelve encoder biases to zero.")
parser.add_argument("--max-iterations", type=int, help="Override the CMA-ES iteration budget.")
parser.add_argument("--checkpoint", type=Path, help="D1: replay these parameters instead of fitting.")
parser.add_argument("--output-dir", type=Path, help="Replay output directory; defaults to the checkpoint directory.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.data_file is not None:
    if len(args_cli.data_file) != 1:
        parser.error("Fit one recording at a time; pass --data-file once")
    args_cli.data_file = args_cli.data_file[0]
if args_cli.task == "Isaac-Pace-D1-v0" and args_cli.data_file is None:
    parser.error("D1 requires --data-file")
if (args_cli.fixed_zero_bias or args_cli.checkpoint) and args_cli.task != "Isaac-Pace-D1-v0":
    parser.error("--fixed-zero-bias and --checkpoint are supported only for Isaac-Pace-D1-v0")
if args_cli.checkpoint and args_cli.fixed_zero_bias:
    parser.error("--fixed-zero-bias applies to fitting; replay uses the saved biases")
if args_cli.max_iterations is not None and args_cli.max_iterations < 1:
    parser.error("--max-iterations must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import pace_sim2real.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg
from pace_sim2real import CMAESOptimizer
from pace_sim2real.utils import project_root


def resolve_path(path: Path) -> Path:
    return (path if path.is_absolute() else project_root() / path).resolve()


def validate_dataset(data: dict, joint_order: list[str], sim_dt: float, path: Path) -> None:
    """Reject recordings that cannot be replayed one sample per simulation step."""
    if not isinstance(data, dict):
        raise ValueError(f"PACE dataset must be a dictionary: {path}")
    missing = {"time", "dof_pos", "des_dof_pos"} - data.keys()
    if missing:
        raise ValueError(f"PACE dataset {path} is missing {sorted(missing)}")
    time = data["time"]
    measured = data["dof_pos"]
    targets = data["des_dof_pos"]
    n = len(joint_order)
    if not all(torch.is_tensor(x) and x.is_floating_point() for x in (time, measured, targets)):
        raise ValueError("PACE time, measured positions and targets must be floating-point tensors")
    if time.ndim != 1 or len(time) < 2 or measured.shape != (len(time), n) or targets.shape != measured.shape:
        raise ValueError(f"Expected time [T] and measured/target positions [T, {n}], T >= 2")
    if not all(torch.isfinite(x).all().item() for x in (time, measured, targets)):
        raise ValueError(f"PACE dataset contains non-finite values: {path}")
    if abs(time[0].item()) > 1.0e-4 or not torch.allclose(
        time[1:] - time[:-1], torch.full_like(time[1:], sim_dt), rtol=0.0, atol=1.0e-5
    ):
        raise ValueError(f"PACE timestamps must start at zero and be sampled at simulation dt={sim_dt} s")
    recorded_order = data.get("joint_order")
    if recorded_order is not None and list(recorded_order) != list(joint_order):
        raise ValueError(f"PACE joint order in {path} differs from task order: {recorded_order}")
    if recorded_order is None:
        print(f"[WARN]: {path} has no joint_order metadata; assuming task order: {joint_order}")


def fit() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.max_iterations is not None:
        cfg.sim2real.cmaes.max_iteration = args_cli.max_iterations
    path = resolve_path(args_cli.data_file or Path("data") / cfg.sim2real.data_dir)
    data = torch.load(path, map_location="cpu", weights_only=False)
    joint_order = cfg.sim2real.joint_order
    validate_dataset(data, joint_order, cfg.sim.dt, path)
    if cfg.sim2real.robot_name.startswith("d1"):
        urdf = Path(cfg.scene.robot.spawn.asset_path).expanduser().resolve()
        digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
        if data.get("source") == "simulation" and data.get("asset_urdf_sha256", digest) != digest:
            raise ValueError("Synthetic recording was generated with a different URDF")
        data.update(sim_urdf_path=str(urdf), sim_urdf_sha256=digest)
        print(f"[INFO]: D1 simulation URDF: {urdf} (SHA256 {digest})")
    data.update(data_file=str(path), data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), sim_dt=cfg.sim.dt)
    print(f"[INFO]: Fitting one recording: {path}")
    env = gym.make(args_cli.task, cfg=cfg)
    opt = None
    try:
        device = env.unwrapped.device
        robot = env.unwrapped.scene["robot"]
        modes = {
            getattr(act.cfg, "delay_mode", "torque")
            for act in robot.actuators.values()
            if hasattr(act, "update_encoder_bias")
        }
        if len(modes) > 1:
            raise ValueError("Identified actuators must share one delay mode")
        data["delay_mode"] = next(iter(modes), "torque")
        joint_ids = torch.tensor([robot.joint_names.index(name) for name in joint_order], device=device)
        measured, targets = data["dof_pos"].to(device), data["des_dof_pos"].to(device)
        opt = CMAESOptimizer(
            bounds=cfg.sim2real.bounds_params.to(device),
            population_size=env.unwrapped.num_envs,
            log_dir=project_root() / "logs" / "pace" / cfg.sim2real.robot_name,
            joint_order=joint_order,
            max_iteration=cfg.sim2real.cmaes.max_iteration,
            data=data,
            device=device,
            epsilon=cfg.sim2real.cmaes.epsilon,
            sigma=cfg.sim2real.cmaes.sigma,
            save_interval=cfg.sim2real.cmaes.save_interval,
            save_optimization_process=cfg.sim2real.cmaes.save_optimization_process,
            fixed_zero_bias=args_cli.fixed_zero_bias,
        )
        actions = torch.zeros(env.action_space.shape, device=device)
        initial = measured[0].unsqueeze(0).expand(env.unwrapped.num_envs, -1)
        with torch.inference_mode():
            while not opt.finished():
                env.reset()
                opt.update_simulator(robot, joint_ids, initial)
                for step in range(len(measured)):
                    if not simulation_app.is_running():
                        raise RuntimeError("Simulation stopped before the recording finished")
                    opt.tell(robot.data.joint_pos[:, joint_ids], measured[step].unsqueeze(0))
                    actions[:, joint_ids] = targets[step].unsqueeze(0)
                    env.step(actions)
                    if (step + 1) % 400 == 0:
                        print(f"[INFO]: Recording: {(step+1)*cfg.sim.dt:.1f} / {len(measured)*cfg.sim.dt:.1f} seconds")
                opt.evolve()
    finally:
        if opt is not None:
            opt.close()
        env.close()


def replay() -> None:
    checkpoint_path = resolve_path(args_cli.checkpoint)
    data_path = resolve_path(args_cli.data_file)
    if not checkpoint_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint or replay data: {checkpoint_path}, {data_path}")

    cfg = parse_env_cfg("Isaac-Pace-D1-v0", device=args_cli.device, num_envs=1)
    joint_order = cfg.sim2real.joint_order
    n = len(joint_order)
    mean = torch.load(checkpoint_path, map_location="cpu", weights_only=False).flatten()
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    fit_data = torch.load(checkpoint_path.parent / "config.pt", map_location="cpu", weights_only=False)
    delay_mode = fit_data.get("delay_mode", "torque")
    if delay_mode not in ("torque", "command"):
        raise ValueError(f"Unknown checkpoint delay mode: {delay_mode}")
    cfg.scene.robot.actuators["legs"].delay_mode = delay_mode
    if abs(float(fit_data.get("sim_dt") or cfg.sim.dt) - cfg.sim.dt) > 1e-9:
        raise ValueError("Checkpoint physics timestep differs from the D1 task")
    if fit_data["joint_order"] != joint_order:
        raise ValueError("Checkpoint joint order differs from the D1 task joint order")
    saved_urdf_sha256 = fit_data.get("sim_urdf_sha256")
    if saved_urdf_sha256 is not None:
        current_urdf_path = Path(cfg.scene.robot.spawn.asset_path).expanduser().resolve()
        if not current_urdf_path.is_file():
            raise FileNotFoundError(f"Selected D1 simulation URDF does not exist: {current_urdf_path}")
        current_urdf_sha256 = hashlib.sha256(current_urdf_path.read_bytes()).hexdigest()
        if current_urdf_sha256 != saved_urdf_sha256:
            raise ValueError(
                f"Checkpoint was fitted with URDF {fit_data.get('sim_urdf_path')} "
                f"(SHA256 {saved_urdf_sha256}), but replay selects {current_urdf_path} "
                f"(SHA256 {current_urdf_sha256})"
            )
    if mean.shape != (4 * n + 1,):
        raise ValueError(f"Expected {4 * n + 1} fit parameters, got {tuple(mean.shape)}")
    real = data["dof_pos"]
    targets = data["des_dof_pos"]
    time = data["time"]
    validate_dataset(data, joint_order, cfg.sim.dt, data_path)
    if not torch.isfinite(mean).all():
        raise ValueError("Checkpoint contains nonfinite values")
    training_records = fit_data.get("datasets") or [fit_data]
    is_fit_data = any(
        torch.equal(real, record["dof_pos"]) and torch.equal(targets, record["des_dof_pos"])
        for record in training_records
    )
    if fit_data.get("training_sha256"):
        is_fit_data = is_fit_data or hashlib.sha256(data_path.read_bytes()).hexdigest() in fit_data["training_sha256"]

    env = gym.make("Isaac-Pace-D1-v0", cfg=cfg)
    try:
        robot = env.unwrapped.scene["robot"]
        device = env.unwrapped.device
        joint_ids = torch.tensor([robot.joint_names.index(name) for name in joint_order], device=device)
        bias = mean[3 * n : 4 * n].to(device).unsqueeze(0)
        env_ids = torch.tensor([0], device=device)

        env.reset()
        armature = mean[:n].to(device).unsqueeze(0)
        viscous = mean[n : 2 * n].to(device).unsqueeze(0)
        friction = mean[2 * n : 3 * n].to(device).unsqueeze(0)
        robot.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_viscous_friction_coefficient_to_sim(viscous, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_dynamic_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)

        legs = robot.actuators["legs"]
        # The fitted order is explicit in the checkpoint, while actuator order can differ.
        fit_index = {name: i for i, name in enumerate(joint_order)}
        drive_indices = legs.joint_indices
        if isinstance(drive_indices, slice):
            drive_indices = range(*drive_indices.indices(len(robot.joint_names)))
        actuator_names = [robot.joint_names[int(i)] for i in drive_indices]
        legs.update_encoder_bias(bias[:, [fit_index[name] for name in actuator_names]])
        legs.update_time_lags(mean[-1:].to(device=device, dtype=torch.int), env_ids)
        legs.reset(env_ids)

        robot.write_joint_position_to_sim(real[0].to(device).unsqueeze(0) + bias, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_velocity_to_sim(torch.zeros((1, n), device=device), joint_ids=joint_ids, env_ids=env_ids)

        simulated = torch.empty_like(real)
        action = torch.zeros(env.action_space.shape, device=device)
        with torch.inference_mode():
            for step in range(len(real)):
                if not simulation_app.is_running():
                    raise RuntimeError("Simulation stopped before replay finished")
                simulated[step] = robot.data.joint_pos[0, joint_ids].cpu() - mean[3 * n : 4 * n]
                action[0, joint_ids] = targets[step].to(device)
                env.step(action)

        error = simulated - real
        rmse = error.square().mean(dim=0).sqrt()
        score = error.square().sum(dim=1).mean()
        label = "Fitting-record replay" if is_fit_data else "Recording replay"
        print(f"{label}: {data_path.name} with {checkpoint_path.parent.name}/{checkpoint_path.name}")
        print(f"Actuator delay mode: {delay_mode}; physics timestep: {cfg.sim.dt} s")
        print(f"PACE score (mean time-wise sum of 12 squared joint errors): {score.item():.9f}")
        print(f"Overall joint RMSE: {error.square().mean().sqrt().item():.6f} rad")
        for name, value in zip(joint_order, rmse.tolist()):
            print(f"  {name}: {value:.6f} rad")

        output_dir = resolve_path(args_cli.output_dir) if args_cli.output_dir else checkpoint_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = "fitting_replay" if is_fit_data else "replay"
        stem = f"{prefix}_{data_path.stem}_{checkpoint_path.stem}"
        result_path = output_dir / f"{stem}.pt"
        torch.save(
            {
                "time": time,
                "joint_order": joint_order,
                "target": targets,
                "real": real,
                "simulated": simulated,
                "rmse": rmse,
                "score": score,
                "delay_mode": delay_mode,
                "sim_dt": cfg.sim.dt,
                "is_fit_data": is_fit_data,
            },
            result_path,
        )
        print(f"Saved replay: {result_path}")

        print(f"Plot replay: python scripts/pace/plot_trajectory.py --replay-file {result_path}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        if args_cli.checkpoint:
            replay()
        else:
            fit()
    finally:
        simulation_app.close()
