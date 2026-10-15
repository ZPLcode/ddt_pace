# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import os
from datetime import datetime

import cmaes
import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter


class CMAESOptimizer:
    def __init__(
        self,
        bounds,
        population_size,
        log_dir,
        joint_order,
        max_iteration,
        data,
        device,
        epsilon=None,
        sigma=0.5,
        save_interval=10,
        save_optimization_process=False,
        fixed_zero_bias=False,
    ):

        num_joints = len(joint_order)
        if not isinstance(data, dict) or len(data["dof_pos"]) < 2:
            raise ValueError("PACE needs one recording with two or more samples")
        self.data_length = len(data["dof_pos"])
        if bounds.shape != (4 * num_joints + 1, 2):
            raise ValueError("PACE bounds must contain armature, damping, friction, bias, and delay")
        if fixed_zero_bias and not torch.all(
            (bounds[3 * num_joints : 4 * num_joints, 0] <= 0) & (bounds[3 * num_joints : 4 * num_joints, 1] >= 0)
        ):
            raise ValueError("Zero encoder bias must lie within the configured bias bounds")

        self.joint_order = joint_order
        self.fixed_zero_bias = fixed_zero_bias
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.save_interval = save_interval
        self.device = device
        self.save_optimization_process = save_optimization_process
        self._timer_start = datetime.now()  # timer for logging purposes

        # create log_dir in YY_MM_DD_hh-mm-ss format
        folder_time = datetime.now().strftime("%y_%m_%d_%H-%M-%S")
        # create string with time and date
        log_dir = os.path.join(
            log_dir,
            folder_time + ("_zero_bias" if fixed_zero_bias else ""),
        )
        os.makedirs(log_dir, exist_ok=True)
        self.writer = TensorboardSummaryWriter(log_dir=log_dir)
        torch.save(
            {
                "bounds": bounds.cpu(),
                "joint_order": joint_order,
                "fixed_zero_bias": fixed_zero_bias,
                "optimized_parameter_indices": (
                    (list(range(3 * num_joints)) + [4 * num_joints])
                    if fixed_zero_bias
                    else list(range(bounds.shape[0]))
                ),
                "dof_pos": data["dof_pos"],
                "des_dof_pos": data["des_dof_pos"],
                "time": data["time"],
                "source": data.get("source", "unknown"),
                "data_file": data.get("data_file"),
                "training_sha256": [data["data_sha256"]] if data.get("data_sha256") else [],
                "loss_reduction": "time_mean_joint_sum_squared_error",
                "solver": "CMA-ES",
                "max_iteration": max_iteration,
                "population_size": population_size,
                "epsilon": epsilon,
                "sigma": sigma,
                "seed": 0,
                "sim_urdf_path": data.get("sim_urdf_path"),
                "sim_urdf_sha256": data.get("sim_urdf_sha256"),
                "delay_mode": data.get("delay_mode", "torque"),
                "sim_dt": data.get("sim_dt"),
            },
            log_dir + "/config.pt",
        )

        self.bounds = bounds

        # Remove fixed biases from the CMA search space. Checkpoints still use
        # the normal 4*N+1 physical-parameter layout for existing replay tools.
        self.optimized_indices = torch.tensor(
            list(range(3 * num_joints)) + [4 * num_joints] if fixed_zero_bias else list(range(bounds.shape[0])),
            device=bounds.device,
            dtype=torch.long,
        )

        bounds_normalized = torch.ones_like(bounds[self.optimized_indices])
        bounds_normalized[:, 0] *= -1
        mean_normalized = torch.zeros_like(bounds_normalized[:, 0])

        self.optimizer = cmaes.CMA(
            mean=mean_normalized.cpu().numpy(),
            sigma=sigma,
            bounds=bounds_normalized.cpu().numpy(),
            seed=0,
            population_size=population_size,
        )

        self.scores_counter = 0
        self.iteration_counter = 0

        self.scores = torch.zeros(population_size, device=device)
        self.scores_buffer = torch.empty((max_iteration, population_size), device="cpu")
        # All candidate traces are needed to select the best *after* an episode.
        # Keep them on the host: a 20 s / 400 Hz D1 run with 4096 candidates
        # otherwise consumes about 1.57 GB of GPU memory for this buffer alone.
        self.sim_dof_pos_buffer = torch.empty(
            (self.data_length, population_size, len(joint_order)),
            dtype=torch.float32,
            device="cpu",
        )

        self.params = torch.zeros((population_size, len(self.optimized_indices)), device=device)
        self.sim_params = torch.zeros((population_size, bounds.shape[0]), device=device)
        if save_optimization_process:
            self.sim_params_buffer = torch.empty((max_iteration, population_size, bounds.shape[0]), device="cpu")
        self._finished = False

        self.armature_idx = slice(0, num_joints)
        self.damping_idx = slice(num_joints, 2 * num_joints)
        self.friction_idx = slice(2 * num_joints, 3 * num_joints)
        self.bias_idx = slice(3 * num_joints, 4 * num_joints)
        self.delay_idx = 4 * num_joints

        self._reset_population()
        print("CMA-ES optimizer initialized.")
        print("CMA-ES run directory:", self.writer.log_dir)
        print("Current iteration: ", self.iteration_counter)

    def ask(self):
        return self.optimizer.ask()

    def tell(self, sim_dof_pos, real_dof_pos):
        if self.scores_counter >= self.data_length:
            raise IndexError("Simulation has more steps than the loaded PACE recording")
        self.scores += torch.sum(torch.square(sim_dof_pos - real_dof_pos - self.sim_params[:, self.bias_idx]), dim=1)
        self.sim_dof_pos_buffer[self.scores_counter].copy_(sim_dof_pos.detach().to("cpu"))
        self.scores_counter += 1

    def evolve(self):
        if self.scores_counter != self.data_length:
            raise ValueError("Cannot evolve before the complete recording has been evaluated")
        self.scores /= self.data_length
        if not torch.isfinite(self.scores).all().item():
            raise ValueError("Non-finite PACE score; check simulation stability and input data")
        self.scores_buffer[self.iteration_counter, :].copy_(self.scores.detach().to("cpu"))
        if self.save_optimization_process:
            self.sim_params_buffer[self.iteration_counter].copy_(self.sim_params.detach().to("cpu"))
        solutions = []
        for i in range(self.optimizer.population_size):
            solutions.append((self.params[i].cpu().numpy(), self.scores[i].item()))
        self.optimizer.tell(solutions)

        min_score, min_index = torch.min(self.scores, dim=0)
        max_score = torch.max(self.scores)
        if min_score.item() == 0.0:
            diff_score = 0.0 if max_score.item() == 0.0 else float("inf")
        else:
            diff_score = ((max_score - min_score) / min_score).item()
        self._finished = self.iteration_counter + 1 >= self.max_iteration or (
            self.epsilon is not None and diff_score < self.epsilon
        )
        if (self.save_interval > 0 and self.iteration_counter % self.save_interval == 0) or self._finished:
            mean = self._params_to_sim_params(
                torch.as_tensor(self.optimizer._mean, device=self.device, dtype=self.bounds.dtype)
            )
            self.save_checkpoint(mean, self.iteration_counter, int(min_index.item()), finished=self._finished)
        self._print_iteration()

        self.iteration_counter += 1
        if self._finished:
            print("CMA-ES optimization finished.")
            return
        self._reset_population()
        self.scores.zero_()
        self.scores_counter = 0
        print("CMA-ES optimizer iteration: ", self.iteration_counter)

    def finished(self):
        return self._finished

    def _reset_population(self):
        for i in range(self.optimizer.population_size):
            self.params[i, :] = torch.tensor(self.optimizer.ask(), device=self.device)
        self.sim_params = self._params_to_sim_params(self.params)

    def update_simulator(self, articulation, joint_ids, initial_position):
        env_ids = torch.arange(len(self.sim_params[:, self.armature_idx]))
        articulation.write_joint_armature_to_sim(
            self.sim_params[:, self.armature_idx], joint_ids=joint_ids, env_ids=env_ids
        )
        articulation.data.default_joint_armature[:, joint_ids] = self.sim_params[:, self.armature_idx]
        articulation.write_joint_viscous_friction_coefficient_to_sim(
            self.sim_params[:, self.damping_idx], joint_ids=joint_ids, env_ids=env_ids
        )
        articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = self.sim_params[:, self.damping_idx]
        # If we set static friction lower than dynamic friction, the sim complains. So we need to do this weird order.
        articulation.write_joint_dynamic_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
        articulation.write_joint_friction_coefficient_to_sim(
            self.sim_params[:, self.friction_idx], joint_ids=joint_ids, env_ids=env_ids
        )
        articulation.data.default_joint_friction_coeff[:, joint_ids] = self.sim_params[:, self.friction_idx]
        articulation.write_joint_dynamic_friction_coefficient_to_sim(
            self.sim_params[:, self.friction_idx], joint_ids=joint_ids, env_ids=env_ids
        )
        articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = self.sim_params[:, self.friction_idx]
        articulation.write_joint_position_to_sim(
            initial_position + self.sim_params[:, self.bias_idx], joint_ids=joint_ids
        )
        articulation.write_joint_velocity_to_sim(torch.zeros_like(initial_position), joint_ids=joint_ids)
        for drive_type in articulation.actuators.keys():
            # Mixed-control robots can contain non-PACE actuators (for example,
            # the velocity-controlled D1 wheels).  They are not part of the
            # position-identification parameter vector.
            actuator = articulation.actuators[drive_type]
            if not hasattr(actuator, "update_encoder_bias") or not hasattr(actuator, "update_time_lags"):
                continue

            drive_indices = actuator.joint_indices
            if isinstance(drive_indices, slice):
                all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
                drive_indices = all_idx[drive_indices]
            comparison_matrix = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
            drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
            articulation.actuators[drive_type].update_encoder_bias(
                self.sim_params[:, self.bias_idx][:, drive_joint_idx]
            )
            articulation.actuators[drive_type].update_time_lags(self.sim_params[:, self.delay_idx].to(torch.int))
            articulation.actuators[drive_type].reset(env_ids)

    def _print_iteration(self):
        min_score = torch.min(self.scores)
        max_score = torch.max(self.scores)
        min_index = torch.argmin(self.scores)
        print("Max score: ", max_score.item())
        print("Min score: ", min_score.item(), " at index: ", min_index.item())
        print("Armature: ", self.sim_params[min_index, self.armature_idx].tolist())
        print("Viscous Friction: ", self.sim_params[min_index, self.damping_idx].tolist())
        print("Static/Dynamic Friction: ", self.sim_params[min_index, self.friction_idx].tolist())
        print("Bias: ", self.sim_params[min_index, self.bias_idx].tolist())
        print("Delay: ", self.sim_params[min_index, self.delay_idx].tolist())
        print(f"Joint RMSE: {(min_score.item() / len(self.joint_order)) ** 0.5:.6f} rad")
        print(f"Elapsed time: {(datetime.now() - self._timer_start).total_seconds():.1f} seconds")
        self._timer_start = datetime.now()
        self._log()

    def _params_to_sim_params(self, params):
        if self.fixed_zero_bias:
            selected_bounds = self.bounds[self.optimized_indices]
            selected = selected_bounds[:, 0] + (params + 1.0) / 2.0 * (selected_bounds[:, 1] - selected_bounds[:, 0])
            sim_params = torch.zeros(
                (*params.shape[:-1], self.bounds.shape[0]), device=params.device, dtype=params.dtype
            )
            sim_params[..., self.optimized_indices] = selected
            return sim_params
        sim_params = (params + 1.0) / 2.0  # change range from 0 to 1
        sim_params = self.bounds[:, 0] + sim_params * (
            self.bounds[:, 1] - self.bounds[:, 0]
        )  # range from lower to upper bound
        return sim_params

    def get_best_sim_params(self):
        best_params = torch.as_tensor(self.optimizer._mean, device=self.device, dtype=self.bounds.dtype)
        return self._params_to_sim_params(best_params)

    def _log(self):
        best = torch.argmin(self.scores)
        for tag, group in (
            ("4_Bias", self.bias_idx),
            ("3_Static_Dynamic_Friction", self.friction_idx),
            ("2_Viscous_Friction", self.damping_idx),
            ("1_Armature", self.armature_idx),
        ):
            for index, name in enumerate(self.joint_order):
                values = self.sim_params[:, group][:, index]
                self.writer.add_histogram(f"{tag}/distribution_{name}", values, self.iteration_counter)
                self.writer.add_scalar(f"{tag}/best_{name}", values[best].item(), self.iteration_counter)
        delay = self.sim_params[:, self.delay_idx]
        self.writer.add_histogram("0_Delay/distribution", delay, self.iteration_counter)
        self.writer.add_scalar("0_Delay/best", delay[best].item(), self.iteration_counter)
        minimum, maximum = self.scores.min().item(), self.scores.max().item()
        spread = (maximum - minimum) / minimum if minimum > 0 else (0.0 if maximum == 0 else float("inf"))
        for name, value in (("score", minimum), ("max_score", maximum), ("diff_score", spread)):
            self.writer.add_scalar(f"0_Episode/{name}", value, self.iteration_counter)

    def save_checkpoint(self, mean, iteration, best_index, finished=False):
        suffix = f"{iteration:03}"
        best_traj = self.sim_dof_pos_buffer[:, best_index, :].contiguous()
        best_params = self.sim_params[best_index].detach().cpu().clone()
        torch.save(best_traj, os.path.join(self.writer.log_dir, f"best_trajectory_{suffix}.pt"))
        torch.save(best_params, os.path.join(self.writer.log_dir, f"best_params_{suffix}.pt"))
        # Retain the old latest-trajectory path for external tooling. The
        # iteration-specific files above are the ones safe to plot together.
        torch.save(best_traj, os.path.join(self.writer.log_dir, "best_trajectory.pt"))
        torch.save(best_params, os.path.join(self.writer.log_dir, "best_params.pt"))
        torch.save(mean.detach().cpu(), os.path.join(self.writer.log_dir, f"mean_{suffix}.pt"))
        if finished and self.save_optimization_process:
            torch.save(
                {
                    "params_buffer": self.sim_params_buffer[: iteration + 1].clone(),
                    "scores_buffer": self.scores_buffer[: iteration + 1].clone(),
                },
                os.path.join(self.writer.log_dir, "progress.pt"),
            )

    def close(self):
        self.writer.close()
