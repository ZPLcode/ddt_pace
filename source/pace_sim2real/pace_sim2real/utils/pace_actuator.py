# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import DCMotor
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    # only for type checking
    from .pace_actuator_cfg import PaceDCMotorCfg


class PaceDCMotor(DCMotor):
    """Pace DC Motor actuator model with encoder bias and action delay.

    The actuator models a DC motor whose controller receives joint positions in the encoder
    frame by subtracting a per-joint encoder bias from the true joint positions. In other words,
    the controller operates on biased (encoder) positions rather than the true joint positions.

    By default the computed torque is delayed. With ``delay_mode="command"``,
    position/velocity/feedforward targets are delayed before evaluating PD with
    the current joint feedback, as in Isaac Lab's DelayedPDActuator.

    The software implementation is inspired by DelayedPDActuator.
    """

    cfg: PaceDCMotorCfg

    def __init__(self, cfg: PaceDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)):
            if len(cfg.encoder_bias) != self.num_joints:
                raise ValueError(
                    f"encoder_bias must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
                )
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)
        if cfg.delay_mode not in ("torque", "command"):
            raise ValueError(f"Unknown PACE delay mode: {cfg.delay_mode}")
        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.update_time_lags(cfg.max_delay)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # reset buffers
        self.torques_delay_buffer.reset(env_ids)
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        if isinstance(delay, torch.Tensor):
            # Isaac Lab 5.1 stores DelayBuffer time lags as torch.int32.
            # Normalize here so callers cannot accidentally pass the default
            # torch.int64 integer dtype and trigger an index_put dtype error.
            delay = delay.to(device=self._device, dtype=torch.int)
        self.torques_delay_buffer.set_time_lag(delay if self.cfg.delay_mode == "torque" else 0, env_ids)
        for buffer in (self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer):
            buffer.set_time_lag(delay if self.cfg.delay_mode == "command" else 0, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if self.cfg.delay_mode == "command":
            control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
            control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
            control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # Compute PD from the encoder-frame position, not the true simulated position.
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(control_action_sim.joint_efforts)
        # ArticulationData.applied_torque reads this value. Keep telemetry in
        # sync with the effort actually sent to the simulator after the delay.
        self.applied_effort = control_action_sim.joint_efforts
        return control_action_sim
