# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""PACE system-identification environment for the DDT D1 robot.

The D1 has twelve position-controlled leg joints and four velocity-controlled
wheel joints. The PACE objective identifies only the leg joints.
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.utils import PaceDCMotorCfg, project_root


D1_LEG_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

D1_URDF_PATH = project_root() / "third_party" / "ddt_ros2_control" / "urdfs" / "d1_description" / "urdf" / "robot.urdf"


D1_LEG_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_(hip|thigh|calf)_joint"],
    saturation_effort=90.0,
    effort_limit=90.0,
    velocity_limit=20.0,
    stiffness={".*": 60.0},
    damping={".*": 2.0},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=8,
)

D1_WHEEL_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*_foot_joint"],
    effort_limit_sim=12.0,
    velocity_limit_sim=30.0,
    stiffness=0.0,
    damping=0.5,
    friction=0.0,
)


D1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(D1_URDF_PATH),
        fix_base=True,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=False,
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": D1_LEG_PACE_ACTUATOR_CFG,
        "wheels": D1_WHEEL_ACTUATOR_CFG,
    },
)


@configclass
class D1PaceCfg(PaceCfg):
    """System-identification settings for the D1 leg joints."""

    robot_name: str = "d1_sim"
    data_dir: str = "d1_sim/chirp_data.pt"
    joint_order: list[str] = D1_LEG_JOINT_ORDER
    bounds_params: torch.Tensor = torch.zeros((4 * len(D1_LEG_JOINT_ORDER) + 1, 2))

    def __post_init__(self):
        num_joints = len(self.joint_order)
        self.bounds_params[:num_joints, 0] = 1e-5
        self.bounds_params[:num_joints, 1] = 0.15
        self.bounds_params[num_joints : 2 * num_joints, 1] = 0.5
        self.bounds_params[2 * num_joints : 3 * num_joints, 1] = 1.2
        self.bounds_params[3 * num_joints : 4 * num_joints, 0] = -0.1
        self.bounds_params[3 * num_joints : 4 * num_joints, 1] = 0.1
        self.bounds_params[4 * num_joints, 1] = 8.0


@configclass
class D1PaceSceneCfg(PaceSim2realSceneCfg):
    """PACE scene containing a fixed-base D1 articulation."""

    robot: ArticulationCfg = D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class D1PaceEnvCfg(PaceSim2realEnvCfg):
    """D1 PACE environment running low-level position control at 400 Hz."""

    scene: D1PaceSceneCfg = D1PaceSceneCfg()
    sim2real: PaceCfg = D1PaceCfg()

    def __post_init__(self):
        if not D1_URDF_PATH.is_file():
            raise FileNotFoundError(
                f"D1 URDF not found at {D1_URDF_PATH}. "
                "Run 'git submodule update --init --recursive' from the repository root."
            )
        super().__post_init__()
        self.sim.dt = 0.0025
        self.decimation = 1
