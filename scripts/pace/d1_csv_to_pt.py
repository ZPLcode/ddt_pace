"""Convert a complete D1 hardware chirp CSV to PACE's [T, 12] .pt format."""

import argparse
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE_HZ = 400
LEGACY_SHAPE = (8000, 25)
EXTENDED_SHAPE = (8000, 66)
JOINT_ORDER = [
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
MOTOR_IDS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]


def convert(csv_path: Path, output: Path) -> None:
    rows = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    if rows.shape not in (LEGACY_SHAPE, EXTENDED_SHAPE):
        raise ValueError(
            f"Expected a complete legacy {LEGACY_SHAPE[0]} x {LEGACY_SHAPE[1]} "
            f"or extended {EXTENDED_SHAPE[0]} x {EXTENDED_SHAPE[1]} recording; "
            f"got {rows.shape}"
        )
    if not np.isfinite(rows).all():
        raise ValueError("CSV contains nonfinite samples")

    t = rows[:, 0]
    elapsed = t - t[0]
    grid = np.arange(len(t), dtype=np.float64) / SAMPLE_RATE_HZ
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Host sample timestamps must be strictly increasing")
    if np.max(dt) > 0.0075:
        raise ValueError("Host sample timestamps have a gap greater than 7.5 ms")
    max_grid_drift = float(np.max(np.abs(elapsed - grid)))
    if max_grid_drift > 0.01:
        raise ValueError("Host sample timestamps drift over 10 ms from the 400 Hz grid")
    quality = {
        "max_host_interval_s": float(np.max(dt)),
        "p99_host_interval_s": float(np.percentile(dt, 99)),
        "max_grid_drift_s": max_grid_drift,
    }

    def on_grid(columns: np.ndarray) -> torch.Tensor:
        interpolated = np.stack(
            [np.interp(grid, elapsed, columns[:, j]) for j in range(columns.shape[1])],
            axis=1,
        )
        return torch.from_numpy(interpolated.astype(np.float32))

    data = {
        "time": torch.from_numpy(grid.astype(np.float32)),
        "dof_pos": on_grid(rows[:, 1:13]),
        "des_dof_pos": on_grid(rows[:, 13:25]),
        "joint_order": JOINT_ORDER.copy(),
        "motor_ids": MOTOR_IDS.copy(),
        "source": "hardware",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "raw_host_sample_time": torch.from_numpy(elapsed.copy()),
        "quality": quality,
    }
    if rows.shape == EXTENDED_SHAPE:
        # Read integer timestamps again so large counter values keep their exact bits.
        feedback_timestamps = np.loadtxt(csv_path, delimiter=",", usecols=range(25, 37), dtype=np.uint64, ndmin=2)
        if np.any(feedback_timestamps > np.iinfo(np.int64).max):
            raise ValueError("Motor feedback timestamp exceeds int64 range")
        if np.any(feedback_timestamps == 0) or np.any(feedback_timestamps[1:] < feedback_timestamps[:-1]):
            raise ValueError("Motor feedback timestamps are missing or nonmonotonic")
        repeat_fraction = np.mean(feedback_timestamps[1:] == feedback_timestamps[:-1], axis=0)
        quality.update(
            motor_feedback_repeat_fraction_per_joint=repeat_fraction.tolist(),
            motor_feedback_repeat_fraction_min=float(np.min(repeat_fraction)),
            motor_feedback_repeat_fraction_median=float(np.median(repeat_fraction)),
            motor_feedback_repeat_fraction_max=float(np.max(repeat_fraction)),
        )
        data.update(
            raw_motor_feedback_timestamp=torch.from_numpy(feedback_timestamps.astype(np.int64)),
            raw_dof_vel=torch.from_numpy(rows[:, 37:49].astype(np.float32)),
            raw_dof_torque=torch.from_numpy(rows[:, 49:61].astype(np.float32)),
            raw_host_send_begin_time=torch.from_numpy(rows[:, 61].copy()),
            raw_host_send_end_time=torch.from_numpy(rows[:, 62].copy()),
            raw_chirp_phase=torch.from_numpy(rows[:, 63].copy()),
            raw_chirp_frequency_hz=torch.from_numpy(rows[:, 64].copy()),
            raw_excitation_scalar=torch.from_numpy(rows[:, 65].copy()),
        )
        if np.any(rows[:, 62] < rows[:, 61]) or np.any(rows[:, 61] < t) or np.any(np.diff(rows[:, 61]) <= 0):
            raise ValueError("Host send times must follow sample time in each CSV row")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    print(f"Saved {len(grid)} x 12 samples to {output}")
    print(
        "Sampling quality: p99/max interval "
        f"{1000 * quality['p99_host_interval_s']:.3f}/"
        f"{1000 * quality['max_host_interval_s']:.3f} ms, "
        f"max grid drift {1000 * quality['max_grid_drift_s']:.3f} ms"
    )
    if rows.shape == EXTENDED_SHAPE:
        print(
            "Repeated motor feedback per joint: min/median/max "
            f"{100 * quality['motor_feedback_repeat_fraction_min']:.1f}/"
            f"{100 * quality['motor_feedback_repeat_fraction_median']:.1f}/"
            f"{100 * quality['motor_feedback_repeat_fraction_max']:.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to the CSV path with a .pt suffix.")
    args = parser.parse_args()
    convert(args.csv, args.output or args.csv.with_suffix(".pt"))


if __name__ == "__main__":
    main()
