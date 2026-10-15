# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Display PACE parameters and plot fitting or fixed-parameter replay traces."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder_name", help="Run folder under logs/pace/ROBOT.")
parser.add_argument("--mean_name", help="Mean checkpoint; defaults to the last saved iteration.")
parser.add_argument("--robot_name", default="anymal_d_sim")
parser.add_argument("--plot_trajectory", action="store_true")
parser.add_argument("--plot_score", action="store_true")
parser.add_argument("--replay-file", type=Path, help="Plot a fixed-parameter replay saved by fit.py --checkpoint.")
args = parser.parse_args()


def plot_curves(time, measured, target, simulated, names, label):
    for index, name in enumerate(names):
        plt.figure(figsize=(8, 4.5))
        plt.plot(time, simulated[:, index], label=label, color="tab:orange", linewidth=2)
        plt.plot(time, measured[:, index], label="Real", color="tab:green", linestyle="--")
        plt.plot(time, target[:, index], label="Target", color="grey", linestyle="--", alpha=0.5)
        plt.title(name)
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.close()


def main():
    if args.replay_file:
        data = torch.load(args.replay_file, map_location="cpu", weights_only=False)
        overall_rmse = (data["simulated"] - data["real"]).square().mean().sqrt().item()
        print(f"Replay RMSE: {overall_rmse:.6f} rad")
        plot_curves(data["time"], data["real"], data["target"], data["simulated"], data["joint_order"], "Fitted sim")
        return

    root = Path(__file__).resolve().parents[2] / "logs" / "pace" / args.robot_name
    if args.folder_name:
        folder = root / args.folder_name
    else:
        candidates = [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "config.pt").is_file() and any(path.glob("mean_*.pt"))
        ]
        if not candidates:
            raise FileNotFoundError(f"No PACE runs under {root}")
        folder = max(candidates, key=lambda path: path.stat().st_mtime)
    if args.mean_name:
        checkpoint = folder / args.mean_name
    else:
        checkpoints = sorted(folder.glob("mean_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
        if not checkpoints:
            raise FileNotFoundError(f"No mean checkpoints under {folder}")
        checkpoint = checkpoints[-1]
    mean = torch.load(checkpoint, map_location="cpu", weights_only=False).flatten()
    config = torch.load(folder / "config.pt", map_location="cpu", weights_only=False)
    names = config["joint_order"]
    n = len(names)
    if mean.shape != (4 * n + 1,):
        raise ValueError(f"Expected {4*n+1} parameters, got {mean.numel()}")
    print(f"Selected mean checkpoint: {checkpoint}")
    for label, group in (
        ("Armature", mean[:n]),
        ("Viscous friction", mean[n : 2 * n]),
        ("Static/dynamic friction", mean[2 * n : 3 * n]),
        ("Bias", mean[3 * n : 4 * n]),
    ):
        print(f"{label}: {group}")
    print(f"Delay: {mean[-1].item()}")

    iteration = int(checkpoint.stem.split("_")[-1])
    if args.plot_trajectory:
        params_file = folder / f"best_params_{iteration:03}.pt"
        trace_file = folder / f"best_trajectory_{iteration:03}.pt"
        if not params_file.is_file() or not trace_file.is_file():
            raise FileNotFoundError(
                "No paired candidate parameters/trace for this iteration; use fit.py --checkpoint to replay the mean."
            )
        params = torch.load(params_file, map_location="cpu", weights_only=False).flatten()
        trace = torch.load(trace_file, map_location="cpu", weights_only=False)
        if params.shape != mean.shape or trace.shape != config["dof_pos"].shape:
            raise ValueError("Candidate parameters and trace have incompatible shapes")
        if not torch.isfinite(params).all() or not torch.isfinite(trace).all():
            raise ValueError("Candidate parameters or trace contain nonfinite values")
        simulated = trace - params[3 * n : 4 * n]
        plot_curves(config["time"], config["dof_pos"], config["des_dof_pos"], simulated, names, "Best sampled sim")
    if args.plot_score:
        progress = torch.load(folder / "progress.pt", map_location="cpu", weights_only=False)
        plt.figure()
        plt.semilogy(progress["scores_buffer"][: iteration + 1].min(dim=1).values)
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.grid()
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
