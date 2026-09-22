"""Convert a complete D1 hardware chirp CSV to PACE's [T, 12] .pt format."""

import argparse
from pathlib import Path

import numpy as np
import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv", type=Path)
parser.add_argument("--output", type=Path, default=None, help="Defaults to the CSV path with a .pt suffix.")
args = parser.parse_args()
output = args.output or args.csv.with_suffix(".pt")

rows = np.loadtxt(args.csv, delimiter=",", ndmin=2)
if rows.shape != (8000, 25) or not np.isfinite(rows).all():
    raise ValueError("Expected a complete 20 s recording: 8000 rows x 25 finite columns")
t = rows[:, 0]
if (np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > 0.0075
        or abs((t[-1] - t[0]) - 7999 / 400) > 0.1):
    raise ValueError("Invalid timestamps: nonmonotonic, gap > 7.5 ms, or duration not near 20 s")

grid = t[0] + np.arange(8000) / 400
measured = np.stack([np.interp(grid, t, rows[:, j]) for j in range(1, 13)], axis=1)
desired = np.stack([np.interp(grid, t, rows[:, j]) for j in range(13, 25)], axis=1)
output.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "time": torch.from_numpy((grid - grid[0]).astype(np.float32)),
        "dof_pos": torch.from_numpy(measured.astype(np.float32)),
        "des_dof_pos": torch.from_numpy(desired.astype(np.float32)),
    },
    output,
)
print(f"Saved {len(grid)} x 12 samples to {output}")
