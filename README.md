# PACE for DDT D1

This repository adapts [PACE](https://github.com/leggedrobotics/pace-sim2real)
to the DDT D1 robot. It provides hardware chirp collection, data conversion,
CMA-ES parameter identification and fitted-parameter replay in Isaac Lab.

The D1 task identifies the twelve hip, thigh and calf joints. The four wheel
joints remain in the articulation and are excluded from the position objective.
The repository includes a D1 URDF and meshes under [assets/d1](assets/d1/README.md).

## Installation

Install Isaac Lab and Isaac Sim 5.0 or newer, then install this extension using
the same Python environment:

```bash
git clone https://github.com/ZPLcode/ddt_pace.git
cd ddt_pace
python -m pip install -e source/pace_sim2real
```

The hardware collector requires the D1 driver installed on the robot. Real
recordings and simulation results are stored locally in `data/` and `logs/`.

## D1 workflow

1. Collect a complete 20-second hardware chirp.
2. Convert the CSV recording into PACE tensors.
3. Fit each recording independently.
4. Replay a recording with its fitted parameters and view the fitting error.

```bash
python scripts/pace/d1_csv_to_pt.py data/d1_real/RECORDING.csv
python scripts/pace/fit.py --task Isaac-Pace-D1-v0 \
  --data-file data/d1_real/RECORDING.pt --num_envs 4096 --headless
```

Add `--fixed-zero-bias` for a zero-bias fit.

## D1 additions

The original PACE entry points remain `data_collection.py`, `fit.py` and
`plot_trajectory.py`. D1 hardware collection adds three files:

| File under `scripts/pace/` | Purpose |
|---|---|
| `d1_hardware_chirp.cpp` | Generate the hardware trajectory and collect feedback |
| `run_d1_hardware_chirp.sh` | Compile and launch the hardware collector |
| `d1_csv_to_pt.py` | Convert recordings for fitting |

D1 robot configuration is in the task package; its URDF, meshes and generated
geometry data are under `assets/d1/`. Fixed-parameter replay uses
`fit.py --checkpoint`, and replay plots use `plot_trajectory.py --replay-file`.

## Other PACE tasks

The upstream ANYmal example and RL scripts remain available. PACE documentation:
[pace.filipbjelonic.com](https://pace.filipbjelonic.com).

## License

PACE is licensed under Apache-2.0. Copyright 2025 ETH Zurich, Robotic Systems Lab,
Filip Bjelonic. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
