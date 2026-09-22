#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
mode=record
if [[ $# -gt 0 && "$1" == "--test" ]]; then
  mode=test
  shift
fi
if [[ "$mode" == "test" && $# -gt 0 ]]; then
  echo "Usage: $0 [--test | output.csv]" >&2
  exit 1
fi
if [[ $# -gt 0 ]]; then
  csv_path="$1"
else
  csv_path="$repo_root/data/d1_real/chirp_$(date +%Y%m%d_%H%M%S).csv"
fi

if [[ ! -f /usr/local/lib/libddt_hardware.so && ! -f /usr/local/lib/libddt_hardware.so.1 ]]; then
  echo "Missing installed D1 driver library: /usr/local/lib/libddt_hardware.so" >&2
  exit 1
fi
if [[ ! -f /usr/local/include/ddt_hardware/ddt_hardware.hpp ]]; then
  echo "Missing installed D1 driver header: /usr/local/include/ddt_hardware/ddt_hardware.hpp" >&2
  exit 1
fi
if [[ "$mode" == "record" && -e "$csv_path" ]]; then
  echo "Output already exists: $csv_path" >&2
  exit 1
fi
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT
g++ -std=c++17 -O2 "$repo_root/scripts/pace/d1_hardware_chirp.cpp" \
  -I/usr/local/include -L/usr/local/lib -Wl,-rpath,/usr/local/lib \
  -lddt_hardware -pthread -o "$build_dir/d1_hardware_chirp"

sudo systemctl stop d1_command.service d1_bringup.service
if pgrep -f 'ros2_control_node' >/dev/null; then
  echo "Stop the active ros2_control_node (for example hw.launch.py) before testing." >&2
  exit 1
fi
if [[ "$mode" == "test" ]]; then
  echo "With the chassis supported: fold four legs, test FL hip, then return to the starting pose."
  timeout --signal=INT --kill-after=2s 45s "$build_dir/d1_hardware_chirp" --test
  exit
fi
mkdir -p "$(dirname "$csv_path")"
echo "With the chassis supported: fold four legs, record, then return to the starting pose: $csv_path"
timeout --signal=INT --kill-after=2s 55s "$build_dir/d1_hardware_chirp" "$csv_path"
if [[ "$(wc -l < "$csv_path")" -ne 8000 ]]; then
  echo "Incomplete recording; expected 8000 CSV rows." >&2
  exit 1
fi
echo "Done: $csv_path"
