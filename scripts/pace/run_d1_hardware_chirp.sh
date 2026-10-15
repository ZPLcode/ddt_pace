#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
mode=record
if [[ $# -gt 0 && "$1" == "--check" ]]; then
  mode=check
  shift
fi
if [[ "$mode" != "record" && $# -gt 0 ]] || [[ $# -gt 1 ]] || [[ $# -gt 0 && "$1" == --* ]]; then
  echo "Usage: $0 [--check | output.csv]" >&2
  exit 1
fi
if [[ $# -gt 0 ]]; then
  csv_path="$1"
else
  csv_path="$repo_root/data/d1_real/chirp_$(date +%Y%m%d_%H%M%S).csv"
fi

if [[ ! -f /usr/local/lib/libddt_hardware.so && ! -f /usr/local/lib/libddt_hardware.so.1 ]]; then
  echo "Missing D1 driver library: /usr/local/lib/libddt_hardware.so. Run this script on the D1 robot with ddt_hardware installed." >&2
  exit 1
fi
if [[ ! -f /usr/local/include/ddt_hardware/ddt_hardware.hpp ]]; then
  echo "Missing D1 driver header: /usr/local/include/ddt_hardware/ddt_hardware.hpp. Run this script on the D1 robot with ddt_hardware installed." >&2
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

if [[ "$mode" == "check" ]]; then
  echo "D1 collector compiled and linked successfully; no service was stopped or motor command sent."
  exit 0
fi

sudo systemctl stop d1_command.service d1_bringup.service
if pgrep -f 'ros2_control_node' >/dev/null; then
  echo "Stop the active ros2_control_node (for example hw.launch.py) before recording." >&2
  exit 1
fi
mkdir -p "$(dirname "$csv_path")"
echo "With the chassis supported: fold four legs, move to trajectory A center, record, then return: $csv_path"
record_log="${csv_path}.log"
echo "Recording log: $record_log"
timeout --signal=INT --kill-after=5s 80s "$build_dir/d1_hardware_chirp" "$csv_path" 2>&1 | tee "$record_log"
if [[ "$(wc -l < "$csv_path")" -ne 8000 ]]; then
  echo "Incomplete recording; expected 8000 CSV rows." >&2
  exit 1
fi
echo "Done: $csv_path"
