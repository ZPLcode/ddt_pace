// Supported D1 PACE collection using the installed libddt_hardware driver.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "ddt_hardware/ddt_hardware.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
volatile std::sig_atomic_t running = 1;
void stop(int) { running = 0; }

constexpr int ids[12] = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14};
// D1 transform_down.fold_jpos spreads the legs clear of the chassis support.
// Calves stop 0.20 rad before the FSM's -2.7 rad target to leave chirp margin.
constexpr std::array<double, 12> fold = {
  0.3, 1.2, -2.50, -0.3, 1.2, -2.50,
  0.3, 1.2, -2.50, -0.3, 1.2, -2.50};
constexpr double amplitude[12] = {
  0.08, 0.12, -0.12, -0.08, 0.12, -0.12,
  -0.08, -0.12, 0.12, 0.08, -0.12, 0.12};
constexpr double pi = 3.14159265358979323846;
constexpr double max_frequency = 4.0;

bool feedback_ok(const std::vector<ddt_hardware::MotorState> & states, int id)
{
  if (states.size() != 16) return false;
  const auto & s = states[id];
  return s.timestamp != 0 && s.online && s.error == 0 &&
         std::isfinite(s.position) && std::isfinite(s.velocity) &&
         std::isfinite(s.torque);
}

void release(ddt_hardware::DdtHardware & hw)
{
  std::vector<ddt_hardware::MotorCommand> zero(16);
  const auto states = hw.get_motor_states();
  for (int i = 0; i < 16; ++i) {
    zero[i].enable = true;  // The D1 transmits only complete four-slot leg frames.
    if (feedback_ok(states, i)) zero[i].position = states[i].position;
  }
  for (int i = 0; i < 10; ++i) {
    if (!hw.send_motors_command(zero))
      std::cerr << "Failed to send zero-gain command\n";
    std::this_thread::sleep_for(10ms);
  }
}

std::vector<ddt_hardware::MotorCommand> commands(
  const std::vector<ddt_hardware::MotorState> & start)
{
  std::vector<ddt_hardware::MotorCommand> cmd(16);
  for (int i = 0; i < 16; ++i) {
    cmd[i].enable = true;
    if (feedback_ok(start, i)) cmd[i].position = start[i].position;
    // Wheels have zero gains and zero feedforward torque.
  }
  return cmd;
}

struct Monitor {
  std::array<uint64_t, 12> timestamp{};
  std::array<Clock::time_point, 12> fresh{};
  uint64_t tx;
  Clock::time_point next;

  Monitor(ddt_hardware::DdtHardware & hw,
          const std::vector<ddt_hardware::MotorState> & start)
      : tx(hw.get_statistics().tx_count), next(Clock::now())
  {
    for (int j = 0; j < 12; ++j) {
      timestamp[j] = start[ids[j]].timestamp;
      fresh[j] = next;
    }
  }

  bool send(ddt_hardware::DdtHardware & hw,
            std::vector<ddt_hardware::MotorCommand> & cmd,
            const std::vector<ddt_hardware::MotorState> & states,
            bool folding, bool hip_test, bool recording, std::string & reason)
  {
    const auto now = Clock::now();
    if (now > next + 20ms) { reason = "400 Hz loop missed its deadline"; return false; }
    for (int j = 0; j < 12; ++j) {
      const int i = ids[j];
      if (!feedback_ok(states, i)) {
        reason = "motor " + std::to_string(i) + " feedback invalid/offline/error";
        return false;
      }
      if (states[i].timestamp != timestamp[j]) {
        timestamp[j] = states[i].timestamp;
        fresh[j] = now;
      }
      if (now - fresh[j] > 100ms) {
        reason = "motor " + std::to_string(i) + " feedback stale";
        return false;
      }
      const double lower = j % 3 == 0 ? -0.70 : (j % 3 == 1 ? -1.70 : -2.74);
      const double upper = j % 3 == 0 ? 0.70 : (j % 3 == 1 ? 3.30 : -0.90);
      if (states[i].position < lower || states[i].position > upper) {
        reason = "motor " + std::to_string(i) + " reached joint safety range";
        return false;
      }
      const double max_error = folding ? 0.35 : (hip_test && j == 0 ? 0.28 : 0.32);
      const double max_torque = folding ? 25.0 : (hip_test && j == 0 ? 18.0 : 20.0);
      const double max_speed = hip_test && j == 0 ? 0.25 :
                               (folding ? 1.0 : (recording ? 4.0 : 0.6));
      if (std::abs(states[i].velocity) > max_speed) {
        reason = "motor " + std::to_string(i) + " velocity=" +
                 std::to_string(states[i].velocity) + " rad/s exceeds " +
                 std::to_string(max_speed);
        return false;
      }
      if (std::abs(states[i].torque) > max_torque) {
        reason = "motor " + std::to_string(i) + " torque=" +
                 std::to_string(states[i].torque) + " Nm exceeds " +
                 std::to_string(max_torque);
        return false;
      }
      const double error = std::abs(states[i].position - cmd[i].position);
      if (error > max_error) {
        reason = "motor " + std::to_string(i) + " tracking error";
        return false;
      }
      // transform_down fold gains, then the 60/2 collection gains.
      const double base_kp = folding ? (j % 3 == 1 ? 150.0 : 300.0) : 60.0;
      cmd[i].kp = folding ? std::min(base_kp, 20.0 / std::max(error, 0.0001))
                          : base_kp;
      cmd[i].kd = folding ? (j % 3 == 0 ? 4.0f : 8.0f) : 2.0f;
      cmd[i].torque = 0.0f;
    }
    if (!hw.send_motors_command(cmd)) {
      reason = "DdtHardware send_motors_command failed";
      return false;
    }
    const uint64_t sent = hw.get_statistics().tx_count;
    if (sent <= tx) { reason = "D1 driver sent no CAN motor frame"; return false; }
    tx = sent;
    next += 2500us;
    std::this_thread::sleep_until(next);
    return true;
  }
};

double smooth(double u) { return u * u * (3.0 - 2.0 * u); }

bool low_gain_targets(const std::vector<ddt_hardware::MotorState> & start,
                      bool chirp, std::array<double, 12> & baseline)
{
  for (int j = 0; j < 12; ++j) {
    const int i = ids[j];
    if (!feedback_ok(start, i) ||
        std::abs(start[i].velocity) > 0.10 ||
        std::abs(start[i].torque) > 15.0) {
      std::cerr << "Motor " << i << " is not settled for the 60/2 handoff\n";
      return false;
    }
    // Preserve the measured holding effort: 60 * (target - measured_q) = tau.
    baseline[j] = start[i].position + start[i].torque / 60.0;
    const double excursion = chirp ? std::abs(amplitude[j]) : (j == 0 ? 0.08 : 0.0);
    const double lower = j % 3 == 0 ? -0.70 : (j % 3 == 1 ? -1.70 : -2.74);
    const double upper = j % 3 == 0 ? 0.70 : (j % 3 == 1 ? 3.30 : -0.90);
    if (baseline[j] - excursion < lower || baseline[j] + excursion > upper) {
      std::cerr << "Motor " << i << " handoff target is too close to a joint limit\n";
      return false;
    }
  }
  return true;
}

bool move_pose(ddt_hardware::DdtHardware & hw,
               const std::array<double, 12> & target,
               const std::string & name)
{
  const auto start = hw.get_motor_states();
  if (start.size() != 16) {
    std::cerr << name << ": incomplete motor feedback\n";
    return false;
  }
  for (int j = 0; j < 12; ++j) {
    if (!feedback_ok(start, ids[j])) {
      std::cerr << name << ": invalid initial leg feedback\n";
      return false;
    }
  }
  auto cmd = commands(start);
  Monitor monitor(hw, start);
  std::string reason;
  const int hold_steps = name == "Fold" ? 1200 : 400;
  std::cout << name << ": slow 10 s motion, then " << hold_steps / 400
            << " s hold\n";
  if (name == "Fold")
    std::cout << "Check that legs and wheels clear the support; press Ctrl+C if they touch.\n";
  int step = 0;
  for (; running && step < 4000; ++step) {
    const double s = smooth((step + 1) / 4000.0);
    for (int j = 0; j < 12; ++j)
      cmd[ids[j]].position = start[ids[j]].position +
                             s * (target[j] - start[ids[j]].position);
    const auto states = hw.get_motor_states();
    if (!monitor.send(hw, cmd, states, true, false, false, reason)) break;
    if (step % 400 == 0)
      std::cout << name << " " << step / 400 << "s FL hip=" << states[0].position
                << " v=" << states[0].velocity << " tau=" << states[0].torque
                << " thigh=" << states[1].position << " calf=" << states[2].position
                << std::endl;
  }
  if (!running) reason = "interrupted";
  if (reason.empty() && step == 4000) {
    for (int hold_step = 0; running && hold_step < hold_steps; ++hold_step) {
      const auto states = hw.get_motor_states();
      if (!monitor.send(hw, cmd, states, true, false, false, reason)) break;
    }
    if (!running) reason = "interrupted";
  }
  if (reason.empty() && step == 4000) {
    const auto end = hw.get_motor_states();
    for (int j = 0; j < 12; ++j) {
      if (!feedback_ok(end, ids[j]) ||
          std::abs(end[ids[j]].position - target[j]) >
              (name == "Fold" && j % 3 == 0 ? 0.05 : 0.12) ||
          std::abs(end[ids[j]].velocity) > 0.10) {
        reason = "motor " + std::to_string(ids[j]) +
                 " did not settle at target";
        break;
      }
    }
    if (reason.empty())
      std::cout << name << " settled: FL hip q=" << end[0].position
                << " v=" << end[0].velocity << " tau=" << end[0].torque
                << std::endl;
  }
  if (!reason.empty() || step != 4000) {
    std::cerr << name << " stopped at step " << step << ": " << reason << '\n';
    return false;
  }
  std::cout << name << " completed\n";
  return true;
}

bool small_motion(ddt_hardware::DdtHardware & hw)
{
  const auto start = hw.get_motor_states();
  if (!feedback_ok(start, 0)) return false;
  std::array<double, 12> baseline{};
  if (!low_gain_targets(start, false, baseline)) return false;
  const double q0 = start[0].position;
  if (std::abs(q0) > 0.65) {
    std::cerr << "FL hip too close to its joint limit\n";
    return false;
  }
  auto cmd = commands(start);
  for (int j = 0; j < 12; ++j) cmd[ids[j]].position = baseline[j];
  Monitor monitor(hw, start);
  std::string reason;
  double max_progress = 0.0;
  int step = 0;
  std::cout << "FL hip test: measured=" << q0
            << " holding target=" << baseline[0]
            << " effort=" << start[0].torque << " Nm; +0.08 rad and return\n";
  for (; running && step < 3600; ++step) {
    double s = 0;
    if (step >= 400 && step < 2000) s = smooth((step - 400) / 1600.0);
    if (step >= 2000) s = 1 - smooth((step - 2000) / 1600.0);
    cmd[0].position = baseline[0] + 0.08 * s;
    const auto states = hw.get_motor_states();
    if (!feedback_ok(states, 0)) { reason = "FL hip feedback invalid"; break; }
    for (int j = 1; j < 12; ++j) {
      if (!feedback_ok(states, ids[j]) ||
          std::abs(states[ids[j]].position - start[ids[j]].position) > 0.12) {
        reason = "another leg joint drifted during the FL hip test";
        break;
      }
    }
    if (!reason.empty()) break;
    max_progress = std::max(max_progress, states[0].position - q0);
    if (step == 2000 && max_progress < 0.01) {
      reason = "FL hip did not follow the outbound motion";
      break;
    }
    if (!monitor.send(hw, cmd, states, false, true, false, reason)) break;
    if (step % 80 == 0)
      std::cout << "target=" << cmd[0].position << " q=" << states[0].position
                << " v=" << states[0].velocity << " tau=" << states[0].torque << std::endl;
  }
  if (!running) reason = "interrupted";
  const auto end = hw.get_motor_states();
  if (reason.empty() && (max_progress < 0.01 ||
                         !feedback_ok(end, 0) ||
                         std::abs(end[0].position - q0) > 0.02))
    reason = "FL hip did not move out and return";
  if (!reason.empty() || step != 3600) {
    std::cerr << "Test stopped at step " << step << ": " << reason << '\n';
    return false;
  }
  std::cout << "Small-motion test completed; movement=" << max_progress << " rad\n";
  return true;
}

bool record(ddt_hardware::DdtHardware & hw, const std::string & path)
{
  const auto start = hw.get_motor_states();
  std::array<double, 12> baseline{};
  if (!low_gain_targets(start, true, baseline)) return false;
  auto cmd = commands(start);
  Monitor monitor(hw, start);
  std::ofstream csv(path);
  if (!csv) { std::cerr << "Cannot open " << path << '\n'; return false; }
  csv.precision(10);  // time, 12 measured positions, 12 position targets
  std::string reason;
  Clock::time_point chirp_start;
  int step = 0;
  double peak_speed = 0.0;
  double peak_torque = 0.0;
  for (; running && step < 8000; ++step) {
    const double t = step / 400.0;
    const double phase = 2 * pi * (0.1 * t + (max_frequency - 0.1) * t * t / 40.0);
    for (int j = 0; j < 12; ++j)
      cmd[ids[j]].position = baseline[j] + amplitude[j] * std::sin(phase);
    const auto states = hw.get_motor_states();
    const auto now = Clock::now();
    if (step == 0) chirp_start = now;
    if (!monitor.send(hw, cmd, states, false, false, true, reason)) break;
    for (int j = 0; j < 12; ++j) {
      peak_speed = std::max(peak_speed,
                            std::abs(double(states[ids[j]].velocity)));
      peak_torque = std::max(peak_torque,
                             std::abs(double(states[ids[j]].torque)));
    }
    if ((step + 1) % 400 == 0) {
      std::cout << "Recording " << (step + 1) / 400 << "/20 s: peak speed="
                << peak_speed << " rad/s, peak torque="
                << peak_torque << " Nm" << std::endl;
      peak_speed = 0.0;
      peak_torque = 0.0;
    }
    csv << std::chrono::duration<double>(now - chirp_start).count();
    for (int j = 0; j < 12; ++j) csv << ',' << states[ids[j]].position;
    for (int j = 0; j < 12; ++j) csv << ',' << cmd[ids[j]].position;
    csv << '\n';
    if (!csv) { reason = "CSV write failed"; break; }
  }
  if (!running) reason = "interrupted";
  if (!reason.empty() || step != 8000) {
    std::cerr << "Recording stopped at step " << step << ": " << reason << '\n';
    return false;
  }
  std::cout << "Recorded 8000 chirp samples\n";
  return true;
}
}  // namespace

int main(int argc, char ** argv)
{
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  try {
    ddt_hardware::DdtHardware hw("d1");
    if (!hw.initialize() || !hw.is_ready()) {
      std::cerr << "D1 hardware initialization failed\n";
      return 1;
    }
    std::this_thread::sleep_for(100ms);
    const auto first = hw.get_motor_states();
    std::this_thread::sleep_for(20ms);
    const auto initial = hw.get_motor_states();
    bool ready = first.size() == 16 && initial.size() == 16;
    for (int j = 0; j < 12; ++j)
      ready &= feedback_ok(first, ids[j]) && feedback_ok(initial, ids[j]) &&
               first[ids[j]].timestamp != initial[ids[j]].timestamp;
    for (int i : {3, 7, 11, 15}) ready &= feedback_ok(initial, i);
    if (!ready) {
      std::cerr << "Missing valid motor feedback or fresh leg feedback\n";
      hw.shutdown();
      return 1;
    }
    for (int j = 0; j < 12; ++j) {
      const double q = initial[ids[j]].position;
      const bool expected = j % 3 == 0 ? std::abs(q) < 0.65 :
                            j % 3 == 1 ? q > -0.2 && q < 2.0 :
                                         q > -2.7 && q < -0.85;
      if (!expected) {
        std::cerr << "Joint " << ids[j] << " is outside the supported start range\n";
        hw.shutdown();
        return 1;
      }
    }
    std::array<double, 12> original{};
    for (int j = 0; j < 12; ++j) original[j] = initial[ids[j]].position;
    bool ok = move_pose(hw, fold, "Fold");
    if (ok) {
      ok = argc > 1 && std::string(argv[1]) == "--test"
             ? small_motion(hw)
             : record(hw, argc > 1 ? argv[1] : "chirp.csv");
      if (ok && running) ok = move_pose(hw, original, "Return to starting pose");
    }
    release(hw);
    hw.shutdown();
    return ok ? 0 : 1;
  } catch (const std::exception & e) {
    std::cerr << "D1 hardware error: " << e.what() << '\n';
    return 1;
  }
}
