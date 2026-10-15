// Supported D1 PACE collection using the installed libddt_hardware driver.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "ddt_hardware/ddt_hardware.hpp"
#include <limits>
#include "../../assets/d1/d1_side_guard_meshes.hpp"

namespace d1_side_guard {
using d1_side_guard_data::Frame;
using d1_side_guard_data::Vertex;
constexpr double kMarginM = 0.003;
constexpr double kRoundoffM = 0.000001;
struct Result {
  double min_signed_y;
  int leg;
  const char * link;
  bool safe() const { return min_signed_y >= kMarginM + kRoundoffM; }
};

inline Frame multiply(const Frame & a, const Frame & b) {
  Frame c{};
  for (int i=0;i<3;++i) {
    c.p[i]=a.p[i];
    for (int k=0;k<3;++k) c.p[i]+=a.r[3*i+k]*b.p[k];
    for (int j=0;j<3;++j)
      for (int k=0;k<3;++k) c.r[3*i+j]+=a.r[3*i+k]*b.r[3*k+j];
  }
  return c;
}

inline Frame motion(const Vertex & axis, double angle) {
  const double c=std::cos(angle), s=std::sin(angle), t=1-c;
  const double x=axis.x,y=axis.y,z=axis.z;
  return {{t*x*x+c,t*x*y-s*z,t*x*z+s*y,
           t*x*y+s*z,t*y*y+c,t*y*z-s*x,
           t*x*z-s*y,t*y*z+s*x,t*z*z+c},{0,0,0}};
}

template<std::size_t N>
inline double min_side_y(const Vertex (&points)[N], const Frame & frame, int side) {
  double best=std::numeric_limits<double>::infinity();
  for (const auto & p:points)
    best=std::min(best,side*(frame.p[1]+frame.r[3]*p.x+frame.r[4]*p.y+frame.r[5]*p.z));
  return best;
}

inline Result check(const std::array<double,12> & q) {
  Result best{std::numeric_limits<double>::infinity(),-1,"none"};
  for (int leg=0;leg<4;++leg) {
    const int side=leg%2==0?1:-1;
    Frame frame{{1,0,0,0,1,0,0,0,1},{0,0,0}};
    for (int part=0;part<4;++part) {
      frame=multiply(frame,d1_side_guard_data::kOrigins[leg][part]);
      if (part<3) {
        const double angle=q[3*leg+part];
        if (!std::isfinite(angle)) return {-std::numeric_limits<double>::infinity(),leg,"nonfinite joint"};
        frame=multiply(frame,motion(d1_side_guard_data::kAxes[leg][part],angle));
      }
      double value;
      if (part==0) value=side>0?min_side_y(d1_side_guard_data::kLeftHip,frame,side):min_side_y(d1_side_guard_data::kRightHip,frame,side);
      else if (part==1) value=side>0?min_side_y(d1_side_guard_data::kLeftThigh,frame,side):min_side_y(d1_side_guard_data::kRightThigh,frame,side);
      else if (part==2) value=side>0?min_side_y(d1_side_guard_data::kLeftCalfBounds,frame,side):min_side_y(d1_side_guard_data::kRightCalfBounds,frame,side);
      else {
        const auto axis=d1_side_guard_data::kAxes[leg][part];
        const double projection=side*(frame.r[3]*axis.x+frame.r[4]*axis.y+frame.r[5]*axis.z);
        const double radial=std::sqrt(std::max(0.0,1-projection*projection));
        const double lo=side>0?d1_side_guard_data::kLeftFootAxialMin:d1_side_guard_data::kRightFootAxialMin;
        const double hi=side>0?d1_side_guard_data::kLeftFootAxialMax:d1_side_guard_data::kRightFootAxialMax;
        const double radius=side>0?d1_side_guard_data::kLeftFootRadius:d1_side_guard_data::kRightFootRadius;
        value=side*frame.p[1]+std::min(projection*lo,projection*hi)-radius*radial;
      }
      static constexpr const char * labels[]={"hip","thigh","calf","foot"};
      if (value<best.min_signed_y) best={value,leg,labels[part]};
    }
  }
  return best;
}
}  // namespace d1_side_guard


namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
volatile std::sig_atomic_t running = 1;
void stop(int) { running = 0; }

constexpr int ids[12] = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14};
constexpr const char * leg_names[4] = {"FL", "FR", "RL", "RR"};
// D1 transform_down.fold_jpos spreads the legs clear of the chassis support.
// Calves stop 0.20 rad before the FSM's -2.7 rad target to leave chirp margin.
constexpr std::array<double, 12> fold = {
  0.3, 1.2, -2.50, -0.3, 1.2, -2.50,
  0.3, 1.2, -2.50, -0.3, 1.2, -2.50};
// Trajectory A is centered on these geometric targets. The gain handoff target
// is temporary and does not change the recorded trajectory center.
constexpr std::array<double, 12> trajectory_center = {
  0.33, -0.20, -1.80, -0.33, -0.20, -1.80,
  0.33, 2.20, -1.80, -0.33, 2.20, -1.80};
constexpr std::array<double, 12> amplitude = {
  0.18, 0.95, -0.70, -0.18, 0.95, -0.70,
  0.18, -0.95, 0.70, -0.18, -0.95, 0.70};
constexpr double pi = 3.14159265358979323846;
constexpr double duration_s = 20.0;
constexpr double min_frequency = 0.1;
constexpr double max_frequency = 2.0;
constexpr int sample_rate = 400;
constexpr int sample_count = 8000;
constexpr double handoff_offset_limit = 0.20;

// Recording stops remain inside the D1 URDF joint limits.
// The calf lower bound is 0.125 rad above its -2.775 rad limit.
double safety_lower(int j)
{
  return j % 3 == 0 ? -0.70 : (j % 3 == 1 ? -1.70 : -2.65);
}

double safety_upper(int j)
{
  return j % 3 == 0 ? 0.70 : (j % 3 == 1 ? 3.30 : -0.90);
}

bool target_in_range(int j, double q)
{
  return std::isfinite(q) && q >= safety_lower(j) && q <= safety_upper(j);
}

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
  std::vector<ddt_hardware::MotorState> states;
  try {
    states = hw.get_motor_states();
  } catch (const std::exception & e) {
    std::cerr << "Cannot read final feedback while releasing gains: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "Cannot read final feedback while releasing gains\n";
  }
  for (int i = 0; i < 16; ++i) {
    zero[i].enable = true;  // The D1 transmits only complete four-slot leg frames.
    if (feedback_ok(states, i)) zero[i].position = states[i].position;
  }
  for (int i = 0; i < 10; ++i) {
    try {
      if (!hw.send_motors_command(zero))
        std::cerr << "Failed to send zero-gain command\n";
    } catch (const std::exception & e) {
      std::cerr << "Zero-gain command failed: " << e.what() << '\n';
    } catch (...) {
      std::cerr << "Zero-gain command failed\n";
    }
    std::this_thread::sleep_for(10ms);
  }
}

struct ShutdownGuard {
  ddt_hardware::DdtHardware & hw;
  bool active = false;

  ~ShutdownGuard() noexcept
  {
    if (!active) return;
    try { release(hw); }
    catch (const std::exception & e) {
      std::cerr << "Cannot release D1 gains: " << e.what() << '\n';
    }
    catch (...) { std::cerr << "Cannot release D1 gains\n"; }
    try { hw.shutdown(); }
    catch (const std::exception & e) {
      std::cerr << "Cannot shut down D1 driver: " << e.what() << '\n';
    }
    catch (...) { std::cerr << "Cannot shut down D1 driver\n"; }
  }
};

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
  std::array<std::uint64_t, 12> timestamp{};
  std::array<Clock::time_point, 12> fresh{};
  std::uint64_t tx;
  Clock::time_point next;
  Clock::time_point send_begin;
  Clock::time_point send_end;

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
            bool folding, bool recording, std::string & reason)
  {
    const auto now = Clock::now();
    if (now > next + 5ms) { reason = "400 Hz loop missed its deadline"; return false; }
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
      if (now - fresh[j] > 10ms) {
        reason = "motor " + std::to_string(i) + " feedback stale";
        return false;
      }
      if (states[i].position < safety_lower(j) ||
          states[i].position > safety_upper(j)) {
        reason = "motor " + std::to_string(i) + " q=" +
                 std::to_string(states[i].position) + " reached joint safety range";
        return false;
      }
      if (!target_in_range(j, cmd[i].position)) {
        reason = "motor " + std::to_string(i) + " target=" +
                 std::to_string(cmd[i].position) + " outside joint safety range";
        return false;
      }
      const double max_error = folding ? 0.35 : 0.32;
      const double max_torque = folding ? 25.0 : 20.0;
      const double max_speed = folding ? 1.0 : (recording ? 4.0 : 0.6);
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
        reason = "motor " + std::to_string(i) + " tracking error=" +
                 std::to_string(error) + " rad exceeds " +
                 std::to_string(max_error);
        return false;
      }
      // transform_down fold gains, then the 60/2 collection gains.
      const double base_kp = folding ? (j % 3 == 1 ? 150.0 : 300.0) : 60.0;
      cmd[i].kp = folding ? std::min(base_kp, 20.0 / std::max(error, 0.0001))
                          : base_kp;
      cmd[i].kd = folding ? (j % 3 == 0 ? 4.0f : 8.0f) : 2.0f;
      cmd[i].torque = 0.0f;
    }
    std::array<double, 12> measured{};
    std::array<double, 12> predicted{};
    std::array<double, 12> midway{};
    std::array<double, 12> commanded{};
    for (int j = 0; j < 12; ++j) {
      const int i = ids[j];
      measured[j] = states[i].position;
      commanded[j] = cmd[i].position;
      // Feedback timestamps are device-local. The host freshness age plus
      // three 400 Hz periods conservatively covers the next command interval
      // and a short transport delay at the measured angular velocity.
      const double horizon = std::chrono::duration<double>(now - fresh[j]).count() + 0.0075;
      predicted[j] = measured[j] + states[i].velocity * horizon;
      midway[j] = 0.5 * (measured[j] + predicted[j]);
      if (!target_in_range(j, predicted[j])) {
        reason = "motor " + std::to_string(i) +
                 " predicted to leave joint safety range";
        return false;
      }
    }
    const auto measured_side = d1_side_guard::check(measured);
    const auto midway_side = d1_side_guard::check(midway);
    const auto predicted_side = d1_side_guard::check(predicted);
    const auto commanded_side = d1_side_guard::check(commanded);
    struct GuardCheck { const char * phase; d1_side_guard::Result result; };
    const GuardCheck guard_checks[] = {
      {"measured", measured_side}, {"midway", midway_side},
      {"predicted", predicted_side}, {"commanded", commanded_side}};
    for (const auto & check : guard_checks) {
      if (!check.result.safe()) {
        reason = std::string(check.phase) + " modeled " +
                 leg_names[check.result.leg] + " " + check.result.link +
                 " approaches chassis midline: own-side margin=" +
                 std::to_string(check.result.min_signed_y * 1000.0) + " mm";
        return false;
      }
    }
    send_begin = Clock::now();
    if (!hw.send_motors_command(cmd)) {
      reason = "DdtHardware send_motors_command failed";
      return false;
    }
    send_end = Clock::now();
    const std::uint64_t sent = hw.get_statistics().tx_count;
    if (sent <= tx) { reason = "D1 driver sent no CAN motor frame"; return false; }
    tx = sent;
    next += 2500us;
    std::this_thread::sleep_until(next);
    return true;
  }
};

double smooth(double u) { return u * u * (3.0 - 2.0 * u); }

bool low_gain_targets(const std::vector<ddt_hardware::MotorState> & start,
                      std::array<double, 12> & baseline)
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
    // For the chirp this target is temporary during the gain switch; the
    // recorded trajectory later uses trajectory_center exactly.
    baseline[j] = start[i].position + start[i].torque / 60.0;
    const double excursion = std::abs(amplitude[j]);
    if (std::abs(start[i].position - trajectory_center[j]) > 0.15 ||
        std::abs(baseline[j] - trajectory_center[j]) > handoff_offset_limit) {
      std::cerr << "Motor " << i << " cannot hold trajectory A center at 60/2: q="
                << start[i].position << " tau=" << start[i].torque
                << " handoff target=" << baseline[j]
                << " planned center=" << trajectory_center[j] << '\n';
      return false;
    }
    const double center = trajectory_center[j];
    // A 0.02 rad buffer beyond the recording stops for the full nominal
    // A excursion. The chirp's high-frequency taper only reduces it.
    if (!target_in_range(j, baseline[j]) ||
        center - excursion < safety_lower(j) + 0.02 ||
        center + excursion > safety_upper(j) - 0.02) {
      std::cerr << "Motor " << i << " handoff target is too close to a joint limit\n";
      return false;
    }
  }
  return true;
}

bool low_gain_handoff(ddt_hardware::DdtHardware & hw,
                      const std::vector<ddt_hardware::MotorState> & start,
                      const std::array<double, 12> & baseline)
{
  auto cmd = commands(start);
  Monitor monitor(hw, start);
  std::string reason;
  int step = 0;
  std::cout << "Switching to 60/2 while holding measured effort, then blending "
               "to exact trajectory A center over 2 s and holding 1 s\n";
  for (; running && step < 1200; ++step) {
    const double blend = smooth(std::min((step + 1) / 800.0, 1.0));
    for (int j = 0; j < 12; ++j)
      cmd[ids[j]].position = baseline[j] +
                             blend * (trajectory_center[j] - baseline[j]);
    const auto states = hw.get_motor_states();
    for (int j = 0; j < 12; ++j) {
      if (!feedback_ok(states, ids[j]) ||
          std::abs(states[ids[j]].position - trajectory_center[j]) > 0.25) {
        reason = "motor " + std::to_string(ids[j]) +
                 " left trajectory A center corridor during 60/2 handoff";
        break;
      }
    }
    if (!reason.empty()) break;
    if (!monitor.send(hw, cmd, states, false, false, reason)) break;
  }
  if (!running) reason = "interrupted";
  if (reason.empty() && step == 1200) {
    const auto states = hw.get_motor_states();
    for (int j = 0; j < 12; ++j) {
      if (!feedback_ok(states, ids[j]) ||
          std::abs(states[ids[j]].position - trajectory_center[j]) > 0.15 ||
          std::abs(states[ids[j]].velocity) > 0.10) {
        reason = "motor " + std::to_string(ids[j]) +
                 " did not settle at trajectory A center under 60/2";
        break;
      }
    }
  }
  if (!reason.empty() || step != 1200) {
    std::cerr << "Low-gain handoff stopped at step " << step << ": " << reason << '\n';
    return false;
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
    if (!monitor.send(hw, cmd, states, true, false, reason)) break;
    if (step % 400 == 0)
      std::cout << name << " " << step / 400 << "s FL hip=" << states[0].position
                << " v=" << states[0].velocity << " tau=" << states[0].torque
                << " thigh=" << states[1].position << " calf=" << states[2].position
                << " FR calf=" << states[6].position
                << " FR calf target=" << cmd[6].position
                << " FR calf tau=" << states[6].torque
                << std::endl;
  }
  if (!running) reason = "interrupted";
  if (reason.empty() && step == 4000) {
    for (int hold_step = 0; running && hold_step < hold_steps; ++hold_step) {
      const auto states = hw.get_motor_states();
      if (!monitor.send(hw, cmd, states, true, false, reason)) break;
    }
    if (!running) reason = "interrupted";
  }
  if (reason.empty() && step == 4000) {
    const auto end = hw.get_motor_states();
    for (int j = 0; j < 12; ++j) {
      const int id = ids[j];
      if (!feedback_ok(end, id)) {
        reason = "motor " + std::to_string(id) +
                 " feedback invalid at settle check (state count=" +
                 std::to_string(end.size()) + ")";
        break;
      }
      const double error = end[id].position - target[j];
      const double position_tolerance = name == "Fold" && j % 3 == 0 ? 0.05 : 0.12;
      if (std::abs(error) > position_tolerance ||
          std::abs(end[id].velocity) > 0.10) {
        reason = "motor " + std::to_string(id) + " did not settle: q=" +
                 std::to_string(end[id].position) + " target=" +
                 std::to_string(target[j]) + " error=" +
                 std::to_string(error) + " v=" +
                 std::to_string(end[id].velocity) + " tau=" +
                 std::to_string(end[id].torque);
        break;
      }
    }
    if (reason.empty())
      std::cout << name << " settled: FL hip q=" << end[0].position
                << " v=" << end[0].velocity << " tau=" << end[0].torque
                << " FL calf q=" << end[ids[2]].position
                << " v=" << end[ids[2]].velocity
                << " FR calf q=" << end[ids[5]].position
                << " v=" << end[ids[5]].velocity
                << std::endl;
  }
  if (!reason.empty() || step != 4000) {
    std::cerr << name << " stopped at step " << step << ": " << reason << '\n';
    return false;
  }
  std::cout << name << " completed\n";
  return true;
}

bool prepare_trajectory_a(ddt_hardware::DdtHardware & hw)
{
  const auto start = hw.get_motor_states();
  std::array<double, 12> baseline{};
  if (!low_gain_targets(start, baseline)) return false;
  return low_gain_handoff(hw, start, baseline);
}

bool record(ddt_hardware::DdtHardware & hw, const std::string & path)
{
  if (!prepare_trajectory_a(hw)) return false;
  const auto settled = hw.get_motor_states();
  for (int j = 0; j < 12; ++j) {
    if (!feedback_ok(settled, ids[j])) {
      std::cerr << "Invalid feedback just before recording\n";
      return false;
    }
  }
  auto cmd = commands(settled);
  std::ofstream csv(path);
  if (!csv) { std::cerr << "Cannot open " << path << '\n'; return false; }
  Monitor monitor(hw, settled);
  // 0..24 retain the legacy time, 12 measured positions, 12 target positions.
  // 25..65 add 12 raw feedback timestamps, 12 velocities, 12 torques,
  // host send begin/end times, phase, frequency, and actual excitation scalar.
  csv.precision(17);
  std::string reason;
  Clock::time_point chirp_start;
  int step = 0;
  double peak_speed = 0.0;
  double peak_torque = 0.0;
  for (; running && step < sample_count; ++step) {
    const double t = static_cast<double>(step) / sample_rate;
    const double frequency = min_frequency +
                             (max_frequency - min_frequency) *
                             (t / duration_s) * (t / duration_s);
    const double phase = 2 * pi *
        (min_frequency * t + (max_frequency - min_frequency) *
         t * t * t / (3 * duration_s * duration_s));
    const double envelope = 0.1 + 0.9 /
        std::sqrt(1.0 + std::pow(frequency / 0.7, 4));
    const double ramp_in = smooth(std::clamp(t, 0.0, 1.0));
    const double ramp_out = smooth(std::clamp(duration_s - t, 0.0, 1.0));
    // D1's synthetic A exceeded the real logger's velocity/tracking stops
    // late in the sweep. Preserve the low-frequency wide pose coverage and
    // taper high-frequency amplitude; the monitor still stops on real feedback.
    const double high_frequency_taper = 1.0 - 0.5 *
        smooth(std::clamp((frequency - 0.8) / 0.4, 0.0, 1.0));
    const double excitation = ramp_in * ramp_out * envelope *
                              high_frequency_taper * std::sin(phase);
    for (int j = 0; j < 12; ++j)
      cmd[ids[j]].position = trajectory_center[j] + amplitude[j] * excitation;
    const auto states = hw.get_motor_states();
    const auto now = Clock::now();
    if (step == 0) chirp_start = now;
    if (!monitor.send(hw, cmd, states, false, true, reason)) break;
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
    for (int j = 0; j < 12; ++j) csv << ',' << states[ids[j]].timestamp;
    for (int j = 0; j < 12; ++j) csv << ',' << states[ids[j]].velocity;
    for (int j = 0; j < 12; ++j) csv << ',' << states[ids[j]].torque;
    csv << ',' << std::chrono::duration<double>(monitor.send_begin - chirp_start).count()
        << ',' << std::chrono::duration<double>(monitor.send_end - chirp_start).count()
        << ',' << phase << ',' << frequency << ',' << excitation;
    csv << '\n';
    if (!csv) { reason = "CSV write failed"; break; }
  }
  if (!running) reason = "interrupted";
  csv.flush();
  if (!csv && reason.empty()) reason = "CSV flush failed";
  if (!reason.empty() || step != sample_count) {
    std::cerr << "Recording stopped at step " << step << ": " << reason << '\n';
    return false;
  }
  std::cout << "Recorded " << sample_count << " chirp samples\n";
  return true;
}
}  // namespace

int main(int argc, char ** argv)
{
  if (argc > 2 || (argc == 2 && std::string(argv[1]).rfind("--", 0) == 0)) {
    std::cerr << "Usage: " << argv[0] << " [output.csv]\n";
    return 1;
  }
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  try {
    ddt_hardware::DdtHardware hw("d1");
    ShutdownGuard guard{hw};
    guard.active = true;
    std::cout << "D1 visual guard URDF SHA256 "
              << d1_side_guard_data::kUrdfSha256 << '\n';
    if (!hw.initialize()) {
      std::cerr << "D1 hardware initialization failed\n";
      return 1;
    }
    if (!hw.is_ready()) {
      std::cerr << "D1 hardware not ready\n";
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
      return 1;
    }
    for (int j = 0; j < 12; ++j) {
      const double q = initial[ids[j]].position;
      const bool expected = j % 3 == 0 ? std::abs(q) < 0.65 :
                            j % 3 == 1 ? q > -0.2 && q < 2.0 :
                                         q > safety_lower(j) && q < safety_upper(j);
      if (!expected) {
        std::cerr << "Joint " << ids[j] << " is outside the supported start range\n";
        return 1;
      }
    }
    std::array<double, 12> original{};
    for (int j = 0; j < 12; ++j) original[j] = initial[ids[j]].position;
    bool ok = move_pose(hw, fold, "Fold");
    if (ok) {
      ok = move_pose(hw, trajectory_center, "Trajectory A center");
      if (ok) ok = record(hw, argc > 1 ? argv[1] : "chirp.csv");
      if (ok && running) ok = move_pose(hw, original, "Return to starting pose");
    }
    return ok ? 0 : 1;
  } catch (const std::exception & e) {
    std::cerr << "D1 hardware error: " << e.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "Unknown D1 hardware error\n";
    return 1;
  }
}
