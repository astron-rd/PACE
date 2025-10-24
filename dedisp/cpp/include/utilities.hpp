#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>

#include "metadata.hpp"

namespace dedisp {

/// Generate a dispersed signal with Gaussian noise.
xt::xarray<float>
simulate_dispersed_signal(const dedisp::SignalInfo &signal,
                          const dedisp::ObservationInfo &observation);

namespace benchmark {

class Timer {
public:
  Timer()
      : is_running{false}, time_sum{std::chrono::duration<double>::zero()},
        time_start{} {};

  void start() {
    if (!is_running) {
      time_start = std::chrono::high_resolution_clock::now();
      is_running = true;
    }
  }

  void pause() {
    if (is_running) {
      auto now = std::chrono::high_resolution_clock::now();
      time_sum += now - time_start;

      is_running = false;
    }
  }

  void reset() {
    time_sum = std::chrono::duration<double>::zero();
    is_running = false;
  }

  double duration() { return time_sum.count(); }

private:
  bool is_running;

  std::chrono::duration<double> time_sum;
  std::chrono::system_clock::time_point time_start;
};

} // namespace benchmark
} // namespace dedisp