#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>

#include "metadata.hpp"

namespace dedisp {

/// Generate a dispersed signal with Gaussian noise.
xt::xarray<float>
simulate_dispersed_signal(const dedisp::SignalInfo &signal,
                          const dedisp::ObservationInfo &observation);

/// ...
uint8_t quantise(float value_in);

/// @brief
/// @param offset use this to undo quantization, e.g. 128 for 8-bits
/// @param scale  // use this to prevent overflows when summing the data
// template <typename InputType, typename OutputType>
// void transpose_data(size_t height, size_t width, size_t in_stride,
//                     size_t out_stride, float offset, float scale,
//                     const InputType *input, OutputType *output);
template <typename InputType, typename OutputType>
void transpose_data(size_t height, size_t width, size_t in_stride,
                    size_t out_stride, float offset, float scale,
                    const InputType *input, OutputType *output) {
// TODO: use OpenMP
// #pragma omp parallel for
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      const InputType *input_ptr = &input[x * in_stride];
      OutputType *output_ptr = &output[y * out_stride];
      output_ptr[x] = (static_cast<OutputType>(input_ptr[y]) - offset) / scale;
    }
  }
}

/// round up int a to a multiple of int b
inline int round_up(int a, int b) { return ((a + b - 1) / b) * b; }

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