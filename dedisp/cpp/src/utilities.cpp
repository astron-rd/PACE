#include <cmath>

#include "utilities.hpp"

namespace dedisp {

xt::xarray<float>
simulate_dispersed_signal(const dedisp::SignalInfo &signal,
                          const dedisp::ObservationInfo &observation) {
  const float frequency_resolution =
      -1.0f * observation.bandwidth / observation.channels;
  const size_t n_samples = observation.duration / observation.sampling_period;

  std::array<size_t, 2> shape = {n_samples, observation.channels};
  xt::xarray<float> data = xt::ones<float>(shape);
  // xt::xarray<float> data =
  //     signal.noise_rms * xt::random::randn<float>(shape, 0.0f, 1.0f);

  for (size_t channel = 0; channel < observation.channels; ++channel) {
    // Calculate the sample index corresponding to the frequency bin
    const float a =
        1 / (observation.peak_frequency + channel * frequency_resolution);
    const float b = 1 / observation.peak_frequency;

    const float channel_delay =
        signal.dispersion_measure * 4.15e3 * (a * a - b * b);

    // Embed the dispersed signal.
    const size_t sample = static_cast<size_t>(
        (signal.arrival_time + channel_delay) / observation.sampling_period);
    data(sample, channel) += signal.intensity;
  }

  return data;
}

uint8_t quantise(float value_in) {
  const float value = value_in + 127.5f;
  uint8_t value_out;
  if (value > 255.0f) {
    value_out = 255;
  } else if (value < 0.0f) {
    value_out = 0;
  } else {
    value_out = round(value_in);
  }
  return value_out;
}

} // namespace dedisp