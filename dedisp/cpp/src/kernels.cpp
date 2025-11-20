#include <numbers>

#include "kernels.hpp"

namespace dedisp {

// TODO: update interface!?
void fourier_domain_dedisperse(size_t dm_count, size_t n_frequencies,
                               size_t n_channels, size_t time_resolution,
                               float *spin_frequencies,
                               float *dispersion_measures, float *delays,
                               size_t stride_in, size_t stride_out,
                               std::complex<float> *input,
                               std::complex<float> *output) {
  for (size_t dm_index = 0; dm_index < dm_count; ++dm_index) {
    // Calculate DM delays
    float dm_delays[n_channels];
    for (size_t channel_index = 0; channel_index < n_channels;
         ++channel_index) {
      dm_delays[channel_index] = dispersion_measures[dm_index] *
                                 delays[channel_index] * time_resolution;
    }

    // Loop over spin frequencies
    for (size_t frequency_index = 0; frequency_index < n_frequencies;
         ++frequency_index) {
      // Sum over observing frequencies
      std::complex<float> sum{0, 0};

      // Loop over observing frequencies
      for (size_t channel_index = 0; channel_index < n_channels;
           ++channel_index) {
        // Compute phasor
        float phase = 2.0f * std::numbers::pi_v<float> *
                      spin_frequencies[frequency_index] *
                      dm_delays[channel_index];
        std::complex<float> phasor{std::cos(phase), std::sin(phase)};

        // Load sample
        std::complex<float> *sample_ptr = &input[channel_index + stride_in];
        std::complex<float> sample = sample_ptr[frequency_index];

        // Update the sum
        sum += sample * phasor;
      }

      // Write the sum to the output
      auto *output_ptr = &output[dm_index * stride_out];
      output_ptr[frequency_index] = sum;
    }
  }
}

} // namespace dedisp
