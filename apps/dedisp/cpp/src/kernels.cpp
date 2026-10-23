#include <iostream>
#include <numbers>

#include <omp.h>

#include "kernels.hpp"

namespace dedisp {

void fourier_domain_dedisperse(size_t dm_count, size_t n_frequencies,
                               size_t n_channels, float time_resolution,
                               float *spin_frequencies,
                               float *dispersion_measures, float *delays,
                               size_t stride_in, size_t stride_out,
                               std::complex<float> *input,
                               std::complex<float> *output) {
#ifdef DEDISP_GPU_TARGET_OMP
#pragma omp target teams distribute parallel for collapse(2)                   \
    map(to : spin_frequencies[0 : n_frequencies],                              \
            dispersion_measures[0 : dm_count], delays[0 : n_channels])         \
    map(to : input[0 : n_channels * stride_in])                                \
    map(from : output[0 : dm_count * stride_out])
#elif defined(DEDISP_GPU_TARGET_ACC)
#pragma acc parallel loop collapse(2)                                          \
    copyin(spin_frequencies[0 : n_frequencies],                                \
           dispersion_measures[0 : dm_count], delays[0 : n_channels])          \
        copyin(input[0 : n_channels * stride_in])                              \
            copyout(output[0 : dm_count * stride_out])
#else
#ifdef DEDISP_USE_OPENMP
#pragma omp parallel for
#endif
#endif
  for (size_t dm_index = 0; dm_index < dm_count; ++dm_index) {
    // Loop over spin frequencies
    for (size_t frequency_index = 0; frequency_index < n_frequencies;
         ++frequency_index) {
      // Sum over observing frequencies
      std::complex<float> sum{0, 0};

      // Loop over observing frequencies
      for (size_t channel_index = 0; channel_index < n_channels;
           ++channel_index) {
        // Calculate the DM delay for this channel. Computed inline (instead
        // of in a per-DM array) so the loop body has no shared scratch state
        // and can be offloaded as-is.
        const float dm_delay = dispersion_measures[dm_index] *
                               delays[channel_index] * time_resolution;

        // Compute phasor
        const float phase = 2.0f * std::numbers::pi_v<float> *
                            spin_frequencies[frequency_index] * dm_delay;
        std::complex<float> phasor{std::cosf(phase), std::sinf(phase)};

        // Load sample
        std::complex<float> *sample_ptr = &input[channel_index * stride_in];
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
