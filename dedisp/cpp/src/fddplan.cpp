#include <cmath>
#include <iostream>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/io/xio.hpp>

#include "fddplan.hpp"
#include "kernels.hpp"

namespace dedisp {

FDDPlan::FDDPlan(size_t n_channels, float time_resolution, float peak_frequency,
                 float frequency_resolution)
    : dm_count_{0}, n_channels_{n_channels}, max_delay_{0},
      time_resolution_{time_resolution}, peak_frequency_{peak_frequency},
      frequency_resolution_{-std::abs(frequency_resolution)} {
  // Generate the delay table without the DM factor, which is applied during
  // dedispersion.
  generate_delay_table();
}

void FDDPlan::execute(xt::xarray<float> input, xt::xarray<float> output) {
  const size_t n_samples = input.shape(0);
  const size_t n_spin_frequencies = (n_samples / 2 + 1);
  const size_t n_output_samples = n_samples - max_delay_;

  // TODO: implemented zero padded FFT; initial version without!
  const bool use_zero_padding = true;
  const size_t n_samples_fft = n_samples;
  const size_t n_samples_padded = n_samples_fft;

  const std::vector<size_t> input_shape = {n_channels_ * n_samples_padded};
  const std::vector<size_t> output_shape = {dm_count_ * n_samples_padded};
  xt::xarray<float> input_data(input_shape);
  xt::xarray<float> output_data(output_shape);

  std::cout << input_data;

  // 1. Generate spin table
  generate_spin_frequency_table(n_spin_frequencies, n_samples);

  // 2. Transpose data (convert input bytes to floats)
  // TODO: can be remove this step? I.e. have the dedispersion code input and
  // output lists of floats?

  // 3. Real-to-complex FFT: time series data to frequency domain
  // 4. Run dedispersion algorithm (CPU reference or optimised version)
  // 5. Complex-to-real FFT: frequency domain back to time series data
}

void FDDPlan::show() const {
  std::cout << "FDD Plan Summary" << std::endl;
  std::cout << "  nr channels:          " << n_channels_ << std::endl;
  std::cout << "  nr dm trials:         " << dm_count_ << std::endl;
  std::cout << "  max delay:            " << max_delay_ << " s" << std::endl;
  std::cout << "  time resolution:      " << time_resolution_ << " s"
            << std::endl;
  std::cout << "  frequency resolution: " << -frequency_resolution_ << " MHz"
            << std::endl;
  std::cout << "  peak frequency:       " << peak_frequency_ << " MHz"
            << std::endl;
}

void FDDPlan::generate_dm_list(float dm_start, float dm_end, float pulse_width,
                               float tolerance) {
  // Fill the DM list
  // TODO: verify calculations.
  const double time_resolution = time_resolution_ * 1e6;
  const double f =
      (peak_frequency_ + ((n_channels_ / 2) - 0.5) * frequency_resolution_) *
      1e-3;
  const double a = 8.3 * frequency_resolution_ / (f * f * f);
  const double a_squared = a * a;
  const double b_squared =
      a_squared * (double)(n_channels_ * n_channels_ / 16.0);
  const double tolerance_squared = tolerance * tolerance;
  const double c =
      (time_resolution_ * time_resolution_ + pulse_width * pulse_width) *
      (tolerance_squared - 1.0);

  std::vector<float> dm_list = {dm_start};
  while (dm_list.back() < dm_end) {
    const double previous_dm = dm_list.back();
    const double previous_dm_squared = previous_dm * previous_dm;
    const double k = c + tolerance_squared * a_squared * previous_dm_squared;
    const double dm = ((b_squared * previous_dm +
                        std::sqrt(-a_squared * b_squared * previous_dm_squared +
                                  (a_squared + b_squared) * k)) /
                       (a_squared + b_squared));
    dm_list.push_back(dm);
  }

  dm_count_ = dm_list.size();

  // Store the DM table in memory
  dm_table_ = xt::adapt(dm_list, {dm_count_});

  // Calculate and store the maximum delay
  const float max_dm = dm_table_(dm_count_);
  const float max_delay = delay_table_(n_channels_);
  max_delay_ = static_cast<size_t>(max_dm * max_delay + 0.5);
}

void FDDPlan::generate_delay_table() {
  delay_table_.resize({n_channels_});

  for (size_t channel = 0; channel < n_channels_; ++channel) {
    const float inverse_channel_frequency =
        1.0f / (peak_frequency_ + channel * frequency_resolution_);
    const float inverse_peak_frequency = 1.0f / peak_frequency_;

    delay_table_(channel) =
        4.148741601e3 / time_resolution_ *
        (inverse_channel_frequency * inverse_channel_frequency -
         inverse_peak_frequency * inverse_peak_frequency);
  }
}

void FDDPlan::generate_spin_frequency_table(size_t n_spin_frequencies,
                                            size_t n_samples) {
  spin_frequency_table_.resize({n_spin_frequencies});

  for (size_t i = 0; i < n_spin_frequencies; ++i) {
    spin_frequency_table_(i) = i * (1.0f / (n_samples * time_resolution_));
  }
}

} // namespace dedisp
