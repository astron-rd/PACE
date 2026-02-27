#include <cmath>
#include <iostream>

#include <xtensor-fftw/basic.hpp>
#include <xtensor-fftw/helper.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>

#include "fddplan.hpp"
#include "kernels.hpp"
#include "utilities.hpp"

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

xt::xarray<float> FDDPlan::execute(const xt::xarray<uint8_t> &input) {
  const size_t n_samples = input.shape(1); // input has dimensions channel x samples
  const size_t n_spin_frequencies = (n_samples / 2 + 1);
  const size_t n_output_samples = n_samples - max_delay_;

  // TODO: understand where all this comes from..!?
  const bool use_zero_padding = true;
  const size_t n_samples_fft =
      use_zero_padding ? round_up(n_samples + 1, 16384) : n_samples;
  const size_t n_samples_padded = round_up(n_samples_fft + 1, 1024);
  const size_t n_fft_frequency_bins = n_samples_padded / 2 + 1;

  // Input is in the frequency domain, while the output is in the DM domain.
  const std::vector<size_t> input_shape = {n_channels_, n_samples_padded};
  xt::xarray<float> frequency_data(input_shape);

  const std::vector<size_t> output_shape = {dm_count_, n_samples_padded};
  xt::xarray<float> dm_data(output_shape);

  // Allocate scratch arrays.
  const std::vector<size_t> frequency_scratch_shape = {n_channels_, n_fft_frequency_bins};
  xt::xarray<std::complex<float>> frequency_scratch(frequency_scratch_shape);

  const std::vector<size_t> dm_scratch_shape = {dm_count_, n_fft_frequency_bins};
  xt::xarray<std::complex<float>> dm_scratch(dm_scratch_shape);

  // 1. Generate spin table
  std::cout << "(1) Generate the spin frequency table." << std::endl;

  generate_spin_frequency_table(n_spin_frequencies, n_samples);

  std::cout << spin_frequency_table_ << std::endl;

  // 2. Transpose data (convert input bytes to floats)
  std::cout << "(2) Transpose data: int -> float." << std::endl;

  constexpr float byte_offset = 127.5;
  transpose_data<uint8_t, float>(n_channels_, n_samples, n_channels_,
                         n_samples_padded, byte_offset, n_channels_,
                         input.data(), frequency_data.data());


  const std::string fn_transpose{"fdd-transpose.npy"};
  xt::dump_npy(fn_transpose, frequency_data);

  // 3. Real-to-complex FFT: time series data to frequency domain
  // Perform an FFT batched over frequency using OpenMP
  std::cout << "(3) Forward FFT: real-to-complex." << std::endl;

  #pragma omp parallel for
  for(size_t c = 0; c < n_channels_; ++c) {
    xt::xarray<float> time_samples = xt::eval(xt::row(frequency_data, c));
    time_samples = xt::fftw::fftshift(time_samples);
    xt::view(frequency_scratch, c, xt::all()) = xt::fftw::rfft(time_samples);
  }


  const std::string fn_r2c{"fdd-fft-r2c.npy"};
  xt::dump_npy(fn_r2c, frequency_scratch);

  // 4. Run dedispersion algorithm (CPU reference or optimised version)
  std::cout << "(4) Run dedispersion algorithm." << std::endl;

  // const size_t in_out_stride = n_fft_frequency_bins;
  const size_t in_out_stride = n_samples_padded / 2;
  dedisp::fourier_domain_dedisperse(
    dm_count_, n_spin_frequencies, n_channels_, time_resolution_,
    spin_frequency_table_.data(), dm_table_.data(), delay_table_.data(),
    in_out_stride, in_out_stride, frequency_scratch.data(), dm_scratch.data()
  );

  const std::string fn_dedisp{"fdd-dedisp.npy"};
  xt::dump_npy(fn_dedisp, dm_scratch);

  // 5. Complex-to-real FFT: frequency domain back to time series data
  // Perform an FFT batched along the DM axis using OpenMP
  std::cout << "(5) Inverse FFT: complex-to-real." << std::endl;

  #pragma omp parallel for
  for(size_t d = 0; d < dm_count_; ++d) {
    xt::xarray<std::complex<float>> samples = xt::eval(xt::row(dm_scratch, d));
    samples = xt::fftw::fftshift(samples);
    xt::view(dm_data, d, xt::all()) = xt::fftw::irfft(samples);
  }

  // CLEAN UP..
  std::cout << "dm count = " << dm_count_;
  std::cout << " / output samps = " << n_output_samples << '\n';
  const std::vector<size_t> computed_shape = {dm_count_, n_output_samples};
  xt::xarray<float> computed_data(computed_shape);
  for(size_t d = 0; d < dm_count_; ++d) {
    for (size_t s = 0; s < n_output_samples; ++s) {
      xt::view(computed_data, d, s) = xt::view(dm_data, d, s);
    }
  }

  return computed_data;
}

void FDDPlan::show() const {
  std::cout << "FDD Plan Summary" << std::endl;
  std::cout << "  nr channels:          " << n_channels_ << std::endl;
  std::cout << "  nr dm trials:         " << dm_count_ << std::endl;
  std::cout << "  max delay:            " << max_delay_ * time_resolution_ << " s (" << max_delay_ << " samples)" << std::endl;
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
      (time_resolution * time_resolution + pulse_width * pulse_width) *
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
  const float max_dm = dm_table_(dm_count_ - 1);
  const float max_delay = delay_table_(n_channels_ - 1);
  max_delay_ = static_cast<size_t>(max_dm * max_delay + 0.5);
}

void FDDPlan::generate_linear_dm_list(float dm_start, float dm_end, float dm_step) {
  assert(dm_step > 0);

  // Linearly fill the DM list
  std::vector<float> dm_list = {dm_start};
  while (dm_list.back() < dm_end) {
    dm_list.push_back(dm_list.back() + dm_step);
  }

  dm_count_ = dm_list.size();

  // Store the DM table in memory
  dm_table_ = xt::adapt(dm_list, {dm_count_});

  // Calculate and store the maximum delay
  const float max_dm = dm_table_(dm_count_ - 1);
  const float max_delay = delay_table_(n_channels_ - 1);
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

#pragma omp parallel for
  for (size_t i = 0; i < n_spin_frequencies; ++i) {
    spin_frequency_table_(i) = i * (1.0f / (n_samples * time_resolution_));
  }
}

} // namespace dedisp
