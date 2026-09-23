#pragma once

#include <cstddef>

#include <xtensor-wrappers/plan_batch.hpp>
#include <xtensor/containers/xarray.hpp>

namespace dedisp {

class FDDPlan {
public:
  FDDPlan(size_t n_channels, float time_resolution, float peak_frequency,
          float frequency_resolution);

  xt::xarray<float> execute(const xt::xarray<uint8_t> &input);

  // Generate a list of trial dispersion measures based on an algorithm by Lina
  // Levin.
  void generate_dm_list(float dm_start, float dm_end, float pulse_width,
                        float tolerance);

  // Generate a list of linearly spaced trial dispersion measures.
  // Note: only meant for debugging purposes!
  void generate_linear_dm_list(float dm_start, float dm_end, float dm_step);

  void show() const;

  xt::xarray<float> get_dm_table() const { return dm_table_; };
  xt::xarray<float> get_delay_table() const { return delay_table_; };
  xt::xarray<float> get_spin_frequency_table() const {
    return spin_frequency_table_;
  };

  size_t dm_count() const { return dm_count_; };
  size_t n_channels() const { return n_channels_; };
  size_t max_delay() const { return max_delay_; };

private:
  // Fill the dispersive delay table.
  void generate_delay_table();

  // Fill the spin frequency table.
  void generate_spin_frequency_table(size_t n_frequencies, size_t n_samples);

  // Allocate/resize the scratch buffers and (re)build the FFT plans when the
  // transform shapes change (n_samples_padded / dm_count). The plans bind the
  // buffers' addresses, so a buffer must never be reallocated without
  // rebuilding its plan, which is exactly what this method does.
  void setup_fft(size_t n_samples_padded, size_t n_fft_frequency_bins);

  // Size parameters
  size_t dm_count_;
  size_t n_channels_;
  size_t max_delay_;

  // Physical parameters
  float time_resolution_;
  float peak_frequency_;
  float frequency_resolution_;

  // Host arrays
  xt::xarray<float> dm_table_;
  xt::xarray<float> delay_table_;
  xt::xarray<float> spin_frequency_table_;

  // Scratch buffers used for the FFT plans
  xt::xarray<float> transposed_input_;
  xt::xarray<std::complex<float>> dm_scratch_;

  // FFT plans, executed per execute() call:
  //   rfft_plan_.output()   == {n_channels, n_fft_frequency_bins} (complex)
  //   irfft_plan_.output()  == {dm_count, n_samples_padded}      (real)
  xt::fftw::batch_plan<float> rfft_plan_;
  xt::fftw::batch_plan<float, float> irfft_plan_;
  size_t plan_n_samples_padded_ = 0;
  size_t plan_dm_count_ = 0;
};

} // namespace dedisp
