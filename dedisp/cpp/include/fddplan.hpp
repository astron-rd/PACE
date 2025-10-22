#pragma once

#include <xtensor/containers/xarray.hpp>

namespace dedisp {

class FDDPlan {
public:
  FDDPlan(size_t n_channels, float time_resolution, float peak_frequency,
          float frequency_resolution);

  void execute(size_t n_samples, const unsigned char *input, size_t n_bits_in,
               unsigned char *output, size_t n_bits_out);

  /// Generate a list of trial dispersion measures based on an algorithm by Lina
  /// Levin.
  void generate_dm_list(float dm_start, float dm_end, float pulse_width,
                        float tolerance);

  void show() const;

  xt::xarray<float> get_dm_table() const { return dm_table_; };
  xt::xarray<float> get_delay_table() const { return delay_table_; };
  xt::xarray<float> get_spin_frequency_table() const {
    return spin_frequency_table_;
  };

private:
  /// Fill the dispersive delay table.
  void generate_delay_table();

  /// Fill the spin frequency table.
  void generate_spin_frequency_table(size_t n_frequencies, size_t n_samples);

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
};

} // namespace dedisp
