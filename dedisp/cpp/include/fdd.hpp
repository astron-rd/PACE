#pragma once

#include <vector>

namespace dedisp {

class FDDPlan {
public:
  FDDPlan(size_t n_channels, float time_resolution, float peak_frequency, float frequency_resolution);

  void execute(size_t n_samples, const unsigned char *input,
                       size_t n_bits_in, unsigned char *output, size_t n_bits_out);

  void generate_dm_list(float dm_start, float dm_end, float pulse_width, float tolerance);

  void show() const;

private:
  /// Generate a list of trial dispersion measures based on an algorithm by Lina Levin.
  void generate_dm_list(float dm_start, float dm_end, double time_resolution, double pulse_width, double peak_frequency, double frequency_resolution, size_t n_channels, double tolerance);

  /// Fill the dispersive delay table.
  void generate_delay_table(size_t n_channels, float time_resolution, float peak_frequency, float frequency_resolution);

  /// Fill the spin frequency table.
  void generate_spin_frequency_table(size_t n_frequencies, size_t n_samples, float time_resolution);

  // Size parameters
  size_t dm_count_;
  size_t n_channels_;
  size_t max_delay_;

  // Physical parameters
  float time_resolution_;
  float peak_frequency_;
  float frequency_resolution_;

  // Host arrays
  std::vector<float> dm_list_;
  std::vector<float> delay_table_;
  std::vector<float> spin_frequency_table_;
};

} // dedisp