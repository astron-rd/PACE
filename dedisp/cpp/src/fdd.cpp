#include "fdd.hpp"

#include <iostream>

namespace dedisp {

FDDPlan::FDDPlan(size_t n_channels, float time_resolution, float peak_frequency, float frequency_resolution) : n_channels_{n_channels}, time_resolution_{time_resolution}, peak_frequency_{peak_frequency}, frequency_resolution_{frequency_resolution} {
  std::cout << "Initialise FDD Plan..." << std::endl;
}

void FDDPlan::execute(size_t n_samples, const unsigned char *input,
                       size_t n_bits_in, unsigned char *output, size_t n_bits_out) {
  std::cout << "Execute FDD Plan..." << std::endl;

  // Rough outline:
  // 1. Generate spin table
  // 2. Transpose data (convert input bytes to floats)
  // 3. Real-to-complex FFT: time series data to frequency domain
  // 4. Run dedispersion algorithm (CPU reference or optimised version)
  // 5. Complex-to-real FFT: frequency domain back to time series data
}

void FDDPlan::generate_dm_list(float dm_start, float dm_end, float pulse_width, float tolerance) {

}

void FDDPlan::show() const {
  std::cout << "FDD Plan Summary" << std::endl;
  std::cout << "  nr channels:          " << n_channels_ << std::endl;
  std::cout << "  dm count:             " << dm_count_ << std::endl;
  std::cout << "  max delay:            " << max_delay_ << std::endl;
  std::cout << "  time resolution:      " << time_resolution_ << std::endl;
  std::cout << "  frequency resolution: " << frequency_resolution_ << std::endl;
  std::cout << "  peak frequency:       " << peak_frequency_ << std::endl;
}

void FDDPlan::generate_dm_list(float dm_start, float dm_end, double time_resolution, double pulse_width, double peak_frequency, double frequency_resolution, size_t n_channels, double tolerance) {

}


void FDDPlan::generate_delay_table(size_t n_channels, float time_resolution, float peak_frequency, float frequency_resolution) {

}

void FDDPlan::generate_spin_frequency_table(size_t n_frequencies, size_t n_samples, float time_resolution) {

}

} // dedisp
