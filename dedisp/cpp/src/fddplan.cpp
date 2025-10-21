#include "fddplan.hpp"

#include <iostream>
#include <cmath>

namespace dedisp {

FDDPlan::FDDPlan(size_t n_channels, float time_resolution, float peak_frequency, float frequency_resolution) : dm_count_{0}, n_channels_{n_channels}, max_delay_{0}, time_resolution_{time_resolution}, peak_frequency_{peak_frequency}, frequency_resolution_{-std::abs(frequency_resolution)} {
  std::cout << "Initialise FDD Plan..." << std::endl;

  generate_delay_table();
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

void FDDPlan::show() const {
  std::cout << "FDD Plan Summary" << std::endl;
  std::cout << "  nr channels:          " << n_channels_ << std::endl;
  std::cout << "  dm count:             " << dm_count_ << std::endl;
  std::cout << "  max delay:            " << max_delay_ << std::endl;
  std::cout << "  time resolution:      " << time_resolution_ << std::endl;
  std::cout << "  frequency resolution: " << frequency_resolution_ << std::endl;
  std::cout << "  peak frequency:       " << peak_frequency_ << std::endl;
}

void FDDPlan::generate_dm_list(float dm_start, float dm_end, float pulse_width, float tolerance) {
    // Fill the DM list
    // TODO: verify calculations.
    const double time_resolution = time_resolution_ * 1e6;
    const double f = (peak_frequency_ + ((n_channels_ / 2) - 0.5) * frequency_resolution_) * 1e-3;
    const double a = 8.3 * frequency_resolution_ / (f * f * f);
    const double a_squared = a * a;
    const double b_squared = a_squared * (double)(n_channels_ * n_channels_ / 16.0);
    const double tolerance_squared = tolerance * tolerance;
    const double c = (time_resolution_ * time_resolution_ + pulse_width * pulse_width) * (tolerance_squared - 1.0);

    dm_list_.clear();
    dm_list_.push_back(dm_start);
    while (dm_list_.back() < dm_end) {
        const double previous_dm = dm_list_.back();
        const double previous_dm_squared = previous_dm * previous_dm;
        const double k = c + tolerance_squared * a_squared * previous_dm_squared;
        const double dm = ((b_squared * previous_dm + std::sqrt(-a_squared * b_squared * previous_dm_squared + (a_squared + b_squared) * k)) / (a_squared + b_squared));
        dm_list_.push_back(dm);
    }

    dm_count_ = dm_list_.size();

    // Calculate and store the maximum delay
    max_delay_ = static_cast<size_t>(dm_list_.back() * delay_table_.back() + 0.5);
}


void FDDPlan::generate_delay_table() {
    delay_table_.resize(n_channels_);


}

void FDDPlan::generate_spin_frequency_table(size_t n_frequencies, size_t n_samples, float time_resolution) {

}

} // dedisp
