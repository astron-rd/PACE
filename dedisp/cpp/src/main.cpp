#include <chrono>
#include <iostream>

#include "fddplan.hpp"

int main() {
    // Mock input
    float sampling_period = 250.0E-6;                                  // Base is 250 microsecond time samples
    float down_sampling_ratio = 1.0;
    float observation_duration = 30.0;                                 // Observation duration in seconds
    float time_resolution = down_sampling_ratio * sampling_period;    // s (0.25 ms sampling)
    float peak_frequency = 1581.0;                                     // MHz (highest channel!)
    float bandwidth = 100.0;                                           // MHz
    size_t n_channels = 1024;
    float frequency_resolution = -1.0 * bandwidth / n_channels;        // MHz   (This must be negative!)

    size_t n_samples = observation_duration / time_resolution;
    float data_rms = 25.0;
    float signal_dm = 41.159;
    float signal_time = 3.14159;                                       // seconds into time series (at f0)
    float signal_amplitude = 25.0;                                     // amplitude of signal

    float dm_start = 2.0;                                              // pc cm^-3
    float dm_end = 100.0;                                              // pc cm^-3
    float pulse_width = 4.0;                                           // ms
    float dm_tolerance = 1.25;
    size_t n_bits_in = 8;
    size_t n_bits_out = 32;

    size_t dm_count;
    size_t max_delay;
    size_t n_samples_computed;
    unsigned char *input = 0;
    unsigned char *output = 0;

    // Initialise and execute the FDD plan
    dedisp::FDDPlan fdd_plan(n_channels, time_resolution, peak_frequency, frequency_resolution);

    fdd_plan.generate_dm_list(dm_start, dm_end, pulse_width, dm_tolerance);

    fdd_plan.execute(n_samples, input, n_bits_in, output, n_bits_out);

    fdd_plan.show();
}
