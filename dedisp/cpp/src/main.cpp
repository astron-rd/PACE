#include <chrono>
#include <iostream>

#include "fddplan.hpp"

int main() {
    // Mock input
    const float sampling_period = 250.0E-6;                                  // Base is 250 microsecond time samples
    const float down_sampling_ratio = 1.0;
    const float observation_duration = 30.0;                                 // Observation duration in seconds
    const float time_resolution = down_sampling_ratio * sampling_period;    // s (0.25 ms sampling)
    const float peak_frequency = 1581.0;                                     // MHz (highest channel!)
    const float bandwidth = 100.0;                                           // MHz
    const size_t n_channels = 1024;
    const float frequency_resolution = -1.0 * bandwidth / n_channels;        // MHz   (This must be negative!)

    const size_t n_samples = observation_duration / time_resolution;
    const float data_rms = 25.0;
    const float signal_dm = 41.159;
    const float signal_time = 3.14159;                                       // seconds into time series (at f0)
    const float signal_amplitude = 25.0;                                     // amplitude of signal

    const float dm_start = 2.0;                                              // pc cm^-3
    const float dm_end = 100.0;                                              // pc cm^-3
    const float pulse_width = 4.0;                                           // ms
    const float dm_tolerance = 1.25;
    const size_t n_bits_in = 8;
    const size_t n_bits_out = 32;

    size_t dm_count;
    size_t max_delay;
    size_t n_samples_computed;
    unsigned char *input = 0;
    unsigned char *output = 0;

    // Initialise and execute the FDD plan
    std::cout << "Initialise FDD Plan <--" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    dedisp::FDDPlan fdd_plan(n_channels, time_resolution, peak_frequency, frequency_resolution);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "--> runtime: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;

    std::cout << "Generate DM list <--" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    fdd_plan.generate_dm_list(dm_start, dm_end, pulse_width, dm_tolerance);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "--> runtime: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;

    // const std::vector<float> dm_list = fdd_plan.get_dm_list();
    // std::cout << "DM list = ";
    // for (const float dm : dm_list) {
    //     std::cout << dm << ",";
    // }
    // std::cout << std::endl;

    std::cout << "Execute FDD Plan <--" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    fdd_plan.execute(n_samples, input, n_bits_in, output, n_bits_out);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "--> runtime: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;

    fdd_plan.show();
}
