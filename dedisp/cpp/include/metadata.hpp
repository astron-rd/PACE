#pragma once

#include <cstddef>

#include "cxxopts.hpp"

namespace dedisp {

struct SignalInfo {
  // RMS of the noise in the randomly generated data
  float noise_rms;

  // Signal DM in pc cm^-3
  float dispersion_measure;

  // Arrival time of the pulse in seconds
  float arrival_time;

  // Amplitude of the signal
  float intensity;
};

struct ObservationInfo {
  // Duration of the observation in seconds
  float duration;

  // Duration of a sample in seconds
  float sampling_period;

  // Highest frequency in the observation (MHz)
  float peak_frequency;

  // Bandwith in MHz
  float bandwidth;

  // Number of channel
  size_t channels;
};

struct DedispersionConstraints {
  DedispersionConstraints(const cxxopts::ParseResult &result) {
    dm_start = result["dm-start"].as<float>();
    dm_end = result["dm-end"].as<float>();
    pulse_width = result["pulse-width"].as<float>();
    tolerance = result["dm-tolerance"].as<float>();
  }
  // Dispersion measures in pc cm^-3
  float dm_start;
  float dm_end;

  // Expected intrinsic width of the pulse in microseconds
  float pulse_width;

  // Smearing tolerance
  float tolerance;
};

} // namespace dedisp
