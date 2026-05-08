#pragma once

#include "cxxopts.hpp"
#include <iostream>

#include "util.h"

class Settings {
public:
  Settings(const cxxopts::ParseResult &result) {
    // Parse input parameters
    subgrid_size = result["subgrid_size"].as<size_t>();
    grid_size = result["grid_size"].as<size_t>();
    observation_hours = result["observation_hours"].as<float>();
    nr_channels = result["nr_channels"].as<size_t>();
    nr_stations = result["nr_stations"].as<size_t>();
    start_frequency = result["start_frequency"].as<double>();
    frequency_increment = result["frequency_increment"].as<double>();

    // Parse output options
    output_uvw = result["output_uvw"].as<bool>();
    output_frequencies = result["output_frequencies"].as<bool>();
    output_metadata = result["output_metadata"].as<bool>();
    output_visibilities = result["output_visibilities"].as<bool>();
    output_taper = result["output_taper"].as<bool>();
    output_subgrids = result["output_subgrids"].as<bool>();
    output_grid = result["output_grid"].as<bool>();
    output_image = result["output_image"].as<bool>();

    // Parse timing options
    report_timing = result["report_timing"].as<bool>();

    // Derive derived params
    nr_timesteps = static_cast<size_t>(observation_hours * 3600);
    nr_baselines = nr_stations * (nr_stations - 1) / 2;
    end_frequency = start_frequency + (nr_channels - 1) * frequency_increment;
    image_size = speed_of_light / end_frequency;
  }

  const size_t nr_correlations_in = 2;  // XX, YY
  const size_t nr_correlations_out = 1; // I
  const float w_step = 1.0f;
  const double speed_of_light = 299792458.0;

  // Input parameters
  size_t subgrid_size;
  size_t grid_size;
  float observation_hours;
  size_t nr_channels;
  size_t nr_stations;
  double start_frequency;
  double frequency_increment;

  // Output generated input
  bool output_uvw;
  bool output_frequencies;
  bool output_metadata;
  bool output_visibilities;
  bool output_taper;

  // Output gridded data
  bool output_subgrids;
  bool output_grid;
  bool output_image;

  // Output timing data
  bool report_timing;

  // Derived parameters
  size_t nr_timesteps;
  size_t nr_baselines;
  double end_frequency;
  double image_size;

  void print_params() {
    print_header("PARAMETERS");
    std::cout << std::left << std::setw(40) << "nr_correlations_in"
              << std::right << std::setw(10) << nr_correlations_in << "\n";
    std::cout << std::left << std::setw(40) << "nr_correlations_out"
              << std::right << std::setw(10) << nr_correlations_out << "\n";
    std::cout << std::left << std::setw(40) << "start_frequency" << std::right
              << std::setw(10) << std::fixed << std::setprecision(1)
              << start_frequency * 1e-6 << "\n";
    std::cout << std::left << std::setw(40) << "frequency_increment"
              << std::right << std::setw(10) << std::fixed
              << std::setprecision(1) << frequency_increment * 1e-6 << "\n";
    std::cout << std::left << std::setw(40) << "nr_channels" << std::right
              << std::setw(10) << nr_channels << "\n";
    std::cout << std::left << std::setw(40) << "nr_timesteps" << std::right
              << std::setw(10) << nr_timesteps << "\n";
    std::cout << std::left << std::setw(40) << "nr_stations" << std::right
              << std::setw(10) << nr_stations << "\n";
    std::cout << std::left << std::setw(40) << "nr_baselines" << std::right
              << std::setw(10) << nr_baselines << "\n";
    std::cout << std::left << std::setw(40) << "subgrid_size" << std::right
              << std::setw(10) << subgrid_size << "\n";
    std::cout << std::left << std::setw(40) << "grid_size" << std::right
              << std::setw(10) << grid_size << "\n";
  }
};
