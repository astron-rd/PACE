#pragma once

#include "cxxopts.hpp"
#include <filesystem>

class Settings {
public:
  Settings(const cxxopts::ParseResult &result) {
    // Input data options
    input_path = result["input_path"].as<std::filesystem::path>();
    subgrid_size = result["subgrid_size"].as<size_t>();
    grid_size = result["grid_size"].as<size_t>();
    nr_correlations_out = result["nr_correlations_out"].as<size_t>();

    // Parse output options
    output_subgrids = result["output_subgrids"].as<bool>();
    output_grid = result["output_grid"].as<bool>();

    // Parse timing options
    report_timing = result["report_timing"].as<bool>();
  }

  const size_t nr_correlations_in = 2; // XX, YY
  const float w_step = 1.0f;
  const double speed_of_light = 299792458.0;

  // Input data
  std::filesystem::path input_path;
  size_t subgrid_size;
  size_t grid_size;
  size_t nr_correlations_out;

  // Output gridded data
  bool output_subgrids;
  bool output_grid;

  // Output timing data
  bool report_timing;
};
