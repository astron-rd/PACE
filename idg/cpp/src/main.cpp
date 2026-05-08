#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <cxxopts.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xnpy.hpp>

#include "IDG.h"
#include "idgtypes.h"
#include "init.h"
#include "settings.h"

#include "util.h"

cxxopts::Options setupOptions(const char *argv[]) {
  cxxopts::Options options(argv[0], "Image-Domain Gridder");

  const std::string inputPath = "input.h5";

  constexpr size_t kSubgridSize = 32;
  constexpr size_t kGridSize = 1024;
  constexpr float kObservationHours = 4.0f;
  constexpr size_t kNrChannels = 16;
  constexpr size_t kNrStations = 20;
  constexpr double kStartFrequency = 150e6;
  constexpr double kFrequencyIncrement = 1e6;

  constexpr bool kOutputData = false;
  constexpr bool kReportTiming = true;

  options.add_options("Load input")(
      "load_input", "Load input from HDF5 file.",
      cxxopts::value<std::filesystem::path>()->default_value(inputPath));

  options.add_options("Input parameters")(
      "subgrid_size", "Size of the subgrid in pixels",
      cxxopts::value<size_t>()->default_value(std::to_string(kSubgridSize)))(
      "grid_size", "Size of the grid in pixels",
      cxxopts::value<size_t>()->default_value(std::to_string(kGridSize)))(
      "observation_hours", "Length of the observation in hours",
      cxxopts::value<float>()->default_value(
          std::to_string(kObservationHours)))(
      "nr_channels", "Number of frequency channels",
      cxxopts::value<size_t>()->default_value(std::to_string(kNrChannels)))(
      "nr_stations", "Number of stations",
      cxxopts::value<size_t>()->default_value(std::to_string(kNrStations)))(
      "start_frequency", "Starting frequency in hertz",
      cxxopts::value<double>()->default_value(std::to_string(kStartFrequency)))(
      "frequency_increment", "Frequency increment in hertz",
      cxxopts::value<double>()->default_value(
          std::to_string(kFrequencyIncrement)));

  options.add_options("Output generated input")(
      "output_uvw", "Output UVW data",
      cxxopts::value<bool>()->default_value(std::to_string(kOutputData)))(
      "output_frequencies", "Output frequencies",
      cxxopts::value<bool>()->default_value(std::to_string(kOutputData)))(
      "output_metadata", "Output metadata",
      cxxopts::value<bool>()->default_value(std::to_string(kOutputData)))(
      "output_visibilities", "Output visibilities",
      cxxopts::value<bool>()->default_value(std::to_string(kOutputData)))(
      "output_taper", "Output taper",
      cxxopts::value<bool>()->default_value(std::to_string(kOutputData)));

  options.add_options("Output gridded data")(
      "output_subgrids", "Output subgrids",
      cxxopts::value<bool>()->default_value(std::to_string(kOutputData)))(
      "output_grid", "Output grid",
      cxxopts::value<bool>()->default_value(std::to_string(kOutputData)))(
      "output_image", "Output image",
      cxxopts::value<bool>()->default_value(std::to_string(kOutputData)));

  options.add_options("Timing")(
      "report_timing", "Report timing data",
      cxxopts::value<bool>()->default_value(std::to_string(kReportTiming)));

  options.add_options("General")("h,help", "Print help");

  return options;
}

cxxopts::ParseResult parseArguments(int argc, const char *argv[]) {
  cxxopts::Options options = setupOptions(argv);

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(EXIT_SUCCESS);
  }

  return result;
}

int main(int argc, const char *argv[]) {
  // Command-line arguments
  Settings settings(parseArguments(argc, argv));

  std::vector<std::pair<const std::string, double>> timings;

  Inputs inputs = generate_inputs(settings, timings);

  xt::xarray<float> taper;
  time_function(timings, "generate taper", [settings, &taper]() {
    taper = get_taper(settings.subgrid_size);
  });

  xt::xarray<std::complex<float>> subgrids;
  time_function(timings, "allocate subgrids", [settings, inputs, &subgrids]() {
    subgrids = xt::zeros<std::complex<float>>(
        {inputs.nr_subgrids, settings.nr_correlations_out,
         settings.subgrid_size, settings.subgrid_size});
  });

  xt::xarray<std::complex<float>> grid;
  time_function(timings, "allocate grid", [settings, &grid]() {
    grid = xt::zeros<std::complex<float>>(
        {settings.nr_correlations_out, settings.grid_size, settings.grid_size});
  });

  // Initialize gridder
  Gridder gridder(settings.nr_correlations_in, settings.subgrid_size);

  print_header("MAIN");

  auto main_start = std::chrono::high_resolution_clock::now();
  time_function(timings, "grid onto subgrids",
                [settings, gridder, inputs, taper, &subgrids]() {
                  gridder.grid_onto_subgrids(
                      settings.w_step, static_cast<float>(settings.image_size),
                      settings.grid_size, inputs.wavenumbers, inputs.uvws,
                      inputs.visibilities, taper, inputs.metadata, subgrids);
                });

  time_function(timings, "ifft the subgrids",
                [gridder, &subgrids]() { gridder.ifft_subgrids(subgrids); });

  time_function(timings, "add subgrids to grid",
                [gridder, inputs, subgrids, &grid]() {
                  gridder.add_subgrids_to_grid(inputs.metadata, subgrids, grid);
                });

  time_function(timings, "transform grid",
                [gridder, &grid]() { gridder.transform(grid); });

  // Print timings summary
  if (settings.report_timing) {
    print_header("TIMINGS");

    double total_time = 0.0;
    for (const auto &timer : timings) {
      total_time += timer.second;
    }

    for (const auto &timer : timings) {
      double percent = (timer.second / total_time) * 100.0;
      print_timing(timer.first, timer.second, percent);
    }

    print_timing("Total", total_time, 100.0);
  }
  if (settings.output_image) {
    xt::dump_npy("image.npy", grid);
  }

  return EXIT_SUCCESS;
}
