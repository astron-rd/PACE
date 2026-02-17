#include <chrono>
#include <iostream>

#include <cxxopts.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xnpy.hpp>

#include "IDG.h"
#include "init.h"

#include "idgtypes-io.h"

cxxopts::Options setupOptions(const char *argv[]) {
  cxxopts::Options options(argv[0], "Image-Domain Gridder");

  constexpr size_t kSubgridSize = 32;
  constexpr size_t kGridSize = 1024;
  constexpr float kObservationHours = 4.0f;
  constexpr size_t kNrChannels = 16;
  constexpr size_t kNrStations = 20;
  constexpr double kStartFrequency = 150e6;
  constexpr double kFrequencyIncrement = 1e6;

  constexpr bool kOutputData = false;
  constexpr bool kReportTiming = true;

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
  }

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
};

int main(int argc, const char *argv[]) {

  // Constants
  const size_t nr_correlations_in = 2;  // XX, YY
  const size_t nr_correlations_out = 1; // I
  const float w_step = 1.0f;
  const double speed_of_light = 299792458.0;

  // Command-line arguments
  Settings arguments(parseArguments(argc, argv));

  const size_t subgrid_size = arguments.subgrid_size;
  const size_t grid_size = arguments.grid_size;
  const float observation_hours = arguments.observation_hours;
  const size_t nr_channels = arguments.nr_channels;
  const double start_frequency = arguments.start_frequency;
  const double frequency_increment = arguments.frequency_increment;
  const size_t nr_stations = arguments.nr_stations;

  // Derived parameters
  const size_t nr_timesteps = static_cast<size_t>(observation_hours * 3600);
  const size_t nr_baselines = nr_stations * (nr_stations - 1) / 2;
  const double end_frequency =
      start_frequency + (nr_channels - 1) * frequency_increment;
  const double image_size = speed_of_light / end_frequency;

  // Print parameters
  std::cout << "Parameters:" << std::endl;
  std::cout << "\tnr_correlations_in: " << nr_correlations_in << std::endl;
  std::cout << "\tnr_correlations_out: " << nr_correlations_out << std::endl;
  std::cout << "\tstart_frequency: " << start_frequency * 1e-6 << " MHz"
            << std::endl;
  std::cout << "\tfrequency_increment: " << frequency_increment * 1e-6 << " MHz"
            << std::endl;
  std::cout << "\tnr_channels: " << nr_channels << std::endl;
  std::cout << "\tnr_timesteps: " << nr_timesteps << std::endl;
  std::cout << "\tnr_stations: " << nr_stations << std::endl;
  std::cout << "\tnr_baselines: " << nr_baselines << std::endl;
  std::cout << "\tsubgrid_size: " << subgrid_size << std::endl;
  std::cout << "\tgrid_size: " << grid_size << std::endl;

  // Generate UVW coordinates
  std::cout << "Initialize UVW" << std::endl;
  xt::xarray<UVW> uvw = get_uvw(observation_hours, nr_baselines, grid_size);
  const std::array<size_t, 3> shape_uvw = {nr_baselines, nr_timesteps, 3};
  if (arguments.output_uvw) {
    xt::dump_npy("uvw.npy", uvw);
  }

  // Generate frequencies
  std::cout << "Initialize frequencies" << std::endl;
  xt::xarray<float> frequencies =
      get_frequencies(static_cast<float>(start_frequency),
                      static_cast<float>(frequency_increment), nr_channels);
  xt::xarray<float> wavenumbers = (frequencies * static_cast<float>(2 * M_PI)) /
                                  static_cast<float>(speed_of_light);

  // Generate metadata
  std::cout << "Initialize metadata" << std::endl;
  xt::xarray<Metadata> metadata =
      get_metadata(nr_channels, subgrid_size, grid_size, uvw);
  const size_t nr_subgrids = metadata.size();
  if (arguments.output_metadata) {
    xt::dump_npy("metadata.npy", metadata);
  }

  // Generate visibilities
  std::cout << "Initialize visibilities" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  xt::xarray<VisibilityType> visibilities = get_visibilities(
      nr_correlations_in, nr_channels, nr_timesteps, nr_baselines,
      static_cast<float>(image_size), grid_size, frequencies, uvw);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "runtime: " << std::chrono::duration<double>(end - start).count()
            << " seconds" << std::endl;
  if (arguments.output_visibilities) {
    xt::dump_npy("visibilities.npy", visibilities);
  }

  // Allocate grid
  xt::xarray<std::complex<float>> grid = xt::zeros<std::complex<float>>(
      {nr_correlations_out, grid_size, grid_size});

  // Taper
  std::cout << "Initialize taper" << std::endl;
  xt::xarray<float> taper = get_taper(subgrid_size);
  if (arguments.output_taper) {
    xt::dump_npy("taper.npy", taper);
  }

  // Allocate subgrids
  xt::xarray<std::complex<float>> subgrids = xt::zeros<std::complex<float>>(
      {nr_subgrids, nr_correlations_out, subgrid_size, subgrid_size});

  // Gridder
  std::cout << "Initialize gridder" << std::endl;
  Gridder gridder(nr_correlations_in, subgrid_size);

  // Grid visibilities onto subgrids
  std::cout << "Grid visibilities onto subgrids" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  gridder.grid_onto_subgrids(w_step, static_cast<float>(image_size), grid_size,
                             frequencies, uvw, visibilities, taper, metadata,
                             subgrids);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "runtime: " << std::chrono::duration<double>(end - start).count()
            << " seconds" << std::endl;
  if (arguments.output_subgrids) {
    xt::dump_npy("subgrids.npy", subgrids);
  }

  // Add subgrids to grid
  std::cout << "Add subgrids to grid" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  gridder.add_subgrids_to_grid(metadata, subgrids, grid);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "runtime: " << std::chrono::duration<double>(end - start).count()
            << " seconds" << std::endl;
  if (arguments.output_grid) {
    xt::dump_npy("grid.npy", grid);
  }

  // Transform to image domain
  std::cout << "Transform to image domain" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  gridder.transform(FourierDomainToImageDomain, grid);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "runtime: " << std::chrono::duration<double>(end - start).count()
            << " seconds" << std::endl;
  if (arguments.output_image) {
    xt::dump_npy("image.npy", grid);
  }

  return EXIT_SUCCESS;
}
