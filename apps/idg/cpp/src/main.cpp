#include <chrono>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

#include <cxxopts.hpp>
#include <xtensor/containers/xarray.hpp>

#include "h5cpp/dataspace/simple.hpp"
#include "h5cpp/file/file.hpp"
#include "h5cpp/file/functions.hpp"
#include "h5cpp/node/group.hpp"
#include <h5cpp/contrib/stl/complex.hpp>

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
      "input_path", "Path to the HDF5 file containing the input data.",
      cxxopts::value<std::filesystem::path>()->default_value(inputPath))(
      "subgrid_size", "Subgrid size",
      cxxopts::value<size_t>()->default_value("32"))(
      "grid_size", "Grid size",
      cxxopts::value<size_t>()->default_value("1024"))(
      "nr_correlations_out", "Number of correlations out",
      cxxopts::value<size_t>()->default_value("1"));

  options.add_options("Output gridded data")(
      "output_subgrids", "Output subgrids",
      cxxopts::value<bool>()->default_value(std::to_string(kOutputData)))(
      "output_grid", "Output grid",
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

  Inputs inputs = load_inputs(settings, timings);

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
  time_function(
      timings, "grid onto subgrids", [settings, gridder, inputs, &subgrids]() {
        gridder.grid_onto_subgrids(settings.w_step, inputs.image_size,
                                   settings.grid_size, inputs.wavenumbers,
                                   inputs.uvws, inputs.visibilities,
                                   inputs.taper, inputs.metadata, subgrids);
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

  hdf5::file::File output_file = hdf5::file::create("output.h5");
  hdf5::node::Group root_node = output_file.root();

  hdf5::datatype::Compound datatype =
      hdf5::datatype::Compound::create(sizeof(std::complex<float>));
  datatype.insert("r", 0, hdf5::datatype::TypeTrait<float>::create(float()));
  datatype.insert("i", alignof(float),
                  hdf5::datatype::TypeTrait<float>::create(float()));

  const std::vector<hsize_t> dims(grid.shape().begin(), grid.shape().end());
  auto dataspace = hdf5::dataspace::Simple(dims);
  auto grid_dataset = root_node.create_dataset("grid", datatype, dataspace);

  grid_dataset.write(*grid.data(), datatype, dataspace);

  return EXIT_SUCCESS;
}
