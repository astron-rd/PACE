#include <chrono>
#include <iostream>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xnpy.hpp>

#include "IDG.h"
#include "init.h"

int main() {
  // Parameters
  const size_t nr_correlations_in = 2;  // XX, YY
  const size_t nr_correlations_out = 1; // I
  const size_t subgrid_size = 32;
  const size_t grid_size = 1024;
  const float observation_hours = 4.0f;
  const size_t nr_timesteps = static_cast<size_t>(observation_hours * 3600);
  const size_t nr_channels = 16;
  const float w_step = 1.0f;

  const double start_frequency = 150e6;
  const double frequency_increment = 1e6;
  const double end_frequency =
      start_frequency + (nr_channels - 1) * frequency_increment;

  const double speed_of_light = 299792458.0;
  const double image_size = speed_of_light / end_frequency;

  const size_t nr_stations = 20;
  const size_t nr_baselines = nr_stations * (nr_stations - 1) / 2;

  // Generate UVW coordinates
  std::cout << "Initialize UVW" << std::endl;
  xt::xarray<UVW> uvw = get_uvw(observation_hours, nr_baselines, grid_size);
//   xt::dump_npy("uvw.npy", uvw);

  // Generate frequencies
  std::cout << "Initialize frequencies" << std::endl;
  xt::xarray<float> frequencies =
      get_frequencies(static_cast<float>(start_frequency),
                      static_cast<float>(frequency_increment), nr_channels);
  xt::xarray<float> wavenumbers = (frequencies * static_cast<float>(2 * M_PI)) /
                                  static_cast<float>(speed_of_light);

  // Generate metadata
  std::cout << "Initialize metadata" << std::endl;
  std::vector<Metadata> metadata =
      get_metadata(nr_channels, subgrid_size, grid_size, uvw);
  size_t nr_subgrids = metadata.size();
  const std::array<size_t, 2> shape = {nr_subgrids, 2};
  xt::xarray<size_t> subgrid_coordinates = xt::zeros<size_t>(shape);
  for (size_t s = 0; s < nr_subgrids; ++s) {
    subgrid_coordinates(s, 0) = metadata[s].coordinate.x;
    subgrid_coordinates(s, 1) = metadata[s].coordinate.y;
  }
  xt::dump_npy("subgrid_coordinates.npy", subgrid_coordinates);

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
  std::cout << "\tnr_subgrids: " << nr_subgrids << std::endl;
  std::cout << "\tgrid_size: " << grid_size << std::endl;

  // Generate visibilities
  std::cout << "Initialize visibilities" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  xt::xarray<VisibilityType> visibilities = get_visibilities(
      nr_correlations_in, nr_channels, nr_timesteps, nr_baselines,
      static_cast<float>(image_size), grid_size, frequencies, uvw);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "runtime: " << std::chrono::duration<double>(end - start).count()
            << " seconds" << std::endl;
  xt::dump_npy("visibilities.npy", visibilities);

  // Allocate grid
  xt::xarray<std::complex<float>> grid = xt::zeros<std::complex<float>>(
      {nr_correlations_out, grid_size, grid_size});

  // Taper
  std::cout << "Initialize taper" << std::endl;
  xt::xarray<float> taper = get_taper(subgrid_size);
  xt::dump_npy("taper.npy", taper);

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

  // Add subgrids to grid
  std::cout << "Add subgrids to grid" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  gridder.add_subgrids_to_grid(metadata, subgrids, grid);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "runtime: " << std::chrono::duration<double>(end - start).count()
            << " seconds" << std::endl;
  xt::dump_npy("grid1.npy", taper);

  // Transform to image domain
  std::cout << "Transform to image domain" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  gridder.transform(FourierDomainToImageDomain, grid);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "runtime: " << std::chrono::duration<double>(end - start).count()
            << " seconds" << std::endl;
  xt::dump_npy("grid2.npy", taper);

  return EXIT_SUCCESS;
}
