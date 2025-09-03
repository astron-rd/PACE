#include <complex>
#include <cstddef>
#include <vector>

#include <omp.h>

#include <xtensor-fftw/basic.hpp>
#include <xtensor-fftw/helper.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>

#include "IDG.h"
#include "idgtypes.h"
#include "kernels.h"

void Gridder::grid_onto_subgrids(
    float w_step, float image_size, size_t grid_size,
    const xt::xarray<float> &wavenumbers, const xt::xarray<UVW> &uvw,
    const xt::xarray<std::complex<float>> &visibilities,
    const xt::xarray<float> &taper, const std::vector<Metadata> &metadata,
    xt::xarray<std::complex<float>> &subgrids) const {
  assert(nr_correlations_in_ == visibilities.shape()[3]);
  assert(nr_correlations_out_ == subgrids.shape()[1]);
  assert(subgrid_size_ == subgrids.shape()[2]);
  const size_t nr_subgrids = metadata.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t s = 0; s < nr_subgrids; ++s) {
    auto subgrid =
        xt::eval(xt::view(subgrids, s, xt::all(), xt::all(), xt::all()));

    visibilities_to_subgrid(s, metadata, w_step, grid_size, image_size,
                            wavenumbers, visibilities, uvw, taper,
                            nr_correlations_in_, subgrid_size_, subgrid);

    subgrid = xt::fftw::ifft2(subgrid);

    xt::view(subgrids, s, xt::all(), xt::all(), xt::all()) = subgrid;
  }
}

void Gridder::add_subgrids_to_grid(
    const std::vector<Metadata> &metadata,
    const xt::xarray<std::complex<float>> &subgrids,
    xt::xarray<std::complex<float>> &grid) const {
  const size_t nr_correlations = grid.shape()[0];
  const size_t grid_size = grid.shape()[1];
  const size_t nr_subgrids = subgrids.shape()[0];
  const size_t subgrid_size = subgrids.shape()[2];

  xt::xarray<std::complex<float>> phasor =
      compute_phasor(static_cast<int>(subgrid_size));

  for (size_t s = 0; s < nr_subgrids; ++s) {
    add_subgrid_to_grid(s, metadata, subgrids, grid, phasor, nr_correlations,
                        subgrid_size, grid_size);
  }
}

void Gridder::transform(int direction,
                        xt::xarray<std::complex<float>> &grid) const {
  assert(nr_correlations_out_ == grid.shape()[0]);
  const size_t height = grid.shape()[1];
  const size_t width = grid.shape()[2];
  assert(height == width);

  grid = xt::fftw::fftshift(grid);

  if (direction == FourierDomainToImageDomain) {
    grid = xt::fftw::ifft2(grid);
  } else {
    grid = xt::fftw::fft2(grid);
  }

  grid = xt::fftw::fftshift(grid);

  std::complex<float> scale = {2.0f, 0.0f};
  if (direction == FourierDomainToImageDomain) {
    grid *= scale;
  } else {
    grid /= scale;
  }
}
