#include <complex>
#include <cstddef>

#include <omp.h>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/views/xview.hpp>

#include <xtensor-wrappers/plan.hpp>

#include "IDG.h"
#include "idgtypes.h"
#include "kernels.h"

void Gridder::grid_onto_subgrids(
    float w_step, float image_size, size_t grid_size,
    const xt::xarray<float> &wavenumbers, const xt::xarray<UVW> &uvw,
    const xt::xarray<std::complex<float>> &visibilities,
    const xt::xarray<float> &taper, const xt::xarray<Metadata> &metadata,
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

    xt::view(subgrids, s, xt::all(), xt::all(), xt::all()) = subgrid;
  }
}

void Gridder::ifft_subgrids(xt::xarray<std::complex<float>> &subgrids) const {
  const size_t nr_subgrids = subgrids.shape(0);
  const size_t nr_correlations = subgrids.shape(1);

  // One batched backward 2-D transform over all polarizations and subgrids.
  xt::fftw::batch_layout layout;
  layout.howmany = nr_subgrids * nr_correlations;
  layout.n = {static_cast<int>(subgrid_size_), static_cast<int>(subgrid_size_)};

  auto plan =
      xt::fftw::make_batch_fft_plan(subgrids.data(), layout, FFTW_BACKWARD);
  plan.execute();

  // xt::fftw::ifft2 normalises by 1/(subgrid_size^2); the raw c2c plan does
  // not, so reproduce that scaling and write the result back in place.
  const float scale = 1.0f / static_cast<float>(subgrid_size_ * subgrid_size_);
  subgrids =
      xt::eval(xt::reshape_view(plan.output(), subgrids.shape()) * scale);
}

void Gridder::add_subgrids_to_grid(
    const xt::xarray<Metadata> &metadata,
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

void Gridder::transform(xt::xarray<std::complex<float>> &grid) const {
  assert(nr_correlations_out_ == grid.shape()[0]);
  const size_t height = grid.shape()[1];
  const size_t width = grid.shape()[2];
  assert(height == width);
  const size_t nr_correlations = nr_correlations_out_;

  // Shift the grid for each polarization so the zero-frequency bin is at the
  // origin.
  xt::xarray<std::complex<float>> input =
      xt::zeros<std::complex<float>>({nr_correlations, height, width});
  for (size_t i = 0; i < nr_correlations; ++i) {
    auto src = xt::view(grid, i, xt::all(), xt::all());
    auto dst = xt::view(input, i, xt::all(), xt::all());
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        dst(y, x) = src((x + width / 2) % width, (y + height / 2) % height);
      }
    }
  }

  auto plan = xt::fftw::make_fft2_plan(input, FFTW_BACKWARD);
  plan.execute();
  const auto &out = plan.output();

  // Unshift and scale by 2/(height*width).
  const std::complex<float> scale{2.0f / static_cast<float>(height * width),
                                  0.0f};
  for (size_t i = 0; i < nr_correlations; ++i) {
    auto dst = xt::view(grid, i, xt::all(), xt::all());
    auto src = xt::view(out, i, xt::all(), xt::all());
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        dst((x + width / 2) % width, (y + height / 2) % height) =
            src(y, x) * scale;
      }
    }
  }
}
