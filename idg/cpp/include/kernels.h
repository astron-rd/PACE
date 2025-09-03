#include <complex>
#include <stdint.h>

#include <xtensor/containers/xarray.hpp>

#include "idgtypes.h"

void add_pt_src_to_baseline(size_t bl, size_t nr_timesteps, size_t nr_channels,
                            std::complex<float> amplitude,
                            const xt::xarray<float> &frequencies,
                            const xt::xarray<UVW> &uvw, float l, float m,
                            xt::xarray<VisibilityType> &visibilities);

xt::xarray<float> evaluate_spheroidal(const xt::xarray<float> &nu);

xt::xarray<std::complex<float>> compute_phasor(size_t subgrid_size);

void add_subgrid_to_grid(size_t s, const std::vector<Metadata> &metadata,
                         const xt::xarray<std::complex<float>> &subgrids,
                         xt::xarray<std::complex<float>> &grid,
                         const xt::xarray<std::complex<float>> &phasor,
                         size_t nr_correlations, size_t subgrid_size,
                         size_t grid_size);

std::vector<Metadata> compute_metadata(size_t grid_size, size_t subgrid_size,
                                       size_t nr_channels, size_t bl,
                                       const xt::xarray<UVW> &uvw,
                                       size_t max_group_size);

void visibilities_to_subgrid(
    size_t s, const std::vector<Metadata> &metadata, float w_step,
    size_t grid_size, float image_size, const xt::xarray<float> &wavenumbers,
    const xt::xarray<std::complex<float>> &visibilities,
    const xt::xarray<float> &uvw_u, const xt::xarray<float> &uvw_v,
    const xt::xarray<float> &uvw_w, const xt::xarray<float> &taper,
    size_t nr_correlations_in, size_t subgrid_size,
    xt::xarray<std::complex<float>> &subgrid);

xt::xarray<std::complex<float>> compute_phasor(int subgrid_size);

void add_subgrid_to_grid(int s, const std::vector<Metadata> &metadata,
                         const xt::xarray<std::complex<float>> &subgrids,
                         xt::xarray<std::complex<float>> &grid,
                         const xt::xarray<std::complex<float>> &phasor,
                         size_t nr_correlations, size_t subgrid_size,
                         size_t grid_size);

void visibilities_to_subgrid(
    size_t s, const std::vector<Metadata> &metadata, float w_step,
    int grid_size, float image_size, const xt::xarray<float> &wavenumbers,
    const xt::xarray<std::complex<float>> &visibilities,
    const xt::xarray<UVW> &uvw, const xt::xarray<float> &taper,
    size_t nr_correlations_in, size_t subgrid_size,
    xt::xarray<std::complex<float>> &subgrid);
