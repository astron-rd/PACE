#include <vector>

#include <xtensor/containers/xarray.hpp>

#include "idgtypes.h"

xt::xarray<UVW> get_uvw(const float observation_hours,
                        const size_t nr_baselines, const size_t grid_size,
                        const float ellipticity = 0.1f, const int seed = 2);

xt::xarray<float> get_frequencies(const float start_frequency,
                                  const float frequency_increment,
                                  const size_t nr_channels);

std::vector<Metadata> get_metadata(const size_t nr_channels,
                                   const size_t subgrid_size,
                                   const size_t grid_size,
                                   const xt::xarray<float> &uvw,
                                   const size_t max_group_size = 256);

xt::xarray<VisibilityType>
get_visibilities(const size_t nr_correlations, const size_t nr_channels,
                 const size_t nr_timesteps, const size_t nr_baselines,
                 const float image_size, const size_t grid_size,
                 const xt::xarray<double> &frequencies,
                 const xt::xarray<float> &uvw,
                 const size_t nr_point_sources = 4, int max_pixel_offset = -1,
                 const int random_seed = 2);

xt::xarray<float> get_taper(const size_t subgrid_size);