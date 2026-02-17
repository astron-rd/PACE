#include <cmath>
#include <random>
#include <vector>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>

#include "idgtypes.h"
#include "kernels.h"

xt::xarray<UVW> get_uvw(const float observation_hours,
                        const size_t nr_baselines, const size_t grid_size,
                        const float ellipticity = 0.1f, const int seed = 2) {
  // Convert observation time to seconds (1 sample per second)
  const size_t observation_seconds =
      static_cast<size_t>(observation_hours * 3600.0f);
  const size_t nr_timesteps = observation_seconds;

  // Create time samples
  xt::xarray<float> time_samples = xt::linspace<float>(
      0.0f, static_cast<float>(observation_seconds), nr_timesteps);

  // Initialize uvw array
  xt::xarray<UVW> uvw = xt::zeros<UVW>({nr_baselines, nr_timesteps});

  // Calculate maximum UV distance
  const float max_u = 0.7f * (static_cast<float>(grid_size) / 2.0f);
  const float max_v = 0.7f * (static_cast<float>(grid_size) / 2.0f);

  // Random number generation setup
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

  // Beta distribution with alpha=1, beta=3 (peaks at 0 and decreases)
  xt::xarray<float> baseline_ratios = xt::zeros<float>({nr_baselines});
  float max_ratio = 0.0f;

  // Generate baseline ratios using inverse transform sampling for Beta(1,3)
  for (size_t i = 0; i < nr_baselines; ++i) {
    // Inverse CDF of Beta(1,3): F^{-1}(u) = 1 - (1 - u)^{1/3}
    const float u = uniform_dist(rng);
    baseline_ratios(i) = 1.0f - std::pow(1.0f - u, 1.0f / 3.0f);
    if (baseline_ratios(i) > max_ratio) {
      max_ratio = baseline_ratios(i);
    }
  }

  // Normalize ratio to have max of one
  baseline_ratios /= max_ratio;

  // Generate random starting angles for each baseline
  xt::xarray<float> start_angles = xt::zeros<float>({nr_baselines});
  for (size_t i = 0; i < nr_baselines; ++i) {
    start_angles(i) = 2.0f * M_PI * uniform_dist(rng);
  }

  // Calculate angular velocity (complete circle in 24 hours)
  const float angular_velocity = 2.0f * M_PI / (24.0f * 3600.0f); // rad/sec

  // Calculate the UV coordinates for each baseline
  for (size_t bl = 0; bl < nr_baselines; ++bl) {
    const float ratio = baseline_ratios(bl);

    // Calculate radius for this baseline
    float u_radius = ratio * max_u;
    float v_radius = ratio * max_v;

    // Apply ellipticity if specified
    if (ellipticity > 0.0f) {
      // Make the ellipse orientation depend on the baseline
      // Longer baselines have more ellipticity
      const float ellipse_factor = 1.0f + ellipticity * ratio;
      u_radius *= ellipse_factor;
      v_radius /= ellipse_factor;
    }

    // Generate UV coordinates with random starting angle
    const float start_angle = start_angles(bl);
    xt::xarray<float> angles = start_angle + angular_velocity * time_samples;

    xt::xarray<float> u_coords =
        u_radius * xt::cos(angles) + static_cast<float>(grid_size) / 2.0f;
    xt::xarray<float> v_coords =
        v_radius * xt::sin(angles) + static_cast<float>(grid_size) / 2.0f;

    for (size_t t = 0; t < nr_timesteps; ++t) {
      uvw(bl, t).u = u_coords(t);
      uvw(bl, t).v = v_coords(t);
      uvw(bl, t).w = 0.0f;
    }
  }

  return uvw;
}

xt::xarray<float> get_frequencies(const float start_frequency,
                                  const float frequency_increment,
                                  const size_t nr_channels) {
  const float end_frequency =
      start_frequency + nr_channels * frequency_increment;
  return xt::arange(start_frequency, end_frequency, frequency_increment);
}

xt::xarray<Metadata> get_metadata(const size_t nr_channels,
                                  const size_t subgrid_size,
                                  const size_t grid_size,
                                  const xt::xarray<UVW> &uvw,
                                  const size_t max_group_size = 256) {
  const size_t nr_baselines = uvw.shape()[0];

  std::vector<Metadata> metadata;

  for (size_t bl = 0; bl < nr_baselines; ++bl) {
    const auto bl_uvw = xt::view(uvw, bl, xt::all());

    const auto bl_metadata = compute_metadata(
        grid_size, subgrid_size, nr_channels, bl, bl_uvw, max_group_size);

    metadata.insert(metadata.end(), bl_metadata.begin(), bl_metadata.end());
  }

  std::vector<size_t> shape({metadata.size()});
  xt::xarray<Metadata> result(shape);
  std::copy(metadata.begin(), metadata.end(), result.begin());
  return result;
}

xt::xarray<VisibilityType>
get_visibilities(const size_t nr_correlations, const size_t nr_channels,
                 const size_t nr_timesteps, const size_t nr_baselines,
                 const float image_size, const size_t grid_size,
                 const xt::xarray<float> &frequencies,
                 const xt::xarray<UVW> &uvw, const size_t nr_point_sources = 4,
                 int max_pixel_offset = -1, const int random_seed = 2) {
  if (max_pixel_offset == -1) {
    max_pixel_offset = static_cast<int>(grid_size / 3);
  }

  // Initialize visibilities to zero
  xt::xarray<VisibilityType> visibilities = xt::zeros<VisibilityType>(
      {nr_baselines, nr_timesteps, nr_channels, nr_correlations});

  // Generate random offsets for point sources
  std::vector<std::pair<float, float>> offsets;
  std::mt19937 rng(random_seed);
  std::uniform_real_distribution<float> dist(-max_pixel_offset / 2.0f,
                                             max_pixel_offset / 2.0f);

  for (size_t i = 0; i < nr_point_sources; ++i) {
    offsets.emplace_back(dist(rng), dist(rng));
  }

  // Loop over point sources
  for (const auto &offset : offsets) {
    const float amplitude = 1.0f;

    // Convert offset from grid cells to radians (l,m coordinates)
    const float l = offset.first * image_size / grid_size;
    const float m = offset.second * image_size / grid_size;

// Parallelize over baselines using OpenMP
#pragma omp parallel for schedule(dynamic)
    for (size_t bl = 0; bl < nr_baselines; ++bl) {
      add_pt_src_to_baseline(bl, nr_timesteps, nr_channels, amplitude,
                             frequencies, uvw, l, m, visibilities);
    }
  }

  return visibilities;
}

xt::xarray<float> get_taper(const size_t subgrid_size) {
  // Generate linspace [-1, 1), subgrid_size samples
  xt::xarray<float> x = xt::linspace<float>(-1.0f, 1.0f, subgrid_size, false);

  // Take absolute value
  xt::xarray<float> abs_x = xt::abs(x);

  // Evaluate spheroidal function
  xt::xarray<float> x_spheroidal = evaluate_spheroidal(abs_x);

  // Construct 2D taper (outer product)
  xt::xarray<float> taper =
      xt::expand_dims(x_spheroidal, 0) * xt::expand_dims(x_spheroidal, 1);

  return taper;
}