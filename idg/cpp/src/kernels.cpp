#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xshape.hpp>
#include <xtensor/views/xindex_view.hpp>
#include <xtensor/views/xview.hpp>

#include "idgtypes.h"

void add_pt_src_to_baseline(size_t bl, size_t nr_timesteps, size_t nr_channels,
                            std::complex<float> amplitude,
                            const xt::xarray<float> &frequencies,
                            const xt::xarray<UVW> &uvw, float l, float m,
                            xt::xarray<std::complex<float>> &visibilities) {
  const float speed_of_light = 299792458.0f;

  for (int t = 0; t < nr_timesteps; ++t) {
    for (int c = 0; c < nr_channels; ++c) {
      float u = (frequencies(c) / speed_of_light) * uvw(bl, t).u;
      float v = (frequencies(c) / speed_of_light) * uvw(bl, t).v;

      float phase = -2.0f * M_PI * (u * l + v * m);
      std::complex<float> value =
          amplitude * std::exp(std::complex<float>(0.0f, phase));

      // Add to all correlations
      for (int pol = 0; pol < visibilities.shape(3); ++pol) {
        visibilities(bl, t, c, pol) += value;
      }
    }
  }
}

// Polynomial evaluation (equivalent to np.polyval)
xt::xarray<float> polyval(const xt::xarray<float> &coefficients,
                          const xt::xarray<float> &x) {
  xt::xarray<float> result = xt::zeros_like(x);

  for (size_t i = 0; i < x.size(); ++i) {
    float val = coefficients(0);
    for (size_t j = 1; j < coefficients.size(); ++j) {
      val = val * x(i) + coefficients(j);
    }
    result(i) = val;
  }
  return result;
}

xt::xarray<float> evaluate_spheroidal(const xt::xarray<float> &nu) {
  xt::xarray<float> P = {
      {8.203343e-2f, -3.644705e-1f, 6.278660e-1f, -5.335581e-1f, 2.312756e-1f},
      {4.028559e-3f, -3.697768e-2f, 1.021332e-1f, -1.201436e-1f, 6.412774e-2f}};

  xt::xarray<float> Q = {{1.0000000e0f, 8.212018e-1f, 2.078043e-1f},
                         {1.0000000e0f, 9.599102e-1f, 2.918724e-1f}};

  xt::xarray<float> result = xt::zeros_like(nu);

  // Process each part
  for (size_t part = 0; part < 2; ++part) {
    float end = (part == 0) ? 0.75f : 1.00f;
    float start = (part == 0) ? 0.0f : 0.75f;

    // Create mask using XTensor's element-wise operations
    xt::xarray<bool> mask = (nu >= start) && (nu <= end);

    // Check if any elements in mask are true
    if (xt::sum(mask)() == 0) {
      continue;
    }

    // Use xt::filter to extract elements
    xt::xarray<float> nu_part = xt::filter(nu, mask);
    xt::xarray<float> nusq = xt::square(nu_part);
    xt::xarray<float> delnusq = nusq - end * end;

    // Reverse coefficients for polyval
    xt::xarray<float> P_part = xt::view(P, part, xt::all());
    xt::xarray<float> Q_part = xt::view(Q, part, xt::all());
    xt::xarray<float> P_reversed = xt::flip(P_part, 0);
    xt::xarray<float> Q_reversed = xt::flip(Q_part, 0);

    // Calculate polynomial using Horner's method
    xt::xarray<float> top = polyval(P_reversed, delnusq);
    xt::xarray<float> bot = polyval(Q_reversed, delnusq);

    // Create valid mask using element-wise comparison
    xt::xarray<bool> valid = xt::not_equal(bot, 0.0f);

    // Calculate result_part
    xt::xarray<float> result_part = xt::zeros_like(nu_part);

    // Use xt::filter to apply the valid mask
    xt::xarray<float> valid_top = xt::filter(top, valid);
    xt::xarray<float> valid_bot = xt::filter(bot, valid);
    xt::xarray<float> valid_nusq = xt::filter(nusq, valid);

    xt::xarray<float> valid_result =
        (1.0f - valid_nusq) * (valid_top / valid_bot);

    // Assign the valid results back to result_part
    xt::xarray<bool> valid_indices = xt::arange(nu_part.size());
    valid_indices = xt::filter(valid_indices, valid);

    for (size_t i = 0; i < valid_indices.size(); ++i) {
      const size_t idx = valid_indices(i);
      result_part(idx) = valid_result(i);
    }

    // Assign result_part back to the original result array
    xt::xarray<bool> mask_indices = xt::arange(nu.size());
    mask_indices = xt::filter(mask_indices, mask);

    for (size_t i = 0; i < mask_indices.size(); ++i) {
      const size_t idx = mask_indices(i);
      result(idx) = result_part(i);
    }
  }

  return result;
}

xt::xarray<std::complex<float>> compute_phasor(size_t subgrid_size) {
  xt::xarray<std::complex<float>> phasor =
      xt::zeros<std::complex<float>>({subgrid_size, subgrid_size});

  for (size_t y = 0; y < subgrid_size; ++y) {
    for (size_t x = 0; x < subgrid_size; ++x) {
      const float phase = M_PI * (x + y - subgrid_size) / subgrid_size;
      phasor(y, x) = std::exp(std::complex<float>(0.0f, phase));
    }
  }

  return phasor;
}

void add_subgrid_to_grid(size_t s, const std::vector<Metadata> &metadata,
                         const xt::xarray<std::complex<float>> &subgrids,
                         xt::xarray<std::complex<float>> &grid,
                         const xt::xarray<std::complex<float>> &phasor,
                         size_t nr_correlations, size_t subgrid_size,
                         size_t grid_size) {
  const Metadata &m = metadata[s];
  const size_t grid_x = m.coordinate.x;
  const size_t grid_y = m.coordinate.y;

  if (grid_x >= 0 && grid_x <= grid_size - subgrid_size && grid_y >= 0 &&
      grid_y <= grid_size - subgrid_size) {

    for (size_t y = 0; y < subgrid_size; ++y) {
      for (size_t x = 0; x < subgrid_size; ++x) {
        const size_t x_src = (x + subgrid_size / 2) % subgrid_size;
        const size_t y_src = (y + subgrid_size / 2) % subgrid_size;

        for (size_t p = 0; p < nr_correlations; ++p) {
          grid(p, grid_y + y, grid_x + x) +=
              subgrids(s, p, y_src, x_src) * phasor(y, x);
        }
      }
    }
  }
}

std::vector<Metadata> compute_metadata(size_t grid_size, size_t subgrid_size,
                                       size_t nr_channels, size_t bl,
                                       const xt::xarray<float> &u_pixels,
                                       const xt::xarray<float> &v_pixels,
                                       size_t max_group_size) {
  std::vector<Metadata> metadata;
  const size_t nr_timesteps = u_pixels.size();
  const float max_distance = 0.8f * subgrid_size;

  size_t timestep = 0;
  while (timestep < nr_timesteps) {
    const float current_u = u_pixels(timestep);
    const float current_v = v_pixels(timestep);

    size_t group_size = 1;
    while (timestep + group_size < nr_timesteps &&
           group_size < max_group_size) {
      const float u_diff = u_pixels(timestep + group_size) - current_u;
      const float v_diff = v_pixels(timestep + group_size) - current_v;
      const float distance = std::sqrt(u_diff * u_diff + v_diff * v_diff);

      if (distance > max_distance) {
        break;
      }
      group_size++;
    }

    // Calculate group center
    float group_u = 0.0f;
    float group_v = 0.0f;
    for (size_t i = 0; i < group_size; ++i) {
      group_u += u_pixels(timestep + i);
      group_v += v_pixels(timestep + i);
    }
    group_u /= group_size;
    group_v /= group_size;

    size_t subgrid_x = group_u - subgrid_size / 2;
    size_t subgrid_y = group_v - subgrid_size / 2;
    subgrid_x = std::max(0ul, std::min(grid_size - subgrid_size, subgrid_x));
    subgrid_y = std::max(0ul, std::min(grid_size - subgrid_size, subgrid_y));

    const Metadata entry{
        static_cast<int>(bl),
        static_cast<int>(timestep),
        static_cast<int>(group_size),
        0,
        static_cast<int>(nr_channels),
        {static_cast<int>(subgrid_x), static_cast<int>(subgrid_y), 0}};

    metadata.push_back(entry);
    timestep += group_size;
  }

  return metadata;
}

void visibilities_to_subgrid(size_t s, const std::vector<Metadata> &metadata,
                             float w_step, size_t grid_size, float image_size,
                             const xt::xarray<float> &wavenumbers,
                             const xt::xarray<VisibilityType> &visibilities,
                             const xt::xarray<float> &uvw_u,
                             const xt::xarray<float> &uvw_v,
                             const xt::xarray<float> &uvw_w,
                             const xt::xarray<float> &taper,
                             size_t nr_correlations_in, size_t subgrid_size,
                             xt::xarray<std::complex<float>> &subgrid) {
  const Metadata &m = metadata[s];
  const int bl = m.baseline;
  const int offset = m.time_index;
  const int nr_timesteps = m.nr_timesteps;
  const int channel_begin = m.channel_begin;
  const int channel_end = m.channel_end;
  const int x_coordinate = m.coordinate.x;
  const int y_coordinate = m.coordinate.y;
  const float w_offset_in_lambda = w_step * (m.coordinate.z + 0.5f);

  const size_t nr_correlations_out = (nr_correlations_in == 4) ? 4 : 1;

  // Compute offsets
  const float u_offset =
      (x_coordinate + subgrid_size / 2.0f - grid_size / 2.0f) *
      (2.0f * M_PI / image_size);
  const float v_offset =
      (y_coordinate + subgrid_size / 2.0f - grid_size / 2.0f) *
      (2.0f * M_PI / image_size);
  const float w_offset = 2.0f * M_PI * w_offset_in_lambda;

  const int half_subgrid = subgrid_size / 2;
  const float image_scale = image_size / subgrid_size;

  for (size_t y = 0; y < subgrid_size; ++y) {
    for (size_t x = 0; x < subgrid_size; ++x) {
      // Compute l, m, n
      const float l = (x + 0.5f - half_subgrid) * image_scale;
      const float m_val = (y + 0.5f - half_subgrid) * image_scale;
      const float tmp = l * l + m_val * m_val;
      const float n = tmp / (1.0f + std::sqrt(1.0f - tmp));

      // Initialize pixels for each correlation
      std::vector<std::complex<float>> pixels(nr_correlations_out, 0.0f);

      for (size_t time = 0; time < nr_timesteps; ++time) {
        const size_t idx = offset + time;
        const float u = uvw_u(bl, idx);
        const float v = uvw_v(bl, idx);
        const float w = uvw_w(bl, idx);

        const float phase_index = u * l + v * m_val + w * n;
        const float phase_offset =
            u_offset * l + v_offset * m_val + w_offset * n;

        for (size_t chan = channel_begin; chan < channel_end; ++chan) {
          const float phase = phase_offset - (phase_index * wavenumbers(chan));
          const std::complex<float> phasor =
              std::exp(std::complex<float>(0.0f, phase));

          for (size_t pol = 0; pol < nr_correlations_in; ++pol) {
            const size_t pol_out = pol % nr_correlations_out;
            pixels[pol_out] += visibilities(bl, idx, chan, pol) * phasor;
          }
        }
      }

      // Apply taper and store
      const float sph = taper(y, x);
      const size_t x_dst = (x + half_subgrid) % subgrid_size;
      const size_t y_dst = (y + half_subgrid) % subgrid_size;

      for (size_t pol = 0; pol < nr_correlations_out; ++pol) {
        subgrid(pol, y_dst, x_dst) = pixels[pol] * sph;
      }
    }
  }
}

xt::xarray<std::complex<float>> compute_phasor(int subgrid_size) {
  xt::xarray<std::complex<float>> phasor =
      xt::zeros<std::complex<float>>({subgrid_size, subgrid_size});

  for (int y = 0; y < subgrid_size; ++y) {
    for (int x = 0; x < subgrid_size; ++x) {
      float phase = M_PI * (x + y - subgrid_size) / subgrid_size;
      phasor(y, x) = std::exp(std::complex<float>(0.0f, phase));
    }
  }

  return phasor;
}

void add_subgrid_to_grid(int s, const std::vector<Metadata> &metadata,
                         const xt::xarray<std::complex<float>> &subgrids,
                         xt::xarray<std::complex<float>> &grid,
                         const xt::xarray<std::complex<float>> &phasor,
                         int nr_correlations, int subgrid_size, int grid_size) {
  const Metadata &m = metadata[s];
  int grid_x = m.coordinate.x;
  int grid_y = m.coordinate.y;

  if (grid_x >= 0 && grid_x < grid_size - subgrid_size && grid_y >= 0 &&
      grid_y < grid_size - subgrid_size) {

    for (int y = 0; y < subgrid_size; ++y) {
      for (int x = 0; x < subgrid_size; ++x) {
        int x_src = (x + subgrid_size / 2) % subgrid_size;
        int y_src = (y + subgrid_size / 2) % subgrid_size;

        for (int p = 0; p < nr_correlations; ++p) {
          grid(p, grid_y + y, grid_x + x) +=
              subgrids(s, p, y_src, x_src) * phasor(y, x);
        }
      }
    }
  }
}

void visibilities_to_subgrid(
    size_t s, const std::vector<Metadata> &metadata, float w_step,
    int grid_size, float image_size, const xt::xarray<float> &wavenumbers,
    const xt::xarray<std::complex<float>> &visibilities,
    const xt::xarray<UVW> &uvw, const xt::xarray<float> &taper,
    size_t nr_correlations_in, size_t subgrid_size,
    xt::xarray<std::complex<float>> &subgrid) {
  const Metadata &m = metadata[s];
  const int bl = m.baseline;
  const int offset = m.time_index;
  const int nr_timesteps = m.nr_timesteps;
  const int channel_begin = m.channel_begin;
  const int channel_end = m.channel_end;
  const int x_coordinate = m.coordinate.x;
  const int y_coordinate = m.coordinate.y;
  const float w_offset_in_lambda = w_step * (m.coordinate.z + 0.5f);

  const size_t nr_correlations_out = (nr_correlations_in == 4) ? 4 : 1;

  // Compute offsets
  const float u_offset =
      (x_coordinate + subgrid_size / 2.0f - grid_size / 2.0f) *
      (2.0f * M_PI / image_size);
  const float v_offset =
      (y_coordinate + subgrid_size / 2.0f - grid_size / 2.0f) *
      (2.0f * M_PI / image_size);
  const float w_offset = 2.0f * M_PI * w_offset_in_lambda;

  const float half_subgrid = subgrid_size / 2.0f;
  const float image_scale = image_size / subgrid_size;

  for (size_t y = 0; y < subgrid_size; ++y) {
    for (size_t x = 0; x < subgrid_size; ++x) {
      // Compute l, m, n
      const float l = (x + 0.5f - half_subgrid) * image_scale;
      const float m_val = (y + 0.5f - half_subgrid) * image_scale;
      const float tmp = l * l + m_val * m_val;
      const float n = tmp / (1.0f + std::sqrt(1.0f - tmp));

      // Compute pixels
      std::vector<std::complex<float>> pixels(nr_correlations_out,
                                              std::complex<float>(0.0f, 0.0f));

      for (size_t time = 0; time < nr_timesteps; ++time) {
        const int idx = offset + time;
        const float u = uvw(bl, idx).u;
        const float v = uvw(bl, idx).v;
        const float w = uvw(bl, idx).w;

        const float phase_index = u * l + v * m_val + w * n;
        const float phase_offset =
            u_offset * l + v_offset * m_val + w_offset * n;

        for (size_t chan = channel_begin; chan < channel_end; ++chan) {
          const float phase = phase_offset - (phase_index * wavenumbers(chan));
          std::complex<float> phasor =
              std::exp(std::complex<float>(0.0f, phase));

          for (size_t pol = 0; pol < nr_correlations_in; ++pol) {
            int pol_out = pol % nr_correlations_out;
            pixels[pol_out] += visibilities(bl, idx, chan, pol) * phasor;
          }
        }
      }

      // Apply taper and store
      const float sph = taper(y, x);
      const int x_dst = static_cast<int>((x + half_subgrid)) % subgrid_size;
      const int y_dst = static_cast<int>((y + half_subgrid)) % subgrid_size;

      for (size_t pol = 0; pol < nr_correlations_out; ++pol) {
        subgrid(pol, y_dst, x_dst) = pixels[pol] * sph;
      }
    }
  }
}