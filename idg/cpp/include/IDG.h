#include <complex>
#include <cstddef>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include "idgtypes.h"

class Gridder {
public:
  Gridder(size_t nr_correlations_in, size_t subgrid_size)
      : nr_correlations_in_(nr_correlations_in),
        nr_correlations_out_(nr_correlations_in == 4 ? 4 : 1),
        subgrid_size_(subgrid_size) {}

  void grid_onto_subgrids(float w_step, float image_size, size_t grid_size,
                          const xt::xarray<float> &wavenumbers,
                          const xt::xarray<UVW> &uvw,
                          const xt::xarray<std::complex<float>> &visibilities,
                          const xt::xarray<float> &taper,
                          const xt::xarray<Metadata> &metadata,
                          xt::xarray<std::complex<float>> &subgrids) const;

  void add_subgrids_to_grid(const xt::xarray<Metadata> &metadata,
                            const xt::xarray<std::complex<float>> &subgrids,
                            xt::xarray<std::complex<float>> &grid) const;

  void transform(int direction, xt::xarray<std::complex<float>> &grid) const;

private:
  size_t nr_correlations_in_;
  size_t nr_correlations_out_;
  size_t subgrid_size_;
};