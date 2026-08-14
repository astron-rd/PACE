#include <vector>

#include <xtensor/containers/xarray.hpp>

#include "idgtypes.h"
#include "settings.h"

xt::xarray<float> get_taper(const size_t subgrid_size);

Inputs load_inputs(Settings settings,
                   std::vector<std::pair<const std::string, double>> &timings);
