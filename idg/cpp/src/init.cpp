#include <vector>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>

#include "h5cpp/file/file.hpp"

#include "h5cpp/file/functions.hpp"
#include "h5cpp/node/dataset.hpp"
#include "h5cpp/node/group.hpp"
#include "idgtypes.h"
#include "kernels.h"
#include "settings.h"
#include "util.h"

using namespace hdf5;

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

template <typename T>
xt::xarray<T> load_dataset_to_xtensor(node::Dataset &dataset) {
  datatype::Datatype datatype = dataset.datatype();
  dataspace::Simple dataspace = dataset.dataspace();
  auto dim = dataspace.current_dimensions();

  xt::xarray<T> data = xt::xarray<T>::from_shape(dim);
  dataset.read(*data.data(), datatype, dataspace);
  return data;
}

Inputs load_inputs(Settings settings,
                   std::vector<std::pair<const std::string, double>> &timings) {
  print_header("LOADING INPUT DATA");

  file::File input_file = file::open(settings.input_path);
  node::Group root_node = input_file.root();

  // Load UVW coordinates
  node::Dataset uvws_ds = root_node.get_dataset("uvws");
  xt::xarray<UVW> uvws;
  time_function(timings, "load uvws", [&uvws, &uvws_ds]() {
    uvws = load_dataset_to_xtensor<UVW>(uvws_ds);
  });

  // Load frequencies
  node::Dataset frequencies_ds = root_node.get_dataset("frequencies");
  xt::xarray<float> frequencies;
  time_function(timings, "load frequencies", [&frequencies, &frequencies_ds]() {
    frequencies = load_dataset_to_xtensor<float>(frequencies_ds);
  });

  // Derive wavenumbers
  xt::xarray<float> wavenumbers;
  time_function(
      timings, "derive wavenumbers", [settings, &wavenumbers, frequencies]() {
        wavenumbers = (frequencies * 2.0 * M_PI) / settings.speed_of_light;
      });

  // Load metadata
  node::Dataset metadata_ds = root_node.get_dataset("metadata");
  xt::xarray<Metadata> metadata;
  time_function(timings, "load metadata", [&metadata, &metadata_ds]() {
    metadata = load_dataset_to_xtensor<Metadata>(metadata_ds);
  });
  const size_t nr_subgrids = metadata.size();

  // Load visibilities
  node::Dataset visibilities_ds = root_node.get_dataset("visibilities");
  xt::xarray<VisibilityType> visibilities;
  time_function(
      timings, "load visibilities", [&visibilities, &visibilities_ds]() {
        visibilities = load_dataset_to_xtensor<VisibilityType>(visibilities_ds);
      });

  // Generate taper
  xt::xarray<float> taper;
  time_function(timings, "generate taper", [settings, &taper]() {
    taper = get_taper(settings.subgrid_size);
  });

  return Inputs{
      uvws,
      frequencies,
      wavenumbers,
      metadata,
      visibilities,
      taper,
      nr_subgrids,
      static_cast<float>(settings.speed_of_light / frequencies.back())};
}
