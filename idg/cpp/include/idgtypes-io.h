#pragma once

#include "idgtypes.h"
#include "npy-parser.h"
#include <string_view>
#include <xtensor/containers/xarray.hpp>

namespace idgtypes_io {
inline constexpr std::string_view uvw_dtype_descr =
    "[('u','<f4'),('v','<f4'),('w','<f4')]";
inline constexpr std::string_view metadata_dtype_descr =
    "[('baseline','<i4'),('time_index','<i4'),('nr_timesteps','<i4'),"
    "('channel_begin','<i4'),('channel_end','<i4'),('coordinate',[('x',"
    "'<i4'),('y','<i4'),('z','<i4')])]";
} // namespace idgtypes_io

namespace npy_parser {
inline xt::xarray<UVW> load_npy_uvw(const std::string &filename) {
  return load_npy_array<UVW>(filename, idgtypes_io::uvw_dtype_descr);
}

inline xt::xarray<Metadata> load_npy_metadata(const std::string &filename) {
  return load_npy_array<Metadata>(filename, idgtypes_io::metadata_dtype_descr);
}
} // namespace npy_parser

namespace xt {
inline void dump_npy(const std::string &filename, const xt::xarray<UVW> &arr) {
  npy_parser::dump_npy_array(filename, arr, idgtypes_io::uvw_dtype_descr);
}

inline void dump_npy(const std::string &filename,
                     const xt::xarray<Metadata> &arr) {
  npy_parser::dump_npy_array(filename, arr, idgtypes_io::metadata_dtype_descr);
}

} // namespace xt
