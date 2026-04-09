#pragma once

#include <cstddef>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xlayout.hpp>
// Complete reimplementation of the reading parts of xnpy.hpp that supports our
// custom types. It is unfortunately necessary to do this because their code is
// not suitable for reuse.

// Derived from
// https://github.com/xtensor-stack/xtensor/blob/master/include/xtensor/io/xnpy.hpp

namespace npy_parser {
struct npy_header {
  std::string descr;
  bool fortran_order;
  std::vector<std::size_t> shape;
};

npy_header load_npy_header(const std::string &filename);
npy_header load_npy_header(std::istream &istream);
void write_npy_header(std::ostream &ostream, std::string_view descr,
                      const std::vector<std::size_t> &shape,
                      bool fortran_order);

template <typename T>
xt::xarray<T> load_npy_array(const std::string &filename,
                             std::string_view expected_descr) {
  std::ifstream stream(filename, std::ifstream::binary);
  if (!stream) {
    XTENSOR_THROW(std::runtime_error, "io error: failed to open a file.");
  }

  npy_header header = load_npy_header(stream);
  if (header.descr != expected_descr) {
    XTENSOR_THROW(std::runtime_error, "type does not match expected layout.");
  }

  const std::size_t item_count = xt::compute_size(header.shape);
  std::vector<T> buffer(item_count);
  stream.read(reinterpret_cast<char *>(buffer.data()),
              std::streamsize(item_count * sizeof(T)));
  if (!stream) {
    XTENSOR_THROW(std::runtime_error, "io error: failed reading array data.");
  }

  const auto layout = header.fortran_order ? xt::layout_type::column_major
                                           : xt::layout_type::row_major;
  xt::xarray<T> output = xt::adapt(std::move(buffer), header.shape, layout);
  return output;
}

template <typename T>
void dump_npy_array(const std::string &filename, const xt::xarray<T> &arr,
                    std::string_view descr) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  std::vector<std::size_t> shape(arr.shape().begin(), arr.shape().end());
  write_npy_header(file, descr, shape, false);
  file.write(reinterpret_cast<const char *>(arr.data()),
             std::streamsize(arr.size() * sizeof(T)));
}

inline xt::xarray<char> load_npy_raw(const std::string &filename,
                                     std::string_view expected_descr) {
  return load_npy_array<char>(filename, expected_descr);
}

inline void dump_npy_raw(const std::string &filename,
                         const xt::xarray<char> &arr, std::string_view descr) {
  dump_npy_array<char>(filename, arr, descr);
}

} // namespace npy_parser
