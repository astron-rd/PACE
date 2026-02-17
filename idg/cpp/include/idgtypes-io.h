#pragma once

#include <fstream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xnpy.hpp>

namespace xt {
inline void dump_npy(const std::string &filename, const xt::xarray<UVW> &arr) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  // NPY magic
  char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  file.write(magic, 6);

  // Version 1.0
  char version[2] = {1, 0};
  file.write(version, 2);

  // Structured dtype
  std::string dtype = "[('u', '<f4'), ('v', '<f4'), ('w', '<f4')]";

  // Build shape string
  std::string shape_str = "(";
  for (size_t i = 0; i < arr.shape().size(); ++i) {
    if (i > 0)
      shape_str += ", ";
    shape_str += std::to_string(arr.shape()[i]);
  }
  shape_str += ",)";

  std::string header = "{'descr': " + dtype +
                       ", 'fortran_order': False, 'shape': " + shape_str +
                       ", }";

  // Pad to 64-byte boundary
  size_t header_len = header.size() + 1;
  size_t padding = (64 - ((10 + header_len) % 64)) % 64;
  header.append(padding, ' ');
  header += '\n';

  uint16_t header_len_le = static_cast<uint16_t>(header.size());
  file.write(reinterpret_cast<char *>(&header_len_le), 2);
  file.write(header.data(), header.size());

  file.write(reinterpret_cast<const char *>(arr.data()),
             arr.size() * sizeof(UVW));
}

inline void dump_npy(const std::string &filename,
                     const xt::xarray<Metadata> &arr) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  // NPY magic
  char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  file.write(magic, 6);

  // Version 1.0
  char version[2] = {1, 0};
  file.write(version, 2);

  // Structured dtype
  std::string dtype =
      "[('baseline', '<i4'), ('time_index', '<i4'), "
      "('nr_timesteps', '<i4'), ('channel_begin', '<i4'), "
      "('channel_end', '<i4'), "
      "('coordinate', [('x', '<i4'), ('y', '<i4'), ('z', '<i4')])]";

  std::string shape_str = "(" + std::to_string(arr.size()) + ",)";

  std::string header = "{'descr': " + dtype +
                       ", 'fortran_order': False, 'shape': " + shape_str +
                       ", }";

  // Pad to 64-byte boundary
  size_t header_len = header.size() + 1;
  size_t padding = (64 - ((10 + header_len) % 64)) % 64;
  header.append(padding, ' ');
  header += '\n';

  uint16_t header_len_le = static_cast<uint16_t>(header.size());
  file.write(reinterpret_cast<char *>(&header_len_le), 2);
  file.write(header.data(), header.size());

  // Write data
  file.write(reinterpret_cast<const char *>(arr.data()),
             arr.size() * sizeof(Metadata));
}

} // namespace xt