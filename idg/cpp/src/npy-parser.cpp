#include "npy-parser.h"
#include "idgtypes.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <sys/types.h>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xbuffer_adaptor.hpp>
#include <xtensor/core/xlayout.hpp>
#include <xtensor/core/xstrides.hpp>
#include <xtensor/core/xtensor_forward.hpp>

xt::xarray<UVW> npy_parser::load_npy_uvw(const std::string &filename) {
  std::ifstream stream(filename, std::ifstream::binary);
  if (!stream) {
    XTENSOR_THROW(std::runtime_error, "io error: failed to open a file.");
  }

  unsigned char v_major, v_minor;
  read_magic(stream, &v_major, &v_minor);

  std::string header;

  if (v_major == 1 && v_minor == 0) {
    header = read_header_1_0(stream);
  } else if (v_major == 2 && v_minor == 0) {
    header = read_header_2_0(stream);
  } else {
    XTENSOR_THROW(std::runtime_error, "unsupported file format version.");
  }

  bool fortran_order;
  std::string typestr;
  std::vector<std::size_t> shape;

  parse_header(header, typestr, &fortran_order, shape);

  if (typestr != "[('u','<f4'),('v','<f4'),('w','<f4')]") {
    XTENSOR_THROW(std::runtime_error,
                  "type does not match expected UVW layout.");
  }

  constexpr size_t item_size = 3 * 4;
  size_t item_count = xt::compute_size(shape);
  size_t buffer_size = item_count * item_size;
  std::vector<size_t> strides(shape.size());
  xt::compute_strides(shape,
                      fortran_order ? xt::layout_type::column_major
                                    : xt::layout_type::row_major,
                      strides);

  char *buf = std::allocator<char>{}.allocate(buffer_size);
  if (buf == nullptr) {
    XTENSOR_THROW(std::runtime_error, "Allocation failed.");
  }

  stream.read(buf, std::streamsize(buffer_size));

  auto uvws = reinterpret_cast<UVW *>(buf);

  xt::xarray<UVW> output = xt::adapt(uvws, item_count, xt::no_ownership(),
                                     std::move(shape), std::move(strides));

  return output;
}

xt::xarray<Metadata>
npy_parser::load_npy_metadata(const std::string &filename) {
  std::ifstream stream(filename, std::ifstream::binary);
  if (!stream) {
    XTENSOR_THROW(std::runtime_error, "io error: failed to open a file.");
  }

  unsigned char v_major, v_minor;
  read_magic(stream, &v_major, &v_minor);

  std::string header;

  if (v_major == 1 && v_minor == 0) {
    header = read_header_1_0(stream);
  } else if (v_major == 2 && v_minor == 0) {
    header = read_header_2_0(stream);
  } else {
    XTENSOR_THROW(std::runtime_error, "unsupported file format version.");
  }

  bool fortran_order;
  std::string typestr;
  std::vector<std::size_t> shape;

  parse_header(header, typestr, &fortran_order, shape);

  if (typestr !=
      "[('baseline','<i4'),('time_index','<i4'),('nr_timesteps','<i4'),"
      "('channel_begin','<i4'),('channel_end','<i4'),('coordinate',[('x',"
      "'<i4'),('y','<i4'),('z','<i4')])]") {
    XTENSOR_THROW(std::runtime_error,
                  "type does not match expected Metadata layout.");
  }

  constexpr size_t item_size = 8 * 4;
  size_t item_count = xt::compute_size(shape);
  size_t buffer_size = item_count * item_size;
  std::vector<size_t> strides(shape.size());
  xt::compute_strides(shape,
                      fortran_order ? xt::layout_type::column_major
                                    : xt::layout_type::row_major,
                      strides);

  char *buf = std::allocator<char>{}.allocate(buffer_size);
  if (buf == nullptr) {
    XTENSOR_THROW(std::runtime_error, "Allocation failed.");
  }

  stream.read(buf, std::streamsize(buffer_size));

  auto uvws = reinterpret_cast<Metadata *>(buf);

  xt::xarray<Metadata> output = xt::adapt(uvws, item_count, xt::no_ownership(),
                                          std::move(shape), std::move(strides));

  return output;
}

const char magic_string[] = "\x93NUMPY";
const std::size_t magic_string_length = sizeof(magic_string) - 1;

inline void npy_parser::read_magic(std::istream &istream,
                                   unsigned char *v_major,
                                   unsigned char *v_minor) {
  std::unique_ptr<char[]> buf(new char[magic_string_length + 2]);
  istream.read(buf.get(), magic_string_length + 2);

  if (!istream) {
    XTENSOR_THROW(std::runtime_error, "io error: failed reading file");
  }

  for (std::size_t i = 0; i < magic_string_length; i++) {
    if (buf[i] != magic_string[i]) {
      XTENSOR_THROW(std::runtime_error,
                    "this file do not have a valid npy format.");
    }
  }

  *v_major = static_cast<unsigned char>(buf[magic_string_length]);
  *v_minor = static_cast<unsigned char>(buf[magic_string_length + 1]);
}

inline std::string npy_parser::read_header_1_0(std::istream &istream) {
  // read header length and convert from little endian
  char header_len_le16[2];
  istream.read(header_len_le16, 2);

  uint16_t header_b0 =
      static_cast<uint16_t>(static_cast<uint8_t>(header_len_le16[0]));
  uint16_t header_b1 =
      static_cast<uint16_t>(static_cast<uint8_t>(header_len_le16[1])) << 8;
  uint16_t header_length = header_b0 | header_b1;

  if ((magic_string_length + 2 + 2 + header_length) % 16 != 0) {
    // TODO: display warning
  }

  std::unique_ptr<char[]> buf(new char[header_length]);
  istream.read(buf.get(), header_length);
  std::string header(buf.get(), header_length);

  return header;
}

inline std::string npy_parser::read_header_2_0(std::istream &istream) {
  // read header length and convert from little endian
  char header_len_le32[4];
  istream.read(header_len_le32, 4);

  uint32_t header_b0 =
      static_cast<uint32_t>(static_cast<uint8_t>(header_len_le32[0]));
  uint32_t header_b1 =
      static_cast<uint32_t>(static_cast<uint8_t>(header_len_le32[1])) << 8;
  uint32_t header_b2 =
      static_cast<uint32_t>(static_cast<uint8_t>(header_len_le32[2])) << 16;
  uint32_t header_b3 =
      static_cast<uint32_t>(static_cast<uint8_t>(header_len_le32[3])) << 24;
  uint32_t header_length = header_b0 | header_b1 | header_b2 | header_b3;

  if ((magic_string_length + 2 + 4 + header_length) % 16 != 0) {
    // TODO: display warning
  }

  std::unique_ptr<char[]> buf(new char[header_length]);
  istream.read(buf.get(), header_length);
  std::string header(buf.get(), header_length);

  return header;
}

inline void npy_parser::parse_header(std::string header, std::string &descr,
                                     bool *fortran_order,
                                     std::vector<std::size_t> &shape) {
  // The first 6 bytes are a magic string: exactly "x93NUMPY".
  //
  // The next 1 byte is an unsigned byte: the major version number of the file
  // format, e.g. x01.
  //
  // The next 1 byte is an unsigned byte: the minor version number of the file
  // format, e.g. x00. Note: the version of the file format is not tied to the
  // version of the NumPy package.
  //
  // The next 2 bytes form a little-endian unsigned short int: the length of the
  // header data HEADER_LEN.
  //
  // The next HEADER_LEN bytes form the header data describing the array's
  // format. It is an ASCII string which contains a Python literal expression of
  // a dictionary. It is terminated by a newline ('n') and padded with spaces
  // ('x20') to make the total length of the magic string + 4 + HEADER_LEN be
  // evenly divisible by 16 for alignment purposes.
  //
  // The dictionary contains three keys:
  //
  // "descr" : dtype.descr
  // An object that can be passed as an argument to the numpy.dtype()
  // constructor to create the array's dtype.
  // "fortran_order" : bool
  // Whether the array data is Fortran-contiguous or not. Since
  // Fortran-contiguous arrays are a common form of non-C-contiguity, we allow
  // them to be written directly to disk for efficiency.
  // "shape" : tuple of int
  // The shape of the array.
  // For repeatability and readability, this dictionary is formatted using
  // pprint.pformat() so the keys are in alphabetic order.

  // remove trailing newline
  if (header.back() != '\n') {
    XTENSOR_THROW(std::runtime_error, "invalid header");
  }
  header.pop_back();

  // remove all whitespaces
  header.erase(std::remove(header.begin(), header.end(), ' '), header.end());

  // unwrap dictionary
  header = unwrap_s(header, '{', '}');

  // find the positions of the 3 dictionary keys
  std::size_t keypos_descr = header.find("'descr'");
  std::size_t keypos_fortran = header.find("'fortran_order'");
  std::size_t keypos_shape = header.find("'shape'");

  // make sure all the keys are present
  if (keypos_descr == std::string::npos) {
    XTENSOR_THROW(std::runtime_error, "missing 'descr' key");
  }
  if (keypos_fortran == std::string::npos) {
    XTENSOR_THROW(std::runtime_error, "missing 'fortran_order' key");
  }
  if (keypos_shape == std::string::npos) {
    XTENSOR_THROW(std::runtime_error, "missing 'shape' key");
  }

  // Make sure the keys are in order.
  // Note that this violates the standard, which states that readers *must* not
  // depend on the correct order here.
  // TODO: fix
  if (keypos_descr >= keypos_fortran || keypos_fortran >= keypos_shape) {
    XTENSOR_THROW(std::runtime_error, "header keys in wrong order");
  }

  // get the 3 key-value pairs
  std::string keyvalue_descr;
  keyvalue_descr = header.substr(keypos_descr, keypos_fortran - keypos_descr);
  pop_char(keyvalue_descr, ',');

  std::string keyvalue_fortran;
  keyvalue_fortran =
      header.substr(keypos_fortran, keypos_shape - keypos_fortran);
  pop_char(keyvalue_fortran, ',');

  std::string keyvalue_shape;
  keyvalue_shape = header.substr(keypos_shape, std::string::npos);
  pop_char(keyvalue_shape, ',');

  // get the values (right side of `:')
  descr = get_value_from_map(keyvalue_descr);
  std::string fortran_s = get_value_from_map(keyvalue_fortran);
  std::string shape_s = get_value_from_map(keyvalue_shape);

  // convert literal Python bool to C++ bool
  if (fortran_s == "True") {
    *fortran_order = true;
  } else if (fortran_s == "False") {
    *fortran_order = false;
  } else {
    XTENSOR_THROW(std::runtime_error, "invalid fortran_order value");
  }

  // parse the shape Python tuple ( x, y, z,)

  // first clear the vector
  shape.clear();
  shape_s = unwrap_s(shape_s, '(', ')');

  // a tokenizer would be nice...
  std::size_t pos = 0;
  for (;;) {
    std::size_t pos_next = shape_s.find_first_of(',', pos);
    std::string dim_s;

    if (pos_next != std::string::npos) {
      dim_s = shape_s.substr(pos, pos_next - pos);
    } else {
      dim_s = shape_s.substr(pos);
    }

    if (dim_s.length() == 0) {
      if (pos_next != std::string::npos) {
        XTENSOR_THROW(std::runtime_error, "invalid shape");
      }
    } else {
      std::stringstream ss;
      ss << dim_s;
      std::size_t tmp;
      ss >> tmp;
      shape.push_back(tmp);
    }

    if (pos_next != std::string::npos) {
      pos = ++pos_next;
    } else {
      break;
    }
  }
}

inline std::string npy_parser::unwrap_s(std::string s, char delim_front,
                                        char delim_back) {
  if ((s.back() == delim_back) && (s.front() == delim_front)) {
    return s.substr(1, s.length() - 2);
  } else {
    XTENSOR_THROW(std::runtime_error, "unable to unwrap");
  }
}

inline std::string npy_parser::get_value_from_map(std::string mapstr) {
  std::size_t sep_pos = mapstr.find_first_of(":");
  if (sep_pos == std::string::npos) {
    return "";
  }

  return mapstr.substr(sep_pos + 1);
}

inline void npy_parser::pop_char(std::string &s, char c) {
  if (s.back() == c) {
    s.pop_back();
  }
}
