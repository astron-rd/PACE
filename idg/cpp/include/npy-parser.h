#include "idgtypes.h"
#include <xtensor/containers/xarray.hpp>
// Complete reimplementation of the reading parts of xnpy.hpp that supports our
// custom types. It is unfortunately necessary to do this because their code is
// not suitable for reuse.

// Derived from
// https://github.com/xtensor-stack/xtensor/blob/master/include/xtensor/io/xnpy.hpp

namespace npy_parser {
xt::xarray<UVW> load_npy_uvw(const std::string &filename);
xt::xarray<Metadata> load_npy_metadata(const std::string &filename);
inline void read_magic(std::istream &istream, unsigned char *v_major,
                       unsigned char *v_minor);
inline std::string read_header_1_0(std::istream &istream);
inline std::string read_header_2_0(std::istream &istream);
inline void parse_header(std::string header, std::string &descr,
                         bool *fortran_order, std::vector<std::size_t> &shape);
inline std::string unwrap_s(std::string s, char delim_front, char delim_back);
inline std::string get_value_from_map(std::string mapstr);
inline void pop_char(std::string &s, char c);

} // namespace npy_parser
