#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include <cassert>

template <typename T>
void ReadScalarColumn(const std::string &path, const std::string &col_name,
                      std::vector<T> &data, size_t n_rows = 0) {
  casacore::Table tab(path);
  casacore::ScalarColumn<T> column(tab, col_name);

  if (n_rows == 0)
    n_rows = column.nrow();

  data.resize(n_rows);
  for (size_t row_id = 0; row_id < n_rows; row_id++) {
    data[row_id] = column.get(row_id);
  }
}

template <typename T, size_t R>
void ReadArrayColumn(const std::string &path, std::string col_name,
                     xt::xtensor<T, R> &data) {
  casacore::Table tab(path);
  casacore::ArrayColumn<T> column(tab, col_name);
  size_t n_rows = column.nrow();
  auto col = column.getColumn();
  auto pos = column.shape(0);
  auto shape = pos.asStdVector();
  shape.emplace(shape.begin(), n_rows);
  assert(shape.size() == data.shape().size());

  data.resize(shape);

  auto source_begin = col.data();
  auto source_end = col.data() + col.size();

  std::copy(source_begin, source_end, data.data());
}

std::vector<double> ReadUniqueTimes(const std::string &ms_path) {
  std::vector<double> t_column, unique_times;
  ReadScalarColumn(ms_path, "TIME", t_column);
  std::sort(t_column.begin(), t_column.end());
  unique_times.resize(t_column.size());
  auto it =
      std::unique_copy(t_column.begin(), t_column.end(), unique_times.begin());
  unique_times.erase(it, unique_times.end());
  return unique_times;
}