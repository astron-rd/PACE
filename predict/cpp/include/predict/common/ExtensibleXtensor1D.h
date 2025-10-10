// ExtensibleXtensor.h: an xtensor that could be dynamically expanded
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief an xtensor that could be dynamically expanded

#ifndef EXTENSIBLE_XTENSOR1D_H_
#define EXTENSIBLE_XTENSOR1D_H_

#include <xtensor/xlayout.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

template <typename T> class ExtensibleXtensor1D : public xt::xtensor<T, 1> {
public:
  // Constructor
  ExtensibleXtensor1D() : xt::xtensor<T, 1>(), n_elements_{0} {}

  ExtensibleXtensor1D(const xt::nested_initializer_list_t<T, 1> &values)
      : xt::xtensor<T, 1>(values), n_elements_{values.size()} {}
  // Destructor
  ~ExtensibleXtensor1D() {}

  void expand(size_t new_size) {
    if (NeedsResizing(new_size)) {
      ResizeAndCopy(new_size);
    }
    n_elements_ = new_size;
  }

  void push_back(const T &value) {
    if (NeedsResizing(size() + 1)) {
      ResizeAndCopy(size() + 1);
    }
    this->at(++n_elements_ - 1) = value;
  }

  void resize(size_t new_size) {
    xt::xtensor<T, 1>::resize({new_size});
    n_elements_ = new_size;
  }

  void reserve(size_t new_size) {
    if (NeedsResizing(new_size)) {
      ResizeAndCopy(new_size);
    }
  }

  void clear() {
    xt::xtensor<T, 1>::resize({0});
    n_elements_ = 0;
  }

  size_t size() const { return xt::xtensor<T, 1>::size(); }

  size_t max_size() const { return n_elements_; }

private:
  bool NeedsResizing(size_t expected_size) const {
    if (expected_size > size() || size() < 0) {
      return true;
    }
    return false;
  }

  void ResizeAndCopy(size_t new_size) {
    xt::xtensor<T, 1> tmp(*this);

    xt::xtensor<T, 1>::resize({new_size});
    std::copy(tmp.data(), tmp.data() + tmp.size(), xt::xtensor<T, 1>::data());
  }

  size_t n_elements_{0};
};

#endif // EXTENSIBLE_XTENSOR1D_H_
