// SmartVector.h: A vector that can be accessed as an xtensor
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief A vector that can be accessed as an xtensor

#ifndef SMART_VECTOR_H_
#define SMART_VECTOR_H_

#include <memory>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

template <typename T>
using SmartVectorView =
    decltype(::xt::adapt(std::add_pointer_t<T>{}, std::array<size_t, 1UL>{}));
;

template <typename T>
using ConstSmartVectorView =
    decltype(::xt::adapt(std::declval<const T *>(), std::array<size_t, 1>{}));

template <typename T> class SmartVector : public std::vector<T> {
public:
  SmartVector() : std::vector<T>() {}
  SmartVector(const std::vector<T> &vec) : std::vector<T>(vec) {}
  SmartVector(const std::initializer_list<T> &vec) : std::vector<T>(vec) {}
  SmartVectorView<T> view() { return xt::adapt(this->data(), {this->size()}); }

  ConstSmartVectorView<T> view() const {
    return xt::adapt(this->data(), {this->size()});
  }

  // Returns a copy of the data in a new xtensor
  explicit operator std::unique_ptr<xt::xtensor<T, 1>>() const {
    std::unique_ptr<xt::xtensor<T, 1>> tensor =
        std::make_unique<xt::xtensor<T, 1>>(
            xt::adapt(this->data(), {this->size()}));

    return std::move(tensor);
  }

  // Return a view of the data in a new xtensor
  explicit operator SmartVectorView<T>() { return view(); }
};

#endif // SMART_VECTOR_H_
