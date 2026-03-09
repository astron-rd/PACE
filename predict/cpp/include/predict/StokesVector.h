// StokesVector.h: A collection of stokes values
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief A collection of stokes values

#ifndef STOKES_VECTOR_H_
#define STOKES_VECTOR_H_

#include <span>

#include "ObjectCollection.h"
#include "common/SmartVector.h"

#include "Stokes.h"
namespace predict {

class StokesVector : ObjectCollection<Stokes> {
public:
  SmartVector<double> I;
  SmartVector<double> Q;
  SmartVector<double> U;
  SmartVector<double> V;
  StokesVector() {}
  void Add(const Stokes &stokes) {
    I.push_back(stokes.I);
    Q.push_back(stokes.Q);
    U.push_back(stokes.U);
    V.push_back(stokes.V);
  }

  void Add(const std::span<Stokes, std::dynamic_extent> &stokes) {
    const size_t expectedSize = Size() + stokes.size();
    I.reserve(expectedSize);
    Q.reserve(expectedSize);
    U.reserve(expectedSize);
    V.reserve(expectedSize);

    for (size_t i = 0; i < stokes.size(); ++i) {
      I.push_back(stokes[i].I);
      Q.push_back(stokes[i].Q);
      U.push_back(stokes[i].U);
      V.push_back(stokes[i].V);
    }
  }

  void Reserve(size_t new_size) {
    I.reserve(new_size);
    Q.reserve(new_size);
    U.reserve(new_size);
    V.reserve(new_size);
  }

  void Clear() {
    I.clear();
    Q.clear();
    U.clear();
    V.clear();
  }

  size_t Size() const { return I.size(); }
};

} // namespace predict

#endif // STOKES_VECTOR_H_
