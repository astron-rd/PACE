// Stokes.h: Complex Stokes vector.
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PREDICT_STOKES_H
#define PREDICT_STOKES_H

namespace predict {

/// \brief Complex Stokes vector.

/// @{

class Stokes {
public:
  Stokes();
  Stokes(const double I, const double Q, const double U, const double V)
      : I(I), Q(Q), U(U), V(V) {}

  double I, Q, U, V;
};

/// @}

} // namespace predict

#endif // PREDICT_STOKES_H
