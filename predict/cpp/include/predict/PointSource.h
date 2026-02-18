// PointSource.h: Point source model component with optional spectral index and
// rotation measure.
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PREDICT_POINTSOURCE_H
#define PREDICT_POINTSOURCE_H

#include "Direction.h"
#include "Spectrum.h"
#include "Stokes.h"
#include <cmath>
#include <memory>
#include <set>
namespace predict {

/// \brief Point source model component with optional spectral index and
/// rotation measure.

/// @{

class PointSource {
public:
  typedef std::shared_ptr<PointSource> Ptr;
  typedef std::shared_ptr<const PointSource> ConstPtr;

  PointSource(const Direction &direction, const Stokes &stokes,
              const size_t beam_id = 0);
  PointSource(const Direction &direction, const Spectrum &stokes,
              const size_t beam_id = 0);

  const Direction &GetDirection() const { return direction_; }
  void SetDirection(const Direction &position);

  void ComputeSpectrum(const xt::xtensor<double, 1> &frequencies,
                       xt::xtensor<double, 3> &result) const;
  const Spectrum &GetSpectrum() const { return spectrum_; }

  Stokes GetStokes(double freq) const;
  const Stokes &GetStokes() const { return spectrum_.GetReferenceFlux(); }

  const size_t &GetBeamId() const { return beam_id_; }
  void SetBeamId(size_t beam_id) { beam_id_ = beam_id; }

  template <typename T>
  void SetSpectralTerms(double refFreq, bool isLogarithmic, T first, T last);

  void SetRotationMeasure(double fraction, double angle, double rm);

  bool HasRotationMeasure() const;
  bool HasSpectralTerms() const;

protected:
  Direction direction_;
  Spectrum spectrum_;
  size_t beam_id_;
};

/// @}

template <typename T>
void PointSource::SetSpectralTerms(double refFreq, bool isLogarithmic, T first,
                                   T last) {
  spectrum_.SetSpectralTerms(refFreq, isLogarithmic,
                             xt::adapt(std::vector<double>(first, last)));
};
} // namespace predict

#endif // PREDICT_POINTSOURCE_H
