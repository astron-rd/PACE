// PointSource.cc: Point source model component with optional spectral index
// and rotation measure.
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// $Id$

#include <predict/PointSource.h>

namespace predict {

PointSource::PointSource(const Direction &position, const Stokes &stokes,
                         const size_t beam_id)
    : direction_(position), beam_id_(beam_id) {
  spectrum_.SetReferenceFlux(stokes);
}

PointSource::PointSource(const Direction &position, const Spectrum &spectrum,
                         const size_t beam_id)
    : direction_(position), spectrum_(spectrum), beam_id_(beam_id) {}

void PointSource::SetDirection(const Direction &direction) {
  direction_ = direction;
}

void PointSource::ComputeSpectrum(const xt::xtensor<double, 1> &frequencies,
                                  xt::xtensor<double, 3> &result) const {
  spectrum_.EvaluateCrossCorrelations(frequencies, result);
}

void PointSource::SetRotationMeasure(double fraction, double angle, double rm) {
  spectrum_.SetPolarization(angle, fraction);
  spectrum_.SetRotationMeasure(rm);
}

// FIXME: legacy code, remove it later
Stokes PointSource::GetStokes(double freq) const {
  Stokes stokes(spectrum_.GetReferenceFlux());

  if (HasSpectralTerms()) {
    if (spectrum_.HasLogarithmicSpectralIndex()) {
      // Compute spectral index as:
      // (v / v0) ^ (c0 + c1 * log10(v / v0) + c2 * log10(v / v0)^2 + ...)
      // Where v is the frequency and v0 is the reference frequency.

      // Compute log10(v / v0).
      double base = log10(freq) - log10(spectrum_.GetReferenceFrequency());

      // Compute c0 + log10(v / v0) * c1 + log10(v / v0)^2 * c2 + ...
      // using Horner's rule.
      double exponent = 0.0;
      typedef xt::xtensor<double, 1>::const_reverse_iterator iterator_type;
      auto spectral_terms = spectrum_.GetSpectralTerms();
      for (iterator_type it = spectral_terms.rbegin(),
                         end = spectral_terms.rend();
           it != end; ++it) {
        exponent = exponent * base + *it;
      }

      // Compute I * (v / v0) ^ exponent, where I is the value of Stokes
      // I at the reference frequency.
      stokes.I *= pow(10., base * exponent);
      stokes.V *= pow(10., base * exponent);
    } else {
      double x = freq / spectrum_.GetReferenceFrequency() - 1.0;
      typedef xt::xtensor<double, 1>::const_reverse_iterator iterator_type;
      double val = 0.0;
      auto spectral_terms_ = spectrum_.GetSpectralTerms();
      for (iterator_type it = spectral_terms_.rbegin(),
                         end = spectral_terms_.rend();
           it != end; ++it) {
        val = val * x + *it;
      }
      stokes.I += val * x;
      stokes.V += val * x;
    }
  }

  if (HasRotationMeasure()) {
    double lambda = casacore::C::c / freq;
    double chi = 2.0 * (spectrum_.GetPolarizationAngle() +
                        spectrum_.GetRotationMeasure() * lambda * lambda);
    double stokesQU = stokes.I * spectrum_.GetPolarizationFactor();
    stokes.Q = stokesQU * cos(chi);
    stokes.U = stokesQU * sin(chi);
  }
  return stokes;
}

bool PointSource::HasSpectralTerms() const {
  return spectrum_.HasSpectralTerms();
}

bool PointSource::HasRotationMeasure() const {
  return spectrum_.HasRotationMeasure();
}

void GroupSources(double max_separation);

} // namespace predict
