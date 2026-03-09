// Spectrum.h: Spectral fit for a given source model
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief Spectral fit for a given source model

#ifndef SPECTRUM_H_
#define SPECTRUM_H_

#include "Stokes.h"
#include "StokesVector.h"
#ifdef ENABLE_TRACY_PROFILING
#include <tracy/Tracy.hpp>
#endif
#include <xtensor/xcomplex.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace predict {

enum CrossCorrelation { XX = 0, XY = 1, YX = 2, YY = 3 };
enum Complex { REAL = 0, IMAG = 1 };

class Spectrum {
public:
  Spectrum()
      : reference_flux_(), reference_frequency_(0.0), polarization_angle_(0.0),
        polarization_factor_(0.0), has_rotation_measure_(false),
        rotation_measure_(0.0), has_logarithmic_spectral_index_(false) {}
  Spectrum(Stokes reference_flux, double reference_frequency,
           double polarization_angle, double polarization_factor,
           double rotation_measure, bool has_rotation_measure,
           bool has_logarithmic_spectral_index)
      : reference_flux_(reference_flux),
        reference_frequency_(reference_frequency),
        polarization_angle_(polarization_angle),
        polarization_factor_(polarization_factor),
        has_rotation_measure_(has_rotation_measure),
        rotation_measure_(rotation_measure),
        has_logarithmic_spectral_index_(has_logarithmic_spectral_index) {}

  void Evaluate(const xt::xtensor<double, 1> &frequencies,
                StokesVector &result) const {
    result.I.resize(frequencies.size());
    result.Q.resize(frequencies.size());
    result.U.resize(frequencies.size());
    result.V.resize(frequencies.size());

    double I0 = reference_flux_.I;
    double V0 = reference_flux_.V;
    double U0 = reference_flux_.U;
    double Q0 = reference_flux_.Q;

    if (spectral_terms_.size() > 0 && has_logarithmic_spectral_index_) {
      for (size_t f_idx = 0; f_idx < frequencies.size(); f_idx++) {
        const double spectral_term = EvaluateLogSpectrum(
            frequencies[f_idx], reference_frequency_, spectral_terms_);
        result.I[f_idx] = I0 * spectral_term;
        result.V[f_idx] = V0 * spectral_term;
        result.U[f_idx] = U0 * spectral_term;
        result.Q[f_idx] = Q0 * spectral_term;
      }
    } else if (spectral_terms_.size()) {
      for (size_t f_idx = 0; f_idx < frequencies.size(); f_idx++) {
        const double spectral_term = EvaluateSpectrum(
            frequencies[f_idx], reference_frequency_, spectral_terms_);
        result.I[f_idx] = I0 + spectral_term;
        result.V[f_idx] = V0 + spectral_term;
        result.U[f_idx] = U0 * spectral_term;
        result.Q[f_idx] = Q0 * spectral_term;
      }
    } else {
      for (size_t f_idx = 0; f_idx < frequencies.size(); f_idx++) {
        result.I[f_idx] = I0;
        result.V[f_idx] = V0;
        result.U[f_idx] = U0;
        result.Q[f_idx] = Q0;
      }
    }

    if (has_rotation_measure_) {
      // This way of computing is faster then using the xtensor extension
      for (size_t f_idx = 0; f_idx < frequencies.size(); f_idx++) {
        const double lambda = casacore::C::c / frequencies[f_idx];
        const double chi =
            2.0 * (polarization_angle_ + (rotation_measure_ * lambda * lambda));
        const double stokesQU = result.I[f_idx] * polarization_factor_;
        result.Q[f_idx] = stokesQU * cos(chi);
        result.U[f_idx] = stokesQU * sin(chi);
      }
    }
  }

  void SetSpectralTerms(double refFreq, bool isLogarithmicSpectralIndex,
                        const xt::xtensor<double, 1> &terms = {}) {
    reference_frequency_ = refFreq;
    has_logarithmic_spectral_index_ = isLogarithmicSpectralIndex;
    spectral_terms_ = terms;
  }
  void SetSpectralTerms(double refFreq, bool isLogarithmicSpectralIndex,
                        const std::vector<double> &terms) {
    reference_frequency_ = refFreq;
    has_logarithmic_spectral_index_ = isLogarithmicSpectralIndex;
    spectral_terms_ = xt::adapt(terms);
  }
  const xt::xtensor<double, 1> &GetSpectralTerms() const {
    return spectral_terms_;
  }

  void ClearSpectralTerms() { spectral_terms_ = xt::xtensor<double, 1>(); }

  double GetReferenceFrequency() const { return reference_frequency_; }
  bool HasLogarithmicSpectralIndex() const {
    return has_logarithmic_spectral_index_;
  }
  bool HasSpectralTerms() const { return spectral_terms_.size() > 0; }
  bool HasRotationMeasure() const { return has_rotation_measure_; }
  void SetPolarization(double angle, double factor) {
    polarization_angle_ = angle;
    polarization_factor_ = factor;
  }

  double GetPolarizationAngle() const { return polarization_angle_; }
  double GetPolarizationFactor() const { return polarization_factor_; }

  void SetRotationMeasure(double rm, bool has_rm = true) {
    rotation_measure_ = rm;
    has_rotation_measure_ = has_rm;
  }

  double GetRotationMeasure() const { return rotation_measure_; }

  void ClearRotationMeasure() {
    rotation_measure_ = 0.0;
    has_rotation_measure_ = false;
  }

  void SetReferenceFlux(const Stokes &flux) { reference_flux_ = flux; }

  const Stokes &GetReferenceFlux() const { return reference_flux_; }

  void EvaluateCrossCorrelations(const xt::xtensor<double, 1> &frequencies,
                                 xt::xtensor<double, 3> &result,
                                 const bool need_reshape = true) const {
#ifdef ENABLE_TRACY_PROFILING
    ZoneScoped;
#endif
    xt::xtensor<double, 1>::shape_type kSpectralCorrectionShape{
        frequencies.size()};
    xt::xtensor<double, 2>::shape_type kRotationCoefficientsShape{
        2, frequencies.size()};
    xt::xtensor<double, 1> spectral_correction =
        xt::zeros<double>(kSpectralCorrectionShape);
    xt::xtensor<double, 2> rotation_coefficients =
        xt::zeros<double>(kRotationCoefficientsShape);

    if (need_reshape) {
      result.resize({4, 2, frequencies.size()});
    }

    if (spectral_terms_.size() > 0) {
      if (has_logarithmic_spectral_index_) {
        for (size_t f_idx = 0; f_idx < frequencies.size(); f_idx++) {
          spectral_correction[f_idx] = EvaluateLogSpectrum(
              frequencies[f_idx], reference_frequency_, spectral_terms_);
        }
      } else {
        for (size_t f_idx = 0; f_idx < frequencies.size(); f_idx++) {
          spectral_correction[f_idx] = EvaluateSpectrum(
              frequencies[f_idx], reference_frequency_, spectral_terms_);
        }
      }
    } else if (has_logarithmic_spectral_index_) {
      spectral_correction = {1.0};
    }

    if (has_rotation_measure_) {
      for (size_t f_idx = 0; f_idx < frequencies.size(); f_idx++) {
        const double kLambda = casacore::C::c / frequencies[f_idx];
        const double kChi = 2.0 * (polarization_angle_ +
                                   (rotation_measure_ * kLambda * kLambda));
        const double stokesQU = polarization_factor_;
        rotation_coefficients(0, f_idx) = stokesQU * cos(kChi);
        rotation_coefficients(1, f_idx) = stokesQU * sin(kChi);
      }
    }

    const std::array<size_t, 1> shape{frequencies.size()};
    xt::xtensor<double, 1> I(shape);
    xt::xtensor<double, 1> V(shape);
    xt::xtensor<double, 1> Q(shape);
    xt::xtensor<double, 1> U(shape);
    if (has_logarithmic_spectral_index_) {
      I = spectral_correction * reference_flux_.I;
    } else {
      I = spectral_correction + reference_flux_.I;
    }

    xt::view(V, xt::all()) = reference_flux_.V;

    if (HasRotationMeasure()) {
      Q = I * xt::view(rotation_coefficients, 0, xt::all());
      U = I * xt::view(rotation_coefficients, 1, xt::all());
    } else {
      xt::view(Q, xt::all()) = reference_flux_.Q;
      xt::view(U, xt::all()) = reference_flux_.U;
    }
    xt::view(result, CrossCorrelation::XX, Complex::REAL, xt::all()) = I + Q;
    xt::view(result, CrossCorrelation::XX, Complex::IMAG, xt::all()) = 0.0;
    xt::view(result, CrossCorrelation::XY, Complex::REAL, xt::all()) = U;
    xt::view(result, CrossCorrelation::XY, Complex::IMAG, xt::all()) = V;
    xt::view(result, CrossCorrelation::YX, Complex::REAL, xt::all()) = U;
    xt::view(result, CrossCorrelation::YX, Complex::IMAG, xt::all()) = -V;
    xt::view(result, CrossCorrelation::YY, Complex::REAL, xt::all()) = I - Q;
    xt::view(result, CrossCorrelation::YY, Complex::IMAG, xt::all()) = 0.0;
  }

private:
  inline double
  EvaluatePolynomial(double x, const xt::xtensor<double, 1> &parameters) const {
    double partial = 0.0;
    for (auto it = spectral_terms_.rbegin(), end = spectral_terms_.rend();
         it != end; ++it) {
      partial = std::fma(partial, x, *it);
    }
    return partial;
  }
  inline double
  EvaluateLogPolynomial(double x,
                        const xt::xtensor<double, 1> &parameters) const {
    const double base = x;
    return EvaluatePolynomial(base, parameters);
  }

  inline double
  EvaluateLogSpectrum(const double frequency, const double reference_frequency,
                      const xt::xtensor<double, 1> &parameters) const {
    double x = log10(frequency) - log10(reference_frequency);
    return pow(10, x * EvaluateLogPolynomial(x, parameters));
  }
  inline double
  EvaluateSpectrum(const double frequency, const double reference_frequency,
                   const xt::xtensor<double, 1> &parameters) const {
    const double x = frequency / reference_frequency - 1.0;

    return EvaluatePolynomial(x, parameters) * x;
  }

private:
  Stokes reference_flux_;
  double reference_frequency_;
  double polarization_angle_;
  double polarization_factor_;

  bool has_rotation_measure_;
  double rotation_measure_;

  xt::xtensor<double, 1> spectral_terms_;
  bool has_logarithmic_spectral_index_;
};

} // namespace predict

#endif // SPECTRUM_H_
