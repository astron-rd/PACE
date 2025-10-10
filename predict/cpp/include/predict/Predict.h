// Predict.h: Compute visibilities for different model components types
// (implementation of ModelComponentVisitor).
//
// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PREDICT_H_
#define PREDICT_H_

#include "PredictPlanExecCPU.h"
#include "common/Datastructures.h"
#include <predict/BeamResponse.h>

#include <xtensor/xlayout.hpp>
#include <xtensor/xtensor.hpp>

#include "Directions.h"
#include "GaussianSourceCollection.h"
#include <predict/Direction.h>
#include <predict/PointSource.h>
#include <predict/Stokes.h>

namespace predict {

/**
 * Compute LMN coordinates of \p direction relative to \p reference.
 * \param[in]   reference
 * Reference direction on the celestial sphere.
 * \param[in]   direction
 * Direction of interest on the celestial sphere.
 * \param[out]   lmn
 * Pointer to a buffer of (at least) length three into which the computed LMN
 * coordinates will be written.
 */
inline void radec2lmn(const Direction &reference, const Direction &direction,
                      double *lmn) {
  /**
   * \f{eqnarray*}{
   *   \ell &= \cos(\delta) \sin(\alpha - \alpha_0) \\
   *      m &= \sin(\delta) \cos(\delta_0) - \cos(\delta) \sin(\delta_0)
   *                                         \cos(\alpha - \alpha_0)
   * \f}
   */
  const double delta_ra = direction.ra - reference.ra;
  const double sin_delta_ra = std::sin(delta_ra);
  const double cos_delta_ra = std::cos(delta_ra);
  const double sin_dec = std::sin(direction.dec);
  const double cos_dec = std::cos(direction.dec);
  const double sin_dec0 = std::sin(reference.dec);
  const double cos_dec0 = std::cos(reference.dec);

  lmn[0] = cos_dec * sin_delta_ra;
  lmn[1] = sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_delta_ra;
  // Normally, n is calculated using n = std::sqrt(1.0 - l * l - m * m).
  // However the sign of n is lost that way, so a calculation is used that
  // avoids losing the sign. This formula can be found in Perley (1989).
  // Be aware that we asserted that the sign is wrong in Perley (1989),
  // so this is corrected.
  lmn[2] = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_delta_ra;
}

// Compute component spectrum.
inline void spectrum(const PointSource &component, size_t nChannel,
                     const xt::xtensor<double, 1> &freq,
                     xt::xtensor<std::complex<double>, 2> &spectrum,
                     bool stokesIOnly = false) {
#pragma GCC ivdep
  for (size_t ch = 0; ch < nChannel; ++ch) {
    Stokes stokes = component.GetStokes(freq(ch));

    if (stokesIOnly) {
      spectrum(0, ch).real(stokes.I);
      spectrum(0, ch).imag(0.0);
    } else {
      spectrum(0, ch).real(stokes.I + stokes.Q);
      spectrum(0, ch).imag(0.0);
      spectrum(1, ch).real(stokes.U);
      spectrum(1, ch).imag(stokes.V);
      spectrum(2, ch).real(stokes.U);
      spectrum(2, ch).imag(-stokes.V);
      spectrum(3, ch).real(stokes.I - stokes.Q);
      spectrum(3, ch).imag(0.0);
    }
  }
}

/// @{

/**
 * @brief Predict class to compute visibilities given a sky model
 *
 * This class computes visibilities given model components in the sky.
 * Effectively, it evaluates the equation:
 *
 * \f[ V(u, v, w) =
 * \iint I(\ell, m) e^{2\pi i(u,v,w)\cdot(\ell,m,n)}\mathrm{d}\ell\mathrm{d} m
 * \f]
 *
 * where \f$ I(\ell, m) \f$ is the model intensity in the sky.
 * For a point source, \f$ I(\ell, m) \f$ is a delta function, with given
 * intensity $\mathrm{I}$, for a Gaussian source oriented along the coordinate
 * axes $\ell$ and $m$, we have
 *
 * \f[
 * I(\ell, m) = \mathrm{I} \frac{1}{\sqrt{2\pi\sigma_\ell\sigma_m}
 *              e^{-\frac{\ell^2}{2\sigma_\ell^2}-\frac{m^2}{2\sigma_m^2}},
 * \]
 *
 * where \f$ \sigma_\ell \f$ and \f$ \sigma_m \f$ are computed from the FWHM
 * major and minor axis in the sky model. The normalization is such that
 * \mathrm{I} represents the integrated flux of the Gaussian source.
 * For Gaussian sources, a position angle or orientation can be
 * given, representing the orientation of the source. Depending on the value
 * of 'OrientationIsAbsolute' in the model component, this is the orientation
 * with respect to the declination axis (if OrientationIsAbsolute is true) or
 * with respect to the m axis for this observation (if OrientationIsAbsolute
 * is false).
 * To compute Gausian sources, the coordinate system is rotated to a new
 * coordinate system where it is oriented along the axes. For the case where
 * 'OrientationIsAbsolute' is true, the $u,v,w$ coordinates are phase-shifted
 * to the position of the source.
 */

class Predict {
public:
  void run(PredictPlanExecCPU &plan, const PointSourceCollection &sources,
           Buffer4D &buffer) const;

  void run(PredictPlanExecCPU &plan, const GaussianSourceCollection &sources,
           Buffer4D &buffer) const;

  void runWithStrategy(PredictPlanExecCPU &plan, PointSourceCollection &sources,
                       Buffer4D &buffer, const computation_strategy strat);

  void runWithStrategy(PredictPlanExecCPU &plan,
                       GaussianSourceCollection &sources, Buffer4D &buffer,
                       const computation_strategy strat);

  void runWithStrategy(PredictPlanExecCPU &plan, BeamResponsePlan &beam,
                       PointSourceCollection &sources,
                       GaussianSourceCollection &gaussian_sources,
                       Buffer4D &buffer,
                       casacore::MDirection::Convert &meas_converter,
                       const computation_strategy strat);
};
} // namespace predict
#endif
