// GaussianSource.h: Gaussian source model component.
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PREDICT_GAUSSIANSOURCE_H
#define PREDICT_GAUSSIANSOURCE_H

#include "PointSource.h"

namespace predict {

/// \brief Gaussian source model component.

/// @{

class GaussianSource : public PointSource {
public:
  typedef std::shared_ptr<GaussianSource> Ptr;
  typedef std::shared_ptr<const GaussianSource> ConstPtr;

  GaussianSource(const Direction &direction);
  GaussianSource(const Direction &direction, const Stokes &stokes,
                 size_t beam_id = 0);
  GaussianSource(const Direction &direction, const Spectrum &spectrum,
                 double position_angle, bool is_position_angle_absolute,
                 double minor_axis, double major_axis, size_t beam_id = 0);

  /// Set position angle in radians. The position angle is the smallest angle
  /// between the major axis and North, measured positively North over East.
  void SetPositionAngle(double angle);
  double GetPositionAngle() const { return position_angle_; }

  /// Set whether the position angle (orientation) is absolute, see
  /// documentation of class member)
  void SetPositionAngleIsAbsolute(bool positionAngleIsAbsolute) {
    is_position_angle_absolute_ = positionAngleIsAbsolute;
  }

  /// Return whether the position angle (orientation) is absolute, see
  /// documentation of class member.
  bool GetPositionAngleIsAbsolute() const {
    return is_position_angle_absolute_;
  }
  /// Set the minor axis length (FWHM in radians).
  void SetMinorAxis(double fwhm);
  double GetMinorAxis() const { return minor_axis_; }

  /// Set the major axis length (FWHM in radians).
  void SetMajorAxis(double fwhm);
  double GetMajorAxis() const { return major_axis_; }

private:
  double position_angle_;
  /// Whether the position angle (also refered to as orientation) is absolute
  /// (w.r.t. to the local declination axis) or with respect to the declination
  /// axis at the phase center (the default until 2022, it was fixed in 5.3.0)
  bool is_position_angle_absolute_;
  double minor_axis_;
  double major_axis_;
};

/// @}

} // namespace predict
#endif // PREDICT_GAUSSIANSOURCE_H
