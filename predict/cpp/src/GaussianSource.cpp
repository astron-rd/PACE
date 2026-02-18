// GaussianSource.cc: Gaussian source model component.
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// $Id$

#include <predict/GaussianSource.h>

namespace predict {

GaussianSource::GaussianSource(const Direction &direction)
    : PointSource(direction, Stokes{}, 0UL), position_angle_(0.0),
      is_position_angle_absolute_(true), minor_axis_(0.0), major_axis_(0.0) {}

GaussianSource::GaussianSource(const Direction &direction, const Stokes &stokes,
                               const size_t beam_id)
    : PointSource(direction, stokes, beam_id), position_angle_(0.0),
      is_position_angle_absolute_(true), minor_axis_(0.0), major_axis_(0.0) {}

GaussianSource::GaussianSource(const Direction &direction,
                               const Spectrum &spectrum, double position_angle,
                               bool is_position_angle_absolute,
                               double minor_axis, double major_axis,
                               size_t beam_id)
    : PointSource(direction, spectrum, beam_id),
      position_angle_(position_angle),
      is_position_angle_absolute_(is_position_angle_absolute),
      minor_axis_(minor_axis), major_axis_(major_axis) {}

void GaussianSource::SetPositionAngle(double angle) { position_angle_ = angle; }

void GaussianSource::SetMajorAxis(double fwhm) { major_axis_ = fwhm; }

void GaussianSource::SetMinorAxis(double fwhm) { minor_axis_ = fwhm; }

} // namespace predict
