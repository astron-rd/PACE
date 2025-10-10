// GaussianSourceCollection.h: A collection of gaussian sources
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief A collection of gaussian sources

#ifndef GAUSSIAN_SOURCE_COLLECTION_H_
#define GAUSSIAN_SOURCE_COLLECTION_H_

#include <vector>

#include "GaussianSource.h"
#include "ObjectCollection.h"
#include "PointSourceCollection.h"
#include "common/SmartVector.h"

namespace predict {

class GaussianSourceCollection : public PointSourceCollection {
public:
  using ObjectCollection<PointSource>::Add;

  void Add(const GaussianSource &gaussian_source) {
    PointSourceCollection::Add(gaussian_source);

    position_angle.push_back(gaussian_source.GetPositionAngle());
    major_axis.push_back(gaussian_source.GetMajorAxis());
    minor_axis.push_back(gaussian_source.GetMinorAxis());
    position_angle_is_absolute.push_back(
        gaussian_source.GetPositionAngleIsAbsolute());
  }

  void Reserve(size_t new_size) {
    position_angle.reserve(new_size);
    major_axis.reserve(new_size);
    minor_axis.reserve(new_size);
    position_angle_is_absolute.reserve(new_size);
    PointSourceCollection::Reserve(new_size);
  }

  GaussianSource operator[](size_t i) const {
    return GaussianSource(direction_vector[i], spectra[i], position_angle[i],
                          position_angle_is_absolute[i], minor_axis[i],
                          major_axis[i], beam_id[i]);
  }

  std::unique_ptr<GaussianSourceCollection> SelectBeamID(size_t beam_id) const {
#ifdef ENABLE_TRACY_PROFILING
    ZoneScoped;
#endif
    auto selected = std::make_unique<GaussianSourceCollection>();

    for (size_t i = 0; i < Size(); ++i) {
      if (this->beam_id[i] == beam_id) {
        selected->Add(operator[](i));
      }
    }
    return selected;
  }

  SmartVector<double> position_angle;
  SmartVector<double> major_axis;
  SmartVector<double> minor_axis;
  SmartVector<bool> position_angle_is_absolute;
};

} // namespace predict

#endif // GAUSSIAN_SOURCE_COLLECTION_H_
