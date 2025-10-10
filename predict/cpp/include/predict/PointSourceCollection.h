// PointSourceCollection.h: a collection of PointSource
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief a collection of PointSource

#ifndef POINT_SOURCE_COLLECTION_H_
#define POINT_SOURCE_COLLECTION_H_

#include <unordered_map>
#include <vector>

#include "predict/Direction.h"
#include <predict/Directions.h>
#include <predict/ObjectCollection.h>
#include <predict/PointSource.h>
#include <predict/StokesVector.h>
#include <set>

namespace predict {
class PointSourceCollection : public ObjectCollection<PointSource> {
public:
  void Add(const PointSource &point_source) {
    direction_vector.Add(point_source.GetDirection());
    spectra.push_back(point_source.GetSpectrum());
    beam_id.push_back(point_source.GetBeamId());
    unique_beam_ids.insert(point_source.GetBeamId());
  }

  void UpdateBeams() {
    unique_beam_ids.clear();
    for (const auto &id : beam_id) {
      unique_beam_ids.insert(id);
    }
  }

  void Reserve(size_t new_size) {
    direction_vector.Reserve(new_size);
    spectra.reserve(new_size);
    beam_id.reserve(new_size);
  }

  void Clear() {
    direction_vector.Clear();
    spectra.clear();
    beam_id.clear();
    unique_beam_ids.clear();
  }

  void Resize(size_t new_size) {
    direction_vector.Resize(new_size);
    spectra.resize(new_size);
    beam_id.resize(new_size);
    UpdateBeams();
  }

  PointSource operator[](size_t i) const {
    return PointSource(direction_vector[i], spectra[i], beam_id[i]);
  }

  std::unique_ptr<PointSourceCollection> SelectBeamID(size_t beam_id) {
    auto selected = std::make_unique<PointSourceCollection>();

    for (size_t i = 0; i < Size(); ++i) {
      if (this->beam_id[i] == beam_id) {
        selected->Add(operator[](i));
      }
    }
    return selected;
  }

  void AddBeamDirection(const size_t beam_id, const Direction &direction) {
    beam_directions.try_emplace(beam_id, direction);
  }

  const Direction GetBeamDirection(const size_t beam_id) const {
    return beam_directions.at(beam_id);
  }

  void EvaluateSpectra(const xt::xtensor<double, 1> &frequencies) {
    evaluated_spectra.resize({4, Size(), 2, frequencies.size()});
    for (size_t source_id = 0; source_id < Size(); source_id++) {
      xt::xtensor<double, 3> source_spectrum({4, 2, frequencies.size()});
      source_spectrum.fill(0.0);

      spectra[source_id].EvaluateCrossCorrelations(frequencies, source_spectrum,
                                                   false);
      xt::view(evaluated_spectra, xt::all(), source_id, xt::all(), xt::all()) =
          source_spectrum;
    }
  }

  size_t Size() const { return direction_vector.Size(); }
  void GroupSources(double max_angular_separation);

  Directions direction_vector;
  std::vector<Spectrum> spectra;
  std::vector<size_t> beam_id;
  std::set<size_t> unique_beam_ids;
  std::unordered_map<size_t, Direction> beam_directions;
  xt::xtensor<double, 4> evaluated_spectra;
};

} // namespace predict

#endif // POINT_SOURCE_COLLECTION_H_
