// Predict.cppp: Compute visibilities for different model components types
// (implementation of ModelComponentVisitor).
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// $Id$

#include <cmath>
#include <omp.h>
#include <xtensor/xlayout.hpp>

#include "predict/PredictPlan.h"
#include "predict/common/Datastructures.h"

#ifdef ENABLE_TRACY_PROFILING
#include <tracy/Tracy.hpp>
#endif

#include <predict/Directions.h>
#include <predict/GaussianSourceCollection.h>
#include <predict/PointSourceCollection.h>
#include <predict/Predict.h>
#include <predict/PredictPlanExecCPU.h>
#include <predict/test/Common.h>

namespace predict {

void Predict::run(
    PredictPlanExecCPU &plan, const PointSourceCollection &sources,
    Buffer4D &buffer) const { // buffer dimensions are (nCor, nFreq, nBaselines)
  if (!sources.Size())
    return;

  plan.parallelize_over_sources = true;
  plan.Precompute(sources);
  plan.Compute(sources, buffer);
}

void Predict::run(
    PredictPlanExecCPU &plan, const GaussianSourceCollection &sources,
    Buffer4D &buffer) const { // buffer dimensions are (nCor, nFreq, nBaselines)
  if (!sources.Size())
    return;

  plan.parallelize_over_sources = true;
  plan.Precompute(sources);
  plan.ComputeWithTarget(sources, buffer, computation_strategy::XSIMD);
}

void Predict::runWithStrategy(PredictPlanExecCPU &plan,
                              PointSourceCollection &sources, Buffer4D &buffer,
                              const computation_strategy strat) {
  if (!sources.Size())
    return;

  plan.parallelize_over_sources = true;
  plan.Precompute(sources);
  sources.EvaluateSpectra(plan.GetFrequencies());
  plan.ComputeWithTarget(sources, buffer, strat);
}

void Predict::runWithStrategy(PredictPlanExecCPU &plan,
                              GaussianSourceCollection &sources,
                              Buffer4D &buffer,
                              const computation_strategy strat) {
  if (!sources.Size())
    return;

  plan.parallelize_over_sources = true;
  plan.Precompute(sources);
  sources.EvaluateSpectra(plan.GetFrequencies());
  plan.ComputeWithTarget(sources, buffer, strat);
}

everybeam::vector3r_t dir2Itrf(const casacore::MDirection &dir,
                               casacore::MDirection::Convert &measConverter) {
  const casacore::MDirection &itrfDir = measConverter(dir);
  const casacore::Vector<double> &itrf = itrfDir.getValue().getValue();
  return {itrf[0], itrf[1], itrf[2]};
}

template <class SourceTypeCollection>
void populateBeamDirections(
    std::vector<std::pair<size_t, everybeam::vector3r_t>> &directions_itrf,
    casacore::MDirection::Convert &meas_converter,
    const SourceTypeCollection &sources) {
  directions_itrf.clear();
  directions_itrf.reserve(sources.unique_beam_ids.size());

  for (const size_t beam_id : sources.unique_beam_ids) {
    const Direction beam_direction = sources.GetBeamDirection(beam_id);
    const casacore::MDirection direction_j2000(
        casacore::Quantity(beam_direction.ra, "rad"),
        casacore::Quantity(beam_direction.dec, "rad"));
    const everybeam::vector3r_t direction_itrf =
        dir2Itrf(direction_j2000, meas_converter);
    directions_itrf.push_back({beam_id, direction_itrf});
  }
}

} // namespace predict
