
// Copyright (C) 2025 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

#define BOOST_TEST_MODULE PREDICT_PLAN_EXEC_TEST
#include <boost/test/unit_test.hpp>
#include <predict/PredictPlanExecCPU.h>
#include <predict/test/Common.h>
using namespace predict;
BOOST_AUTO_TEST_SUITE(PredictPlanExecCPUTestSuite)

BOOST_AUTO_TEST_CASE(test_PredictPlanExecCPU) { BOOST_CHECK(true); }

// The assertions within the following two tests are disabled since smear terms
// are not precomputed anymore. As of now, they are computed inline within the
// main computation per-polarization loop, making the tests obsolete.

BOOST_AUTO_TEST_CASE(compute_smear_terms) {
  static constexpr size_t test_n_stations = 4;
  static constexpr size_t test_n_channels = 2;
  static constexpr size_t test_n_sources = 10;
  const Direction test_reference_point{0.5, 0.1};
  const Direction test_offset_point{test_reference_point.ra + 0.02,
                                    test_reference_point.dec + 0.02};
  // Create a SimulationPlanExec object
  std::unique_ptr<PredictRun> run =
      MakePredictRun(test_reference_point, test_offset_point, test_n_stations,
                     test_n_channels, false, true);
  run->Initialize();

  PredictPlan plan;
  PointSourceCollection sources;
  PredictPlanExecCPU sim_plan_exec(run->plan);

  for (size_t s = 0; s < test_n_sources; ++s) {
    sources.Add(run->makeSource<PointSource>());
  }

  // Precompute the station phases and shifts
  sim_plan_exec.Precompute(sources);
}

BOOST_AUTO_TEST_CASE(dont_compute_smear_terms) {
  static constexpr size_t test_n_stations = 4;
  static constexpr size_t test_n_channels = 2;
  static constexpr size_t test_n_sources = 10;
  const Direction test_reference_point{0.5, 0.1};
  const Direction test_offset_point{test_reference_point.ra + 0.02,
                                    test_reference_point.dec + 0.02};
  // Create a SimulationPlanExec object
  std::unique_ptr<PredictRun> run =
      MakePredictRun(test_reference_point, test_offset_point, test_n_stations,
                     test_n_channels, false, false);
  run->Initialize();

  PredictPlan plan;
  PointSourceCollection sources;
  PredictPlanExecCPU sim_plan_exec(run->plan);

  for (size_t s = 0; s < test_n_sources; ++s) {
    sources.Add(run->makeSource<PointSource>());
  }

  // Precompute the station phases and shifts
  sim_plan_exec.Precompute(sources);
}

BOOST_AUTO_TEST_SUITE_END()