// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <boost/test/tools/old/interface.hpp>
#include <predict/Directions.h>
#include <predict/GaussianSourceCollection.h>
#include <predict/PointSourceCollection.h>
#include <xtensor/xtensor.hpp>

#include <predict/Direction.h>
#include <predict/GaussianSource.h>
#include <predict/PointSource.h>
#include <predict/PredictPlan.h>
#include <predict/PredictPlanExecCPU.h>
#include <predict/Stokes.h>
#include <xtensor/xtensor_forward.hpp>

#define BOOST_TEST_MODULE SIMULATOR_TEST
#include <boost/test/unit_test.hpp>

#include <predict/test/Common.h>

namespace predict {
namespace test {

inline double ConvToAbsComplex(const Buffer4D &buffer, const size_t pol,
                               const size_t bl, const size_t channel) {
  return std::abs(std::complex<double>{buffer(pol, bl, 0, channel),
                                       buffer(pol, bl, 1, channel)});
}

struct PredictTestFixture {
  PredictTestFixture() : predict_run(nullptr) {}

  void SetupPredictRun(bool onlyI, bool freqsmear, size_t test_n_stations_ = 4,
                       size_t test_n_channels_ = 2) {
    test_n_channels = test_n_channels_;
    test_n_stations = test_n_stations_;
    predict_run =
        MakePredictRun(test_reference_point, test_offset_point,
                       test_n_stations_, test_n_channels_, onlyI, freqsmear);
  }

  size_t test_n_stations = 4;
  size_t test_n_channels = 2;
  const Direction test_reference_point{0.5, 0.1};
  const Direction test_offset_point{test_reference_point.ra + 0.02,
                                    test_reference_point.dec + 0.02};

  std::unique_ptr<PredictRun> predict_run;
};

BOOST_FIXTURE_TEST_SUITE(PredictTestSuite, PredictTestFixture)

BOOST_AUTO_TEST_CASE(pointsource_onlyI) {
  SetupPredictRun(true, false);
  auto pointsource = predict_run->makeSource<PointSource>();

  PointSourceCollection sources;
  sources.Add(pointsource);
  predict_run->Run(sources);

  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 0), 1.0,
                    1.0e-3); // Channel 0
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 1, 0), 1.0,
                    1.0e-3); // Channel 1
}

BOOST_AUTO_TEST_CASE(pointsource_fullstokes) {
  SetupPredictRun(false, false);
  auto pointsource = predict_run->makeSource<PointSource>();

  PointSourceCollection sources;
  sources.Add(pointsource);
  predict_run->Run(sources);

  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 0), 1.0,
                    1.0e-3); // Channel 0, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 1, 0, 0), 0.0,
                    1.0e-3); // Channel 0, XY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 2, 0, 0), 0.0,
                    1.0e-3); // Channel 0, YX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 3, 0, 0), 1.0,
                    1.0e-3); // Channel 0, YY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 1, 0), 1.0,
                    1.0e-3); // Channel 1, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 1, 1, 0), 0.0,
                    1.0e-3); // Channel 1, XY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 2, 1, 0), 0.0,
                    1.0e-3); // Channel 1, YX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 3, 1, 0), 1.0,
                    1.0e-3); // Channel 1, YY
}

BOOST_AUTO_TEST_CASE(pointsource_onlyI_freqsmear) {
  SetupPredictRun(true, true);
  auto pointsource = predict_run->makeSource<PointSource>();

  PointSourceCollection sources;
  sources.Add(pointsource);
  predict_run->RunWithStrategy(sources, computation_strategy::XSIMD);

  BOOST_CHECK_CLOSE(
      ConvToAbsComplex(predict_run->buffer, 0, 0, test_n_channels - 1), 0.75915,
      1.0e-3);
  BOOST_CHECK_CLOSE(
      ConvToAbsComplex(predict_run->buffer, 1, 0, test_n_channels - 1), 0.75915,
      1.0e-3);
}

BOOST_AUTO_TEST_CASE(pointsource_fullstokes_freqsmear_all_strategies) {
  test_n_stations = 16;
  test_n_channels =
      51; // Take an odd number of channels to verify boundary cases
  SetupPredictRun(false, true);
  PointSourceCollection sources;

  // Generate multiple point sources
  const size_t num_sources = 4;
  for (size_t i = 0; i < num_sources; ++i) {
    auto pointsource = predict_run->makeSource<PointSource>();
    sources.Add(pointsource);
  }

  // Define computation strategies to test
  const std::vector<computation_strategy> strategies = {
      computation_strategy::SINGLE, computation_strategy::XSIMD};

  const auto ref_buffer = [&]() {
    predict_run->RunWithStrategy(sources, computation_strategy::SINGLE);
    return predict_run->buffer;
  }();

  for (const auto &strategy : strategies) {
    predict_run->Clean();
    predict_run->RunWithStrategy(sources, strategy);

    BOOST_TEST_CONTEXT("Testing strategy: " << static_cast<int>(strategy)) {
      for (size_t bl = 0; bl < predict_run->plan.nbaselines; ++bl) {
        for (size_t ch = 0; ch < predict_run->plan.nchannels; ++ch) {
          for (size_t pol = 0; pol < 4; ++pol) {
            BOOST_CHECK_CLOSE(
                ConvToAbsComplex(ref_buffer, pol, bl, ch),
                ConvToAbsComplex(predict_run->buffer, pol, bl, ch), 1.0e-3);
          }
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(pointsource_fullstokes_freqsmear_xtensor) {
  test_n_stations = 16;
  test_n_channels =
      51; // Take a odd number of channels to verify boundary cases
  SetupPredictRun(false, true);
  PointSourceCollection sources;

  // Generate multiple point sources
  const size_t num_sources = 4;
  for (size_t i = 0; i < num_sources; ++i) {
    auto pointsource = predict_run->makeSource<PointSource>();
    sources.Add(pointsource);
  }

  predict_run->RunWithStrategy(sources, computation_strategy::SINGLE);

  const auto ref_buffer = predict_run->buffer;
  predict_run->Clean();

  predict_run->RunWithStrategy(sources, computation_strategy::XSIMD);

  for (size_t bl = 0; bl < predict_run->plan.nbaselines; ++bl) {
    for (size_t ch = 0; ch < predict_run->plan.nchannels; ++ch) {
      for (size_t pol = 0; pol < 4; ++pol) {
        BOOST_CHECK_CLOSE(ConvToAbsComplex(ref_buffer, pol, bl, ch),
                          ConvToAbsComplex(predict_run->buffer, pol, bl, ch),
                          1.0e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(gaussiansource_onlyI) {
  SetupPredictRun(true, false);
  auto gaussiansource = predict_run->makeSource<GaussianSource>();

  GaussianSourceCollection sources;
  sources.Add(gaussiansource);
  predict_run->Run(sources);

  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 0), 1.0,
                    1.0e-3); // Channel 0
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 1), 1.0,
                    1.0e-3); // Channel 1
}

BOOST_AUTO_TEST_CASE(gaussiansource_fullstokes) {
  SetupPredictRun(false, false);
  auto gaussiansource = predict_run->makeSource<GaussianSource>();

  GaussianSourceCollection sources;
  sources.Add(gaussiansource);
  predict_run->Run(sources);

  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 0), 1.0,
                    1.0e-3); // Channel 0, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 1, 0, 0), 0.0,
                    1.0e-3); // Channel 0, XY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 2, 0, 0), 0.0,
                    1.0e-3); // Channel 0, YX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 3, 0, 0), 1.0,
                    1.0e-3); // Channel 0, YY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 1, 0), 1.0,
                    1.0e-3); // Channel 1, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 1, 1, 0), 0.0,
                    1.0e-3); // Channel 1, XY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 2, 1, 0), 0.0,
                    1.0e-3); // Channel 1, YX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 3, 1, 0), 1.0,
                    1.0e-3); // Channel 1, YY
}

BOOST_AUTO_TEST_CASE(gaussiansource_onlyI_freqsmear) {
  SetupPredictRun(true, true);
  auto gaussiansource = predict_run->makeSource<GaussianSource>();

  GaussianSourceCollection sources;
  sources.Add(gaussiansource);
  predict_run->Run(sources);

  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 0), 0.75915,
                    1.0e-3); // Channel 0
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 1), 0.75915,
                    1.0e-3); // Channel 1
}

BOOST_AUTO_TEST_CASE(gaussiansource_absolute_angle_empty_initialization) {
  SetupPredictRun(false, false);
  GaussianSource gaussiansource{predict_run->offset_source};

  BOOST_CHECK_EQUAL(gaussiansource.GetPositionAngle(), 0.0);
  BOOST_CHECK_EQUAL(gaussiansource.GetPositionAngleIsAbsolute(), true);
  BOOST_CHECK_EQUAL(gaussiansource.GetMajorAxis(), 0.0);
  BOOST_CHECK_EQUAL(gaussiansource.GetMinorAxis(), 0.0);
}

BOOST_AUTO_TEST_CASE(gaussiansource_absolute_angle_onlyI) {
  SetupPredictRun(true, false);
  auto gaussiansource = predict_run->makeSource<GaussianSource>();

  gaussiansource.SetPositionAngleIsAbsolute(true);
  gaussiansource.SetPositionAngle(0.00025);
  gaussiansource.SetMajorAxis(0.00008);
  gaussiansource.SetMinorAxis(0.00002);

  GaussianSourceCollection sources;
  sources.Add(gaussiansource);

  predict_run->Run(sources);

  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 0), 0.98917,
                    1.0e-3); // Channel 0, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 1), 0.98900,
                    1.0e-3); // Channel 0, XY
}

BOOST_AUTO_TEST_CASE(gaussiansource_absolute_angle_fullstokes) {
  SetupPredictRun(false, false);
  auto gaussiansource = predict_run->makeSource<GaussianSource>();

  gaussiansource.SetPositionAngleIsAbsolute(true);
  gaussiansource.SetPositionAngle(0.00025);
  gaussiansource.SetMajorAxis(0.00008);
  gaussiansource.SetMinorAxis(0.00002);

  GaussianSourceCollection sources;
  sources.Add(gaussiansource);

  predict_run->Run(sources);

  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 0), 0.98917,
                    1.0e-3); // Channel 0, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 1, 0, 0), 0.0,
                    1.0e-3); // Channel 0, XY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 2, 0, 0), 0.0,
                    1.0e-3); // Channel 0, YX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 3, 0, 0), 0.98917,
                    1.0e-3); // Channel 0, YY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 1), 0.98900,
                    1.0e-3); // Channel 1, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 1, 0, 1), 0.0,
                    1.0e-3); // Channel 1, XY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 2, 0, 1), 0.0,
                    1.0e-3); // Channel 1, YX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 3, 0, 1), 0.98900,
                    1.0e-3); // Channel 1, YY
}

BOOST_AUTO_TEST_CASE(gaussiansource_absolute_angle_onlyI_freqsmear) {
  SetupPredictRun(true, true);
  auto gaussiansource = predict_run->makeSource<GaussianSource>();

  gaussiansource.SetPositionAngleIsAbsolute(true);
  gaussiansource.SetPositionAngle(0.00025);
  gaussiansource.SetMajorAxis(0.00008);
  gaussiansource.SetMinorAxis(0.00002);

  GaussianSourceCollection sources;
  sources.Add(gaussiansource);

  predict_run->Run(sources);

  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 0), 0.75093,
                    1.0e-3); // Channel 0, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 1), 0.75081,
                    1.0e-3); // Channel 0, XY
}

BOOST_AUTO_TEST_CASE(gaussiansource_absolute_angle_fullstokes_freqsmear) {
  SetupPredictRun(false, true);
  auto gaussiansource = predict_run->makeSource<GaussianSource>();

  gaussiansource.SetPositionAngleIsAbsolute(true);
  gaussiansource.SetPositionAngle(0.00025);
  gaussiansource.SetMajorAxis(0.00008);
  gaussiansource.SetMinorAxis(0.00002);

  GaussianSourceCollection sources;
  sources.Add(gaussiansource);

  predict_run->Run(sources);

  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 0), 0.750934,
                    1.0e-3); // Channel 0, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 1, 0, 0), 0.0,
                    1.0e-3); // Channel 0, XY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 2, 0, 0), 0.0,
                    1.0e-3); // Channel 0, YX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 3, 0, 0), 0.750934,
                    1.0e-3); // Channel 0, YY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 0, 0, 1), 0.750807,
                    1.0e-3); // Channel 1, XX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 1, 0, 1), 0.0,
                    1.0e-3); // Channel 1, XY
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 2, 0, 1), 0.0,
                    1.0e-3); // Channel 1, YX
  BOOST_CHECK_CLOSE(ConvToAbsComplex(predict_run->buffer, 3, 0, 1), 0.750807,
                    1.0e-3); // Channel 1, YY
}

BOOST_AUTO_TEST_CASE(radec_to_lmn_conversion_simple) {
  Directions dirs;
  // RA 0, DEC 90 degrees: (=north celestial pole)
  const Direction reference(0.0, 0.5 * M_PI);
  const Direction direction(0.0, 0.5 * M_PI);

  dirs.Add(direction);

  xt::xtensor<double, 2> lmn({1, 3});

  dirs.RaDec2Lmn<Directions::XSIMD>(reference, lmn);
  BOOST_CHECK_LT(std::abs(lmn(0, 0)), 1e-6);
  BOOST_CHECK_LT(std::abs(lmn(0, 1)), 1e-6);
  BOOST_CHECK_CLOSE_FRACTION(lmn(0, 2), 1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(radec_to_lmn_conversion_negative_n) {
  Directions dirs;

  // Check sign of N when reference and direction are opposite
  const Direction reference(0.0, 0.5 * M_PI);
  const Direction direction(0.0, -0.5 * M_PI);

  dirs.Add(direction);

  xt::xtensor<double, 2> lmn({1, 3});

  dirs.RaDec2Lmn(reference, lmn);
  BOOST_CHECK_LT(std::abs(lmn(0, 0)), 1e-6);
  BOOST_CHECK_LT(std::abs(lmn(0, 1)), 1e-6);
  BOOST_CHECK_CLOSE_FRACTION(lmn(0, 2), -1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(radec_to_lmn_conversion_simple_2) {
  Directions dirs;

  const Direction reference(0.0, 0.5 * M_PI);
  const Direction direction(0.0, 0.25 * M_PI);

  dirs.Add(direction);

  xt::xtensor<double, 2> lmn({1, 3});
  dirs.RaDec2Lmn(reference, lmn);
  BOOST_CHECK_LT(std::abs(lmn(0, 0)), 1e-6);
  BOOST_CHECK_CLOSE_FRACTION(lmn(0, 1), -M_SQRT1_2, 1e-6);
  BOOST_CHECK_CLOSE_FRACTION(lmn(0, 2), M_SQRT1_2, 1e-6);
}

BOOST_AUTO_TEST_CASE(radec_to_lmn_conversion_negative_n_2) {
  Directions dirs;

  const Direction reference(0.0, 0.5 * M_PI);
  const Direction direction(0.0, -0.25 * M_PI);

  dirs.Add(direction);

  xt::xtensor<double, 2> lmn({1, 3});

  dirs.RaDec2Lmn(reference, lmn);
  BOOST_CHECK_LT(std::abs(lmn(0, 0)), 1e-6);
  BOOST_CHECK_CLOSE_FRACTION(lmn(0, 1), -M_SQRT1_2, 1e-6);
  BOOST_CHECK_CLOSE_FRACTION(lmn(0, 2), -M_SQRT1_2, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_calc_phase) {
  SetupPredictRun(false, false, 2, 2);

  PointSourceCollection sources;
  sources.Add(predict_run->makeSource<PointSource>());

  predict_run->plan.uvw = {{100.0, 200.0, 300.0}, {100.0, 200.0, 300.0}};
  predict_run->plan.frequencies = {1e8, 2e8};

  PredictPlanExecCPU plan_exec{predict_run->plan};
  plan_exec.Precompute(sources);

  const double cinv = 2.0 * M_PI / casacore::C::c;

  // plan_exec.GetLmn()(0, 2) is already subtracted by 1.0, see
  // PredictPlanExecCPU::RaDec2Lmn. Therefore, we do not need to subtract 1.0
  // here.
  double expectedPhase = cinv * (100.0 * plan_exec.GetLmn()(0, 0) +
                                 200.0 * plan_exec.GetLmn()(0, 1) +
                                 300.0 * (plan_exec.GetLmn()(0, 2)));

  BOOST_CHECK_CLOSE(plan_exec.GetStationPhases()(0, 0), expectedPhase, 1e-6);

  const auto &freq = plan_exec.GetFrequencies();

  double phase_term_0 = expectedPhase * freq[0];
  double phase_term_1 = expectedPhase * freq[1];

  const auto &shift = plan_exec.GetShiftData();

  BOOST_CHECK_CLOSE(shift(0, 0, 0, 0), std::cos(phase_term_0), 1e-4);
  BOOST_CHECK_CLOSE(shift(0, 0, 1, 1), std::sin(phase_term_1), 1e-4);
  BOOST_CHECK_CLOSE(shift(0, 1, 0, 0), std::cos(phase_term_0), 1e-4);
  BOOST_CHECK_CLOSE(shift(0, 1, 1, 1), std::sin(phase_term_1), 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace test
} // namespace predict
