// Copyright (C) 2025 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache-2.0

#include "predict/Directions.h"
#include <chrono>
#include <filesystem>

#include <xtensor/xadapt.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <casacore/measures/Measures/MCDirection.h>
#include <casacore/measures/Measures/MeasConvert.h>

#include <EveryBeam/load.h>
#include <EveryBeam/options.h>

#include <predict/Baseline.h>
#include <predict/PredictPlan.h>
#include <predict/PredictPlanExecCPU.h>
#include <predict/SkyModel.h>
#include <predict/test/Common.h>
#include <predict/test/MSUtils.h>

// FIXME: currently, only the kFull correction mode is tested. We should
// implement tests for other modes as well.

#define BOOST_TEST_MODULE SKYMODEL_PLAN_TEST
#include <boost/test/unit_test.hpp>

std::string skymodel_path = "../tests/test_data/calibration_skymodel_short.txt";
std::string ms_path = "../tests/test_data/short_test.ms";

inline double ConvToAbsComplex(const Buffer4D &buffer, const size_t pol,
                               const size_t bl, const size_t channel) {
  return std::abs(std::complex<double>{buffer(pol, bl, 0, channel),
                                       buffer(pol, bl, 1, channel)});
}

BOOST_AUTO_TEST_SUITE(SKYMODEL_TESTS)

double dump_buffer(const xt::xtensor<double, 4> &buffer) {
  auto nan_count = xt::sum(xt::isnan(buffer))();
  auto inf_count = xt::sum(xt::isinf(buffer))();
  auto finite_count = xt::sum(xt::isfinite(buffer))();

  std::cout << "Buffer statistics:" << std::endl;
  std::cout << "  Total elements: " << buffer.size() << std::endl;
  std::cout << "  NaN count: " << nan_count << std::endl;
  std::cout << "  Inf count: " << inf_count << std::endl;
  std::cout << "  Finite count: " << finite_count << std::endl;

  const double buffer_sum = xt::sum(buffer)();
  std::cout << "Buffer sum: " << buffer_sum << std::endl;

  // Also show min/max of finite values
  auto min_val = xt::amin(buffer)();
  auto max_val = xt::amax(buffer)();
  std::cout << "Buffer range: [" << min_val << ", " << max_val << "]"
            << std::endl;

  return buffer_sum;
}

using namespace predict;
BOOST_AUTO_TEST_CASE(test_skymodel_and_measurement_set_exist) {
  BOOST_CHECK_MESSAGE(std::filesystem::exists(skymodel_path),
                      "Could not find skymodel file at " << skymodel_path);
  BOOST_CHECK_MESSAGE(std::filesystem::exists(ms_path),
                      "Could not find measurementset at " << ms_path);
}

BOOST_AUTO_TEST_CASE(test_predict_skymodel_with_frequency_smear) {
  GaussianSourceCollection gaussians;
  PointSourceCollection points;

  everybeam::Options options;

  auto telescope = everybeam::Load(ms_path, options);

  std::vector<double> unique_times = ReadUniqueTimes(ms_path);

  xt::xtensor<double, 2> uvw;
  xt::xtensor<double, 2> uvw_antenna;
  xt::xtensor<double, 2> chan_width;
  xt::xtensor<double, 2> chan_freq;
  xt::xtensor<double, 3> pointing;
  xt::xtensor<double, 2> antenna_position;
  std::vector<int> antenna1;
  std::vector<int> antenna2;

  ReadArrayColumn(ms_path, "UVW", uvw);
  ReadArrayColumn(ms_path + "/SPECTRAL_WINDOW", "CHAN_FREQ", chan_freq);
  ReadArrayColumn(ms_path + "/SPECTRAL_WINDOW", "CHAN_WIDTH", chan_width);
  ReadScalarColumn(ms_path, "ANTENNA1", antenna1);
  ReadScalarColumn(ms_path, "ANTENNA2", antenna2);
  ReadArrayColumn(ms_path + std::string("/ANTENNA"), "POSITION",
                  antenna_position);
  ReadArrayColumn(ms_path + "/POINTING", "DIRECTION", pointing);
  size_t n_antennas = antenna_position.shape(0);
  std::cout << "N antennas are " << n_antennas << std::endl;
  std::cout << "N antenna1 are " << antenna1.size() << std::endl;

  auto baseline_indices =
      NSetupSplitUVW(n_antennas, antenna1, antenna2, antenna_position);

  uvw_antenna.resize({n_antennas, 3});

  auto baselines = MakeBaselines(antenna1, antenna2);

  std::vector<double> channel_widths(chan_width.shape(1));
  xt::adapt(channel_widths) = xt::view(chan_width, 0, xt::all());
  std::vector<double> channel_frequencies(chan_freq.shape(1));

  NSplitUVW(baseline_indices, baselines, uvw, uvw_antenna);

  BeamResponsePlan beamplan{telescope.get(), unique_times[0], 0,
                            everybeam::CorrectionMode::kFull};

  std::cout << "Reading and parsing skymodel" << std::endl;

  ParseSkyModel(skymodel_path, gaussians, points);
  double mean_ra = pointing(0, 0, 0);
  double mean_dec = pointing(0, 0, 1);

  std::cout << "Found: " << points.beam_directions.size() << " directions"
            << std::endl;

  std::cout << "Found: " << gaussians.beam_directions.size() << " directions"
            << std::endl;

  PredictPlan plan;
  plan.baselines = baselines;
  plan.channel_widths = channel_widths;
  plan.compute_stokes_I_only = false;
  plan.correct_frequency_smearing = true;
  plan.uvw = uvw_antenna;
  plan.nbaselines = baselines.size();
  plan.nstations = n_antennas;
  plan.frequencies = xt::view(chan_freq, 0, xt::all());
  plan.nchannels = plan.frequencies.size();
  plan.nstokes = 4;
  plan.reference.ra = mean_ra;
  plan.reference.dec = mean_dec;
  plan.apply_beam = true;
  points.EvaluateSpectra(plan.frequencies);

  beamplan.SetFrequencies(plan.frequencies);
  beamplan.SetBaselines(baselines);

  xt::xtensor<double, 4, xt::layout_type::row_major> buffer(
      {plan.nstokes, plan.nbaselines, 2, plan.nchannels}, 0.0);
  PredictPlanExecCPU plan_exec(plan);

  std::cout << "Created plan " << std::endl;
  std::chrono::time_point start = std::chrono::steady_clock::now();

  casacore::MDirection::Convert meas_converter;
  Predict predict;
  Buffer4D reference_buffer;

  const std::array<computation_strategy, 3> test_strategies = {
      computation_strategy::SINGLE, computation_strategy::XSIMD};

  for (const computation_strategy strategy : test_strategies) {
    buffer.fill(0.0);
    predict.runWithStrategy(plan_exec, beamplan, points, gaussians, buffer,
                            meas_converter, strategy);
    std::cout << "Finished run with strategy: " << static_cast<int>(strategy)
              << std::endl;
    double sum = 0.0;
    sum = dump_buffer(buffer);

    if (strategy == computation_strategy::SINGLE) {
      reference_buffer = Buffer4D(buffer);
    } else {
      std::cout << "Comparing strategy " << static_cast<int>(strategy)
                << " with reference SINGLE strategy" << std::endl;

      BOOST_TEST_CONTEXT("Testing strategy: " << static_cast<int>(strategy)) {
        for (size_t bl = 0; bl < plan.nbaselines; ++bl) {
          for (size_t ch = 0; ch < plan.nchannels; ++ch) {
            for (size_t pol = 0; pol < plan.nstokes; ++pol) {
              BOOST_CHECK_CLOSE(ConvToAbsComplex(reference_buffer, pol, bl, ch),
                                ConvToAbsComplex(buffer, pol, bl, ch), 1.0e-2);
            }
          }
        }
      }
    }
    BOOST_CHECK_CLOSE(sum, -105.8834, 1e-3);
  }

  std::chrono::time_point end = std::chrono::steady_clock::now();
  std::cout << "Predict with frequency smear done in: " << std::endl;
  std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start)
            << std::endl;
}

BOOST_AUTO_TEST_CASE(test_predict_skymodel_without_frequency_smear) {
  GaussianSourceCollection gaussians;
  PointSourceCollection points;

  everybeam::Options options;

  auto telescope = everybeam::Load(ms_path, options);

  std::vector<double> unique_times = ReadUniqueTimes(ms_path);
  /*
   Get from UVW to station UVW
  */
  xt::xtensor<double, 2> uvw;
  xt::xtensor<double, 2> uvw_antenna;
  xt::xtensor<double, 2> chan_width;
  xt::xtensor<double, 2> chan_freq;
  xt::xtensor<double, 3> pointing;
  xt::xtensor<double, 2> antenna_position;
  std::vector<int> antenna1;
  std::vector<int> antenna2;

  ReadArrayColumn(ms_path, "UVW", uvw);
  ReadArrayColumn(ms_path + "/SPECTRAL_WINDOW", "CHAN_FREQ", chan_freq);
  ReadArrayColumn(ms_path + "/SPECTRAL_WINDOW", "CHAN_WIDTH", chan_width);
  ReadScalarColumn(ms_path, "ANTENNA1", antenna1);
  ReadScalarColumn(ms_path, "ANTENNA2", antenna2);
  ReadArrayColumn(ms_path + std::string("/ANTENNA"), "POSITION",
                  antenna_position);
  ReadArrayColumn(ms_path + "/POINTING", "DIRECTION", pointing);
  size_t n_antennas = antenna_position.shape(0);
  std::cout << "N antennas are " << n_antennas << std::endl;
  std::cout << "N antenna1 are " << antenna1.size() << std::endl;

  auto baseline_indices =
      NSetupSplitUVW(n_antennas, antenna1, antenna2, antenna_position);

  uvw_antenna.resize({n_antennas, 3});

  auto baselines = MakeBaselines(antenna1, antenna2);

  std::vector<double> channel_widths(chan_width.shape(1));
  xt::adapt(channel_widths) = xt::view(chan_width, 0, xt::all());
  std::vector<double> channel_frequencies(chan_freq.shape(1));

  NSplitUVW(baseline_indices, baselines, uvw, uvw_antenna);

  BeamResponsePlan beamplan{telescope.get(), unique_times[0], 0,
                            everybeam::CorrectionMode::kFull};

  std::cout << "Reading and parsing skymodel" << std::endl;

  ParseSkyModel(skymodel_path, gaussians, points);
  double mean_ra = pointing(0, 0, 0);
  double mean_dec = pointing(0, 0, 1);

  std::cout << "Found: " << points.beam_directions.size() << " directions"
            << std::endl;

  std::cout << "Found: " << gaussians.beam_directions.size() << " directions"
            << std::endl;

  PredictPlan plan;
  plan.baselines = baselines;
  plan.channel_widths = channel_widths;
  plan.compute_stokes_I_only = false;
  plan.correct_frequency_smearing = false;
  plan.uvw = uvw_antenna;
  plan.nbaselines = baselines.size();
  plan.nstations = n_antennas;
  plan.frequencies = xt::view(chan_freq, 0, xt::all());
  plan.nchannels = plan.frequencies.size();
  plan.nstokes = 4;
  plan.reference.ra = mean_ra;
  plan.reference.dec = mean_dec;
  points.EvaluateSpectra(plan.frequencies);

  beamplan.SetFrequencies(plan.frequencies);
  beamplan.SetBaselines(baselines);

  Buffer4D buffer({plan.nstokes, plan.nbaselines, 2, plan.nchannels}, 0.0);
  PredictPlanExecCPU plan_exec(plan);

  std::cout << "Created plan " << std::endl;
  std::chrono::time_point start = std::chrono::steady_clock::now();
  Predict predict;
  Buffer4D reference_buffer;

  // Placeholder meas_converter
  casacore::MDirection::Convert meas_converter;
  const std::array<computation_strategy, 3> test_strategies = {
      computation_strategy::SINGLE, computation_strategy::XSIMDD};

  for (const computation_strategy strategy : test_strategies) {
    buffer.fill(0.0);
    predict.runWithStrategy(plan_exec, beamplan, points, gaussians, buffer,
                            meas_converter, strategy);
    std::cout << "Finished run with strategy: " << static_cast<int>(strategy)
              << std::endl;
    double sum = 0.0;
    sum = dump_buffer(buffer);

    if (strategy == computation_strategy::SINGLE) {
      reference_buffer = Buffer4D(buffer);
    } else {
      std::cout << "Comparing strategy " << static_cast<int>(strategy)
                << " with reference SINGLE strategy" << std::endl;

      BOOST_TEST_CONTEXT("Testing strategy: " << static_cast<int>(strategy)) {
        for (size_t bl = 0; bl < plan.nbaselines; ++bl) {
          for (size_t ch = 0; ch < plan.nchannels; ++ch) {
            for (size_t pol = 0; pol < plan.nstokes; ++pol) {
              BOOST_CHECK_CLOSE(ConvToAbsComplex(reference_buffer, pol, bl, ch),
                                ConvToAbsComplex(buffer, pol, bl, ch), 1.0e-2);
            }
          }
        }
      }
    }
    BOOST_CHECK_CLOSE(sum, -105.9426, 1e-3);
  }

  std::chrono::time_point end = std::chrono::steady_clock::now();
  std::cout << "Predict is without frequency smearing is done in: "
            << std::endl;
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
            << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()