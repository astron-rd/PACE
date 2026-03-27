// PredictPlan.h: PredictPlan
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief PredictPlan

#ifndef PREDICT_PLAN_H_
#define PREDICT_PLAN_H_

#include <cassert>
#include <complex>

#include "Baseline.h"
#include "Direction.h"
#include <xtensor/xlayout.hpp>
#include <xtensor/xtensor.hpp>

#include "GaussianSourceCollection.h"
#include "PointSourceCollection.h"
#include "common/Datastructures.h"

namespace predict {

typedef std::complex<double> dcomplex;

struct PredictPlan {
  enum StokesComponent : uint_fast8_t { I = 0, Q = 1, U = 2, V = 3 };

  Direction reference;

  std::vector<Baseline> baselines;
  std::vector<double> channel_widths;
  xt::xtensor<double, 2, xt::layout_type::column_major> uvw;
  xt::xtensor<double, 1> frequencies;
  size_t nchannels = 0;
  size_t nstations = 0;
  size_t nbaselines = 0;
  size_t nstokes = 0;

  bool correct_frequency_smearing = false;
  bool compute_stokes_I_only = false;
  bool apply_beam = true;
  bool parallelize_over_sources = true;
  bool parallelize_over_beams = false;

  virtual ~PredictPlan() = default;

  virtual void Precompute(const PointSourceCollection &) {}
  virtual void Precompute(const GaussianSourceCollection &) {}

  virtual void Compute(const PointSourceCollection &sources, Buffer4D &buffer) {
  }

  virtual void Compute(const GaussianSourceCollection &sources,
                       Buffer4D &buffer) {}

  virtual void Verify() final {
    if (is_plan_valid_)
      return;

    if (nchannels == 0) {
      throw std::invalid_argument("PredictPlan::Verify(): nchannels == 0");
    }

    if (nstations == 0) {
      throw std::invalid_argument("PredictPlan::Verify(): nstations == 0");
    }

    if (nbaselines == 0) {
      throw std::invalid_argument("PredictPlan::Verify(): nbaselines == 0");
    }

    if (nstokes == 0) {
      throw std::invalid_argument("PredictPlan::Verify(): nstokes == 0");
    }

    if (frequencies.shape(0) != nchannels) {
      throw std::invalid_argument(
          "PredictPlan::Verify(): frequencies.shape(0) != nchannels");
    }

    if (uvw.shape(0) != nstations && uvw.shape(1) != 3) {
      throw std::invalid_argument("PredictPlan::Verify(): uvw.shape(0) != "
                                  "nstations && uvw.shape(1) != 3");
    }

    if (baselines.size() != nbaselines) {
      throw std::invalid_argument(
          "PredictPlan::Verify(): baselines.size() != nbaselines");
    }

    if (channel_widths.size() != nchannels) {
      throw std::invalid_argument(
          "PredictPlan::Verify(): channel_widths.size() != nchannels");
    }

    if (compute_stokes_I_only && nstokes != 1) {
      throw std::invalid_argument(
          "PredictPlan::Verify(): compute_stokes_I_only && nstokes != 1");
    }

    if (!compute_stokes_I_only && nstokes != 4) {
      throw std::invalid_argument(
          "PredictPlan::Verify(): !compute_stokes_I_only && nstokes != 4");
    }

    is_plan_valid_ = true;
  }

  virtual void ValidateBeforeComputation(const PointSourceCollection &sources,
                                         const Buffer4D &buffer) final {
    if (nstations == 0) {
      throw std::invalid_argument("PredictPlan::ValidateBeforeComputation(): "
                                  "nstations == 0");
    }
    if (nchannels == 0) {
      throw std::invalid_argument("PredictPlan::ValidateBeforeComputation(): "
                                  "nchannels == 0");
    }
    if (baselines.empty()) {
      throw std::invalid_argument("PredictPlan::ValidateBeforeComputation(): "
                                  "baselines.empty()");
    }

    if (buffer.shape().size() != 4) {
      throw std::invalid_argument(
          "PredictPlan::ValidateBeforeComputation(): buffer.shape().size() != "
          "4");
    }

    if (buffer.shape(0) != nstokes) {
      throw std::invalid_argument(
          "PredictPlan::ValidateBeforeComputation(): buffer.shape(0) != "
          "nstokes");
    }
    if (buffer.shape(1) != nbaselines) {
      throw std::invalid_argument("PredictPlan::ValidateBeforeComputation(): "
                                  "buffer.shape(1) != nbaselines");
    }
    if (buffer.shape(2) != 2) {
      throw std::invalid_argument(
          "PredictPlan::ValidateBeforeComputation(): buffer.shape(2) != "
          "2 {imag/real}");
    }
    if (buffer.shape(3) != nchannels) {
      throw std::invalid_argument(
          "PredictPlan::ValidateBeforeComputation(): buffer.shape(3) != "
          "nchannels");
    }
    if (sources.evaluated_spectra.shape(0) != 4) {
      throw std::invalid_argument(
          "PredictPlan::ValidateBeforeComputation(): sources.evaluated_spectra"
          ".shape(0) != 4");
    }
    if (sources.evaluated_spectra.shape(1) != sources.Size()) {
      throw std::invalid_argument(
          "PredictPlan::ValidateBeforeComputation(): sources.evaluated_spectra"
          ".shape(1) != sources.Size()");
    }
    if (sources.evaluated_spectra.shape(2) != 2) {
      throw std::invalid_argument(
          "PredictPlan::ValidateBeforeComputation(): sources.evaluated_spectra"
          ".shape(2) != 2");
    }
    if (sources.evaluated_spectra.shape(3) != nchannels) {
      throw std::invalid_argument(
          "PredictPlan::ValidateBeforeComputation(): "
          "sources.evaluated_spectra.shape(3) != nchannels");
    }
    if (uvw.shape(0) != nstations) {
      throw std::invalid_argument("PredictPlan::ValidateBeforeComputation(): "
                                  "uvw.shape(0) != nstations");
    }
    if (uvw.shape(1) != 3) {
      throw std::invalid_argument("PredictPlan::ValidateBeforeComputation(): "
                                  "uvw.shape(1) != 3");
    }
  }

  virtual void
  ValidateBeforeComputation(const GaussianSourceCollection &sources,
                            const Buffer4D &buffer) final {
    ValidateBeforeComputation(
        static_cast<const PointSourceCollection &>(sources), buffer);
    // Add any additional checks specific to GaussianSourceCollection here.
  }

private:
  bool is_plan_valid_ = false;
};
} // namespace predict
#endif // PREDICT_PLAN_H_
