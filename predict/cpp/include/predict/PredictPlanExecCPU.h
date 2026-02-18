// PredictPlanExecCPU.h: PredictPlanExecCPU
//
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: Apache2.0

/// \file
/// \brief PredictPlanExecCPU

#ifndef PREDICT_PLAN_EXEC_CPU_H_
#define PREDICT_PLAN_EXEC_CPU_H_

#include <omp.h>
#include <xtensor/xlayout.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "Directions.h"
#include "GaussianSourceCollection.h"
#include "PointSourceCollection.h"
#include "PredictPlan.h"
#include "common/Datastructures.h"

#include <trigdx/lookup_xsimd.hpp>
#include <trigdx/reference.hpp>

namespace predict {

class PredictPlanExecCPU : public PredictPlan {
public:
  explicit PredictPlanExecCPU(const PredictPlan &plan,
                              const int num_threads = omp_get_max_threads())
      : PredictPlan(plan), compute_pol_dispatch_(*this),
        compute_pol_gaussian_dispatch_(*this), max_num_threads_(num_threads) {
    // Initialize the station phases and shifts
    Initialize();
    SetNumThreads(num_threads);
  }

  virtual void Precompute(const PointSourceCollection &sources) final override;
  virtual void
  Precompute(const GaussianSourceCollection &sources) final override;

  virtual void Compute(const PointSourceCollection &sources,
                       Buffer4D &buffer) final override;

  virtual void Compute(const GaussianSourceCollection &sources,
                       Buffer4D &buffer) final override;

  void ComputeWithTarget(const PointSourceCollection &sources, Buffer4D &buffer,
                         const computation_strategy strat);

  void ComputeWithTarget(const GaussianSourceCollection &sources,
                         Buffer4D &buffer, const computation_strategy strat);

  void ComputeStationPhases(const bool resize_shift_data = false);

  const xt::xtensor<double, 2> &GetStationPhases() const {
    return station_phases;
  }

  const xt::xtensor<float, 4, xt::layout_type::row_major> &
  GetShiftData() const {
    return shift_data;
  }

  void FillEulerMatrix(xt::xtensor<double, 2> &mat, const double ra,
                       const double dec);

  const xt::xtensor<double, 2> &GetLmn() const { return lmn; }
  const xt::xtensor<double, 2, xt::layout_type::column_major> &GetUvw() const {
    return uvw;
  }
  const xt::xtensor<double, 1> &GetFrequencies() const { return frequencies; }

  void SetNumThreads(int num_threads);
  int GetNumOMPThreads() const;
  int GetMaxNumThreads() const;

  template <class E, class S, class Tag, class Arch>
  void operator()(Arch, E &buffer_expr, const S &spectrum, Tag);

  template <bool DO_ANGULAR_CORRECTION> struct ComputePolarizationDispatch {
    template <class E, class S, class Tag, class Arch>
    void operator()(Arch, E &buffer_expr, const S &spectrum, Tag) const;

    const PredictPlanExecCPU &parent;

    explicit ComputePolarizationDispatch(const PredictPlanExecCPU &parent)
        : parent(parent) {}
  };

  ComputePolarizationDispatch<false> compute_pol_dispatch_;
  ComputePolarizationDispatch<true> compute_pol_gaussian_dispatch_;

  xt::xtensor<double, 2> lmn;
  xt::xtensor<double, 2> station_phases;
  xt::xtensor<float, 4, xt::layout_type::row_major> shift_data;
  xt::xtensor<float, 3, xt::layout_type::row_major> angular_smear_terms;
  xt::xtensor<float, 1> channel_widths_floats;

  struct SmearTermsDispatch {
    template <class E, class S, class Tag, class Arch>
    void operator()(Arch, const xt::xtensor<float, 1> &channel_widths,
                    const E &station_phase_p, const E &station_phase_q,
                    S &smear_terms_buffer, const float phase_diff, Tag) const;
  };

  SmearTermsDispatch smear_terms_dispatch_;

  PredictPlanExecCPU(const PredictPlanExecCPU &other)
      : PredictPlan(static_cast<const PredictPlan &>(other)),
        compute_pol_dispatch_(*this), compute_pol_gaussian_dispatch_(*this),
        lmn(other.lmn), station_phases(other.station_phases),
        shift_data(other.shift_data),
        angular_smear_terms(other.angular_smear_terms),
        channel_widths_floats(other.channel_widths_floats),
        smear_terms_dispatch_(other.smear_terms_dispatch_),
        max_num_threads_(other.max_num_threads_),
        trig_lookup_table_(other.trig_lookup_table_) {
    // No need to call Initialize() or SetNumThreads() here,
    // as we are copying all relevant data members.
  }

private:
  void Initialize();

  template <class E, class S>
  void ComputeSmearTermsSingle(const E &station_phases_p,
                               const E &station_phases_q, S &smear_terms,
                               const float phase_diff) const;

  void ComputeAngularSmearTerms(const GaussianSourceCollection &sources);

  template <computation_strategy T = computation_strategy::XSIMD, class E,
            class S>
  inline constexpr void ProcessPolarizationComponents(E &stoke_buffer_expr,
                                                      const S &stoke_spectrum) {
    if constexpr (T == computation_strategy::SINGLE) {
      ProcessPolarizationComponentSingle<E, S, false>(stoke_buffer_expr,
                                                      stoke_spectrum);
    } else if constexpr (T == computation_strategy::XSIMD) {
      ProcessPolarizationComponentXtensor<E, S, false>(stoke_buffer_expr,
                                                       stoke_spectrum);
    } else {
      static_assert(T == computation_strategy::SINGLE ||
                    T == computation_strategy::XSIMD ||
                    "Unknown computation strategy");
    }
  }

  template <computation_strategy T = computation_strategy::XSIMD, class E,
            class S>
  inline constexpr void
  ProcessPolarizationComponentsGaussian(E &stoke_buffer_expr,
                                        const S &stoke_spectrum) {
    if constexpr (T == computation_strategy::SINGLE) {
      ProcessPolarizationComponentSingle<E, S, true>(stoke_buffer_expr,
                                                     stoke_spectrum);
    } else if constexpr (T == computation_strategy::XSIMD) {
      ProcessPolarizationComponentXtensor<E, S, true>(stoke_buffer_expr,
                                                      stoke_spectrum);
    } else {
      static_assert(T == computation_strategy::SINGLE ||
                    T == computation_strategy::XSIMD ||
                    "Unknown computation strategy");
    }
  }

  template <class E, class S>
  inline void ProcessPolarizationComponents(const computation_strategy T,
                                            E &buffer_expr, const S &spectrum) {
#ifdef ENABLE_TRACY_PROFILING
    ZoneScoped;
#endif
    switch (T) {
    case computation_strategy::SINGLE:
      ProcessPolarizationComponentSingle<E, S, false>(buffer_expr, spectrum);
      break;
    case computation_strategy::XSIMD:
      ProcessPolarizationComponentXtensor<E, S, false>(buffer_expr, spectrum);
      break;
    default:
      break;
    }
  }

  template <class E, class S>
  inline void
  ProcessPolarizationComponentsGaussian(const computation_strategy T,
                                        E &buffer_expr, const S &spectrum) {
#ifdef ENABLE_TRACY_PROFILING
    ZoneScoped;
#endif
    switch (T) {
    case computation_strategy::SINGLE:
      ProcessPolarizationComponentSingle<E, S, true>(buffer_expr, spectrum);
      break;
    case computation_strategy::XSIMD:
      ProcessPolarizationComponentXtensor<E, S, true>(buffer_expr, spectrum);
      break;
    default:
      break;
    }
  }

  template <class E, class S, bool DO_ANGULAR_CORRECTION>
  void ProcessPolarizationComponentSingle(E &stoke_buffer_expr,
                                          const S &stoke_spectrum);

  template <class E, class S, bool DO_ANGULAR_CORRECTION>
  void ProcessPolarizationComponentXtensor(E &stoke_buffer_expr,
                                           const S &stoke_spectrum);

private:
  const double kCInv_ = 2.0 * M_PI / casacore::C::c;
  const double kFwhm2Sigma = 1.0 / (2.0 * std::sqrt(2.0 * std::log(2.0)));
  const double kInvCSqr = 1.0 / (casacore::C::c * casacore::C::c);

  const int max_num_threads_;

  ReferenceBackend trig_lookup_table_;
};

template <class E, class S, class Tag, class Arch>
void PredictPlanExecCPU::SmearTermsDispatch::operator()(
    Arch, const xt::xtensor<float, 1> &channel_widths, const E &station_phase_p,
    const E &station_phase_q, S &smear_terms_buffer, const float phase_diff,
    Tag) const {
  const float *const channel_width_ptr = &(channel_widths.unchecked(0));

  using b_type_float = xsimd::batch<float, Arch>;
  size_t inc = b_type_float::size;
  size_t size = channel_widths.size();
  // size for which the vectorization is possible
  size_t vec_size = size - size % inc;

  const b_type_float ones = b_type_float::broadcast(1.0f);
  const b_type_float half = b_type_float::broadcast(0.5f);

  float *const smear_terms_buffer_ptr = &(smear_terms_buffer.unchecked(0));

  if (fabs(phase_diff) > 1.e-6) [[likely]] {
    size_t ch_i = 0;
    for (; ch_i < vec_size; ch_i += inc) {
      b_type_float ch_vec = b_type_float::load(channel_width_ptr + ch_i, Tag());
      const b_type_float shift =
          xsimd::mul(xsimd::mul(ch_vec, phase_diff), half);
      const b_type_float smear_term =
          xsimd::abs(xsimd::div(xsimd::sin(shift), shift));
      xsimd::store(smear_terms_buffer_ptr + ch_i, smear_term, Tag());
    }
    for (; ch_i < size; ++ch_i) {
      const float shift = phase_diff * channel_widths[ch_i] * 0.5f;

      if (shift == 0.0f) {
        *(smear_terms_buffer_ptr + ch_i) = 1.0;
      } else {
        *(smear_terms_buffer_ptr + ch_i) = std::fabs(sinf(shift) / shift);
      }
    }
  } else {
    size_t ch_i = 0;
    for (; ch_i < vec_size; ch_i += inc) {
      xsimd::store(smear_terms_buffer_ptr + ch_i, ones, Tag());
    }
    for (; ch_i < size; ++ch_i) {
      *(smear_terms_buffer_ptr + ch_i) = 1.0;
    }
  }
}

template <class E, class S>
void PredictPlanExecCPU::ComputeSmearTermsSingle(const E &station_phases_p,
                                                 const E &station_phases_q,
                                                 S &smear_terms,
                                                 const float phase_diff) const {
  xsimd::dispatch(smear_terms_dispatch_)(
      channel_widths_floats, station_phases_p, station_phases_q, smear_terms,
      phase_diff, xsimd::unaligned_mode());
}

} // namespace predict

#endif // PREDICT_PLAN_EXEC_CPU_H_
