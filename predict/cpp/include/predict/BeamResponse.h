#ifndef BEAM_RESPONSE_H_
#define BEAM_RESPONSE_H_

#include "Baseline.h"
#include "common/Datastructures.h"
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

namespace predict {

class BeamResponsePlan {
public:
  BeamResponsePlan(everybeam::telescope::Telescope *telescope,
                   const double time, size_t field_id,
                   everybeam::CorrectionMode mode, bool invert = false)
      : telescope(telescope), time(time), field_id(field_id), mode(mode),
        invert(invert) {}

  BeamResponsePlan() = delete;

  void SetBaselines(const std::vector<Baseline> &baselines) {
    this->baselines = baselines;
  }
  void SetFrequencies(const std::vector<double> &frequencies) {
    this->frequencies = frequencies;
  }

  void SetFrequencies(const xt::xtensor<double, 1> &frequencies) {
    xt::adapt(this->frequencies) = frequencies;
  }

  // void ComputeBeam(std::vector<aocommon::MC2x2F> &beam_values, double ra,
  //                  double dec, const bool needs_resize = true);

  // void ComputeBeamOriginal(std::vector<aocommon::MC2x2F> &beam_values,
  //                          const everybeam::vector3r_t &srcdir,
  //                          bool needs_resize);

  // void ComputeArrayFactorOriginal(std::vector<aocommon::MC2x2F> &beam_values,
  //                                 const everybeam::vector3r_t &srcdir,
  //                                 bool needs_resize);

  // void ApplyBeamToDataAndAdd(const std::vector<Baseline> &baselines,
  //                            const xt::xtensor<double, 1> &frequencies,
  //                            const Buffer4D &direction_buffer,
  //                            Buffer4D &model_data,
  //                            const std::vector<aocommon::MC2x2F>
  //                            &beam_values);

  // void ApplyArrayFactorAndAdd(const std::vector<Baseline> &baselines,
  //                             const xt::xtensor<double, 1> &frequencies,
  //                             const Buffer4D &direction_buffer,
  //                             Buffer4D &model_data,
  //                             const std::vector<aocommon::MC2x2F>
  //                             &beam_values);

  // everybeam::telescope::Telescope *telescope;
  double time;
  size_t field_id;
  // everybeam::CorrectionMode mode;
  bool invert;
  std::vector<double> frequencies;
  std::vector<Baseline> baselines;
  // everybeam::vector3r_t srcdir;
};
} // namespace predict
#endif // BEAM_RESPONSE_H_