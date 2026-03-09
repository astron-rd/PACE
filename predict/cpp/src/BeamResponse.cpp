// Routines to compute and apply the beams
// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// @author Tammo Jan Dijkema

// ApplyBeam is templated on the type of the data, could be complex<double>
// or complex<float>

#include <predict/BeamResponse.h>
#include <span>

#include <predict/Spectrum.h>
#include <predict/common/Datastructures.h>

#ifdef ENABLE_TRACY_PROFILING
#include <tracy/Tracy.hpp>
#endif
#include <xtensor/xcomplex.hpp>
#include <xtensor/xview.hpp>

#include <predict/Baseline.h>

namespace predict {

// void ApplyWeights(const aocommon::MC2x2F &gain_a,
//                   const aocommon::MC2x2F &gain_b, float *weight) {
//   float cov[4], normGainA[4], normGainB[4];
//   for (unsigned int i = 0; i < 4; ++i) {
//     cov[i] = 1.f / weight[i];
//     normGainA[i] = std::norm(gain_a.Get(i));
//     normGainB[i] = std::norm(gain_b.Get(i));
//   }

//   weight[0] = cov[0] * (normGainA[0] * normGainB[0]) +
//               cov[1] * (normGainA[0] * normGainB[1]) +
//               cov[2] * (normGainA[1] * normGainB[0]) +
//               cov[3] * (normGainA[1] * normGainB[1]);
//   weight[0] = 1.f / weight[0];

//   weight[1] = cov[0] * (normGainA[0] * normGainB[2]) +
//               cov[1] * (normGainA[0] * normGainB[3]) +
//               cov[2] * (normGainA[1] * normGainB[2]) +
//               cov[3] * (normGainA[1] * normGainB[3]);
//   weight[1] = 1.f / weight[1];

//   weight[2] = cov[0] * (normGainA[2] * normGainB[0]) +
//               cov[1] * (normGainA[2] * normGainB[1]) +
//               cov[2] * (normGainA[3] * normGainB[0]) +
//               cov[3] * (normGainA[3] * normGainB[1]);
//   weight[2] = 1.f / weight[2];

//   weight[3] = cov[0] * (normGainA[2] * normGainB[2]) +
//               cov[1] * (normGainA[2] * normGainB[3]) +
//               cov[2] * (normGainA[3] * normGainB[2]) +
//               cov[3] * (normGainA[3] * normGainB[3]);
//   weight[3] = 1.f / weight[3];
// }

// template <typename T>
// void ApplyBeamToData(std::vector<Baseline> baselines, const size_t
// n_channels,
//                      const size_t n_baselines, const size_t n_stations,
//                      T *data0, float *weight0, aocommon::MC2x2 *beam_values,
//                      bool doUpdateWeights) {
//   /*
//     Applies the beam to each baseline and each frequency of the
//     model patch
//   */

//   // Apply beam for channel ch on all baselines
//   // For mode=ARRAY_FACTOR, too much work is done here
//   // because we know that r and l are diagonal
//   for (size_t bl = 0; bl < n_baselines; ++bl) {
//     // If the beam is the same for all stations (i.e. when n_stations = 1),
//     // all baselines will have the same beam values
//     size_t index_left =
//         (n_stations == 1 ? 0 : n_channels * baselines[bl].first);
//     size_t index_right =
//         (n_stations == 1 ? 0 : n_channels * baselines[bl].second);
//     for (size_t ch = 0; ch < n_channels; ++ch) {
//       T *data = data0 + bl * 4 * n_channels + ch * 4;
//       const aocommon::MC2x2F mat(data);

//       const aocommon::MC2x2F left(beam_values[index_left + ch]);
//       const aocommon::MC2x2F right(beam_values[index_right + ch]);
//       const aocommon::MC2x2F result = left * mat.MultiplyHerm(right);
//       result.AssignTo(data);
//       if (doUpdateWeights) {
//         ApplyWeights(left, right, weight0 + bl * 4 * n_channels + ch * 4);
//       }
//     }
//   }
// }

std::vector<size_t>
UniqueStationIndices(const std::vector<Baseline> &baselines) {
  /*
      Returns a vector of unique station indices from the baselines
  */
  std::set<size_t> unique_stations;
  for (const auto &bl : baselines) {
    unique_stations.insert(bl.first);
    unique_stations.insert(bl.second);
  }
  return std::vector<size_t>(unique_stations.begin(), unique_stations.end());
}

// void BeamResponsePlan::ComputeBeamOriginal(
//     std::vector<aocommon::MC2x2F> &beam_values,
//     const everybeam::vector3r_t &srcdir, bool needs_resize) {
//   /*
//     Compute the beam values for each station in a specific direction
//     and store them into beam_values

//     For convenience it returns the number of stations the beam
//     was computed for.
//   */
// #ifdef ENABLE_TRACY_PROFILING
//   ZoneScoped;
// #endif
//   std::unique_ptr<everybeam::pointresponse::PointResponse> point_response =
//       telescope->GetPointResponse(time);

//   const std::vector<size_t> station_indices =
//   UniqueStationIndices(baselines); const size_t n_stations =
//   station_indices.size(); const size_t n_channels = frequencies.size(); const
//   std::span<double> frequency_span(frequencies.data(),
//                                          frequencies.size());
//   if (needs_resize)
//     beam_values.resize(n_stations * n_channels);
//     // Apply the beam values of both stations to the ApplyBeamed data.

// #ifdef PREDICT_DIAGNOSTICS
//   std::cout << "new beam direction: " << srcdir[0] << ", " << srcdir[1] << ",
//   "
//             << srcdir[2] << std::endl;
// #endif
//   for (size_t ch = 0; ch < n_channels; ++ch) {
//     switch (mode) {
//     case everybeam::CorrectionMode::kFull:
//     case everybeam::CorrectionMode::kElement:
//       // Fill beam_values for channel ch
//       for (size_t st = 0; st < n_stations; ++st) {
//         beam_values[n_channels * st + ch] =
//             static_cast<aocommon::MC2x2F>(point_response->Response(
//                 mode, station_indices[st], frequency_span[ch], srcdir));
//         if (invert) {
//           // Terminate if the matrix is not invertible.
//           [[maybe_unused]] bool status =
//               beam_values[n_channels * st + ch].Invert();
//           assert(status);
//         }
//       }
//       break;
//     case everybeam::CorrectionMode::kArrayFactor: {
//       aocommon::MC2x2F af_tmp;
//       for (size_t st = 0; st < n_stations; ++st) {
//         af_tmp = static_cast<aocommon::MC2x2F>(point_response->Response(
//             mode, station_indices[st], frequency_span[ch], srcdir));

//         if (invert) {
//           af_tmp = aocommon::MC2x2F(1.0f / af_tmp.Get(0), 0.0f, 0.0f,
//                                     1.0f / af_tmp.Get(3));
//         }
//         beam_values[n_channels * st + ch] = af_tmp;
//       }
//       break;
//     }
//     case everybeam::CorrectionMode::kNone: // this should not happen
//       for (size_t st = 0; st < n_stations; ++st) {
//         beam_values[n_channels * st + ch] = aocommon::MC2x2F::Unity();
//       }
//       break;
//     }
//   }
// }

// void BeamResponsePlan::ComputeArrayFactorOriginal(
//     std::vector<aocommon::MC2x2F> &beam_values,
//     const everybeam::vector3r_t &srcdir, bool needs_resize) {
// #ifdef ENABLE_TRACY_PROFILING
//   ZoneScoped;
// #endif
//   std::unique_ptr<everybeam::pointresponse::PointResponse> point_response =
//       telescope->GetPointResponse(time);

//   const std::vector<size_t> station_indices =
//   UniqueStationIndices(baselines); const size_t n_stations =
//   station_indices.size(); const size_t n_channels = frequencies.size(); const
//   std::span<double> frequency_span(frequencies.data(),
//                                          frequencies.size());
//   if (needs_resize)
//     beam_values.resize(n_stations * n_channels);

//   for (size_t ch = 0; ch < n_channels; ++ch) {
//     for (size_t st = 0; st < n_stations; ++st) {
//       auto &beam_value = beam_values[n_channels * st + ch];

//       beam_value = static_cast<aocommon::MC2x2F>(point_response->Response(
//           everybeam::CorrectionMode::kArrayFactor, station_indices[st],
//           frequency_span[ch], srcdir));

//       if (invert) {
//         beam_value.Set(0, 1.0f / beam_value.Get(0));
//       }
//     }
//   }
// }

// void BeamResponsePlan::ComputeBeam(std::vector<aocommon::MC2x2F>
// &beam_values,
//                                    double ra, double dec, bool needs_resize)
//                                    {
//   /*
//     Compute the beam values for each station in a specific direction
//     and store them into beam_values

//     For convenience it returns the number of stations the beam
//     was computed for.
//   */
// #ifdef ENABLE_TRACY_PROFILING
//   ZoneScoped;
// #endif
//   std::unique_ptr<everybeam::pointresponse::PointResponse> point_response =
//       telescope->GetPointResponse(time);

//   const std::vector<size_t> station_indices =
//   UniqueStationIndices(baselines); const size_t n_stations =
//   station_indices.size(); const size_t n_channels = frequencies.size(); const
//   std::span<double> frequency_span(frequencies.data(),
//                                          frequencies.size());
//   if (needs_resize)
//     beam_values.resize(n_stations * n_channels);
//   // Apply the beam values of both stations to the ApplyBeamed data.

//   for (const size_t station_id : station_indices) {
//     aocommon::MC2x2F &beam_response = beam_values[station_id * n_channels];

//     switch (mode) {
//     case everybeam::CorrectionMode::kFull:
//     case everybeam::CorrectionMode::kElement: {
//       point_response->Response(mode, &beam_response, ra, dec, frequency_span,
//                                station_id, field_id);

//       if (invert) {
//         for (size_t ch = 0; ch < n_channels; ++ch) {
//           // Terminate if the matrix is not invertible.
//           [[maybe_unused]] bool status =
//               beam_values[station_id * n_channels + ch].Invert();
//           assert(status);
//         }
//       }
//       break;
//     }
//     case everybeam::CorrectionMode::kArrayFactor: {
//       point_response->Response(mode, &beam_response, ra, dec, frequency_span,
//                                station_id, field_id);

//       if (invert) {
//         beam_response = aocommon::MC2x2F(1.0f / beam_response.Get(0), 0.0f,
//                                          0.0f, 1.0f / beam_response.Get(3));
//       }
//       break;
//     }
//     case everybeam::CorrectionMode::kNone: {
//       beam_response = aocommon::MC2x2F::Unity();
//       break;
//     }
//     }
//   }
// }

// void BeamResponsePlan::ApplyBeamToDataAndAdd(
//     const std::vector<Baseline> &baselines,
//     const xt::xtensor<double, 1> &frequencies, const Buffer4D
//     &direction_buffer, Buffer4D &model_data, const
//     std::vector<aocommon::MC2x2F> &beam_values) {
//   /*
//     Applies the beam to each baseline and each frequency of the
//     model patch and sum the contribution to the model data
//   */
// #ifdef ENABLE_TRACY_PROFILING
//   ZoneScoped;
// #endif
//   const size_t n_channels = frequencies.size();
//   const size_t n_baselines = baselines.size();

//   // Apply beam for channel ch on all baselines
//   // For mode=ARRAY_FACTOR, too much work is done here
//   // because we know that r and l are diagonal
//   xt::xtensor<std::complex<float>, 2> mat({2, 2});
//   for (size_t bl = 0; bl < n_baselines; ++bl) {
//     // If the beam is the same for all stations (i.e. when n_stations = 1),
//     // all baselines will have the same beam values
//     const size_t index_left = n_channels * baselines[bl].first;
//     const size_t index_right = n_channels * baselines[bl].second;
//     for (size_t ch = 0; ch < n_channels; ++ch) {
//       const aocommon::MC2x2F left = beam_values[index_left + ch];
//       const aocommon::MC2x2F right = beam_values[index_right + ch];
//       const aocommon::MC2x2F direction_vis = aocommon::MC2x2F(
//           std::complex<float>(
//               static_cast<float>(direction_buffer(0, bl, Complex::REAL, ch)),
//               static_cast<float>(direction_buffer(0, bl, Complex::IMAG,
//               ch))),
//           std::complex<float>(
//               static_cast<float>(direction_buffer(1, bl, Complex::REAL, ch)),
//               static_cast<float>(direction_buffer(1, bl, Complex::IMAG,
//               ch))),
//           std::complex<float>(
//               static_cast<float>(direction_buffer(2, bl, Complex::REAL, ch)),
//               static_cast<float>(direction_buffer(2, bl, Complex::IMAG,
//               ch))),
//           std::complex<float>(
//               static_cast<float>(direction_buffer(3, bl, Complex::REAL, ch)),
//               static_cast<float>(direction_buffer(3, bl, Complex::IMAG,
//               ch))));

//       const aocommon::MC2x2F result = left *
//       direction_vis.MultiplyHerm(right); result.AssignTo(mat.data());

//       xt::view(model_data, 0, bl, Complex::REAL, ch) += xt::real(mat(0, 0));
//       xt::view(model_data, 1, bl, Complex::REAL, ch) += xt::real(mat(0, 1));
//       xt::view(model_data, 2, bl, Complex::REAL, ch) += xt::real(mat(1, 0));
//       xt::view(model_data, 3, bl, Complex::REAL, ch) += xt::real(mat(1, 1));

//       xt::view(model_data, 0, bl, Complex::IMAG, ch) += xt::imag(mat(0, 0));
//       xt::view(model_data, 1, bl, Complex::IMAG, ch) += xt::imag(mat(0, 1));
//       xt::view(model_data, 2, bl, Complex::IMAG, ch) += xt::imag(mat(1, 0));
//       xt::view(model_data, 3, bl, Complex::IMAG, ch) += xt::imag(mat(1, 1));
//     }
//   }
// }

// void BeamResponsePlan::ApplyArrayFactorAndAdd(
//     const std::vector<Baseline> &baselines,
//     const xt::xtensor<double, 1> &frequencies, const Buffer4D
//     &direction_buffer, Buffer4D &model_data, const
//     std::vector<aocommon::MC2x2F> &beam_values) {
// #ifdef ENABLE_TRACY_PROFILING
//   ZoneScoped;
// #endif
//   const size_t n_channels = frequencies.size();
//   const size_t n_baselines = baselines.size();

//   for (size_t bl = 0; bl < n_baselines; ++bl) {
//     const size_t index_left = n_channels * baselines[bl].first;
//     const size_t index_right = n_channels * baselines[bl].second;
//     for (size_t ch = 0; ch < n_channels; ++ch) {
//       const std::complex<double> &left = beam_values[index_left + ch].Get(0);
//       const std::complex<double> &right = beam_values[index_right +
//       ch].Get(0); const std::complex<double> res =
//           (left *
//            std::complex<double>{direction_buffer(0, bl, Complex::REAL, ch),
//                                 direction_buffer(0, bl, Complex::IMAG, ch)})
//                                 *
//           std::conj(right);

//       xt::view(model_data, 0, bl, Complex::REAL, ch) += xt::real(res);
//       xt::view(model_data, 0, bl, Complex::IMAG, ch) += xt::imag(res);
//     }
//   }
// }

} // namespace predict