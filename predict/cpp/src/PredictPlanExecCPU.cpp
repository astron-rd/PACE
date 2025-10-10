#include <cstddef>
#include <predict/PredictPlanExecCPU.h>

#include <immintrin.h>
#include <omp.h>

#include <cstdint>
#include <predict/Directions.h>
#include <predict/GaussianSourceCollection.h>
#include <predict/PointSourceCollection.h>
#include <predict/PredictPlan.h>
#include <predict/Spectrum.h>
#include <stdexcept>
#include <sys/types.h>
#include <xtensor/xbuilder.hpp>

#ifdef ENABLE_TRACY_PROFILING
#include <tracy/Tracy.hpp>
#endif

#include <cmath>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xlayout.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xtensor_forward.hpp>
#include <xtensor/xview.hpp>

namespace predict {

void PredictPlanExecCPU::Initialize() {
  // Verify the plan correctness.
  this->Verify();

  // Compute the trigonometric lookup table.
  trig_lookup_table_.init();
}

void PredictPlanExecCPU::Precompute(const PointSourceCollection &sources) {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  if (!sources.Size())
    return;

  sources.direction_vector.RaDec2Lmn(reference, lmn);
  // smear_terms = xt::ones<float>({sources.Size(), nchannels});
  shift_data.resize({sources.Size(), nstations, 2, nchannels});

  channel_widths_floats = xt::adapt(channel_widths);
  ComputeStationPhases();
}

void PredictPlanExecCPU::Precompute(const GaussianSourceCollection &sources) {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  if (!sources.Size())
    return;

  sources.direction_vector.RaDec2Lmn(reference, lmn);
  // smear_terms = xt::ones<float>({sources.Size(), nchannels});
  angular_smear_terms.resize({sources.Size(), nbaselines, nchannels});
  shift_data.resize({sources.Size(), nstations, 2, nchannels});

  channel_widths_floats = xt::adapt(channel_widths);
  ComputeStationPhases();
  ComputeAngularSmearTerms(sources);
}

/**
 * Compute station phase shifts.
 *
 * \f[ \mathrm{stationphases}(p) = \frac{2\pi}{c}((u_p, v_p, w_p) \cdot (\ell,
 * m, n)) \f]
 *
 * \f[ \mathrm{phases}(p) = e^{\mathrm{stationphases}(p)} \f]
 *
 * @param nStation Number of stations
 * @param nChannel Number of channels
 * @param lmn LMN coordinates of all sources.
 * @param uvw Station UVW coordinates, matrix of shape (3, nSt)
 * @param freq Channel frequencies, should be length nChannel
 * @param shift Output matrix (2 for real,imag), shift per station, matrix of
 * shape (3, nSt)
 * @param stationPhases Output vector, store per station \f$(x_1,y_1)\f$
 */
void PredictPlanExecCPU::ComputeStationPhases(const bool resize_shift_data) {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  // Compute station phases
  const size_t nsources = lmn.shape(0);

  if (resize_shift_data) {
    shift_data.resize({nsources, nstations, 2, nchannels});
  }

  station_phases.resize({uvw.shape(0), lmn.shape(0)});
  xt::view(lmn, xt::all(), 2) -= 1.0;

  const int nthreads = std::min(GetMaxNumThreads(), static_cast<int>(nsources));
#pragma omp parallel if (parallelize_over_sources) num_threads(nthreads)
  {
    xt::xtensor<double, 1> phase_terms(std::array<size_t, 1>{nchannels});
    xt::xtensor<float, 1> sin_phase(std::array<size_t, 1>{nchannels});
    xt::xtensor<float, 1> cos_phase(std::array<size_t, 1>{nchannels});

#pragma omp for schedule(static)
    for (uint_fast32_t s = 0; s < nsources; ++s) {
      for (uint_fast32_t st = 0; st < nstations; ++st) {
        xt::view(station_phases, st, s) = xt::sum(
            xt::view(lmn, s, xt::all()) * xt::view(uvw, st, xt::all()) * kCInv_,
            0);
        phase_terms = station_phases(st, s) * frequencies;

        // // FIXME: directly write to shift_data instead of using sin_phase and
        // // cos_phase as intermediate storage.
        // trig_lookup_table_.compute_sincosf(nchannels, phase_terms.data(),
        //                                       sin_phase.data(),
        //                                       cos_phase.data());

        // xt::view(shift_data, s, st, 0, xt::all()) = cos_phase;
        // xt::view(shift_data, s, st, 1, xt::all()) = sin_phase;

        const auto sin_phase = xt::sin(phase_terms);
        const auto cos_phase = xt::cos(phase_terms);

        // Assign to shifts as std::complex
        for (size_t ch = 0; ch < nchannels; ++ch) {
          shift_data.unchecked(s, st, 0, ch) = cos_phase(ch);
          shift_data.unchecked(s, st, 1, ch) = sin_phase(ch);
        }
      }
    }
  }
}

void PredictPlanExecCPU::ComputeAngularSmearTerms(
    const GaussianSourceCollection &sources) {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  const std::array<size_t, 2> kStationUvwShape = {GetUvw().shape(1),
                                                  GetUvw().shape(0)};
  xt::xtensor<double, 2> xtStationUvw = xt::adapt(
      GetUvw().data(), GetUvw().size(), xt::no_ownership(), kStationUvwShape);

  xt::xtensor<double, 2> euler_matrix_phasecenter = xt::zeros<double>({3, 3});
  FillEulerMatrix(euler_matrix_phasecenter, reference.ra, reference.dec);

  const std::array<size_t, 1> phi_shape = {sources.Size()};
  xt::xtensor<float, 1> kCosPhi(phi_shape);
  xt::xtensor<float, 1> kSinPhi(phi_shape);
  xt::xtensor<float, 1> phi(phi_shape);

  // Convert position angle from North over East to the angle used to
  // rotate the right-handed UV-plane.
  for (uint32_t source_index = 0; source_index < sources.Size();
       ++source_index) {
    phi.unchecked(source_index) =
        M_PI_2 + sources.position_angle[source_index] + M_PI;
  }

  trig_lookup_table_.compute_sincosf(sources.Size(), phi.data(), kSinPhi.data(),
                                     kCosPhi.data());

  const int nthreads =
      std::min(GetMaxNumThreads(), static_cast<int>(sources.Size()));

#pragma omp parallel if (parallelize_over_sources) num_threads(nthreads)
  {
    xt::xtensor<double, 2> euler_matrix_source = xt::zeros<double>({3, 3});
    xt::xtensor<double, 2> uvwShifted;

#pragma omp for schedule(dynamic)
    for (uint_fast32_t source_index = 0; source_index < sources.Size();
         ++source_index) {

      const bool position_angle_is_absolute =
          sources.position_angle_is_absolute[source_index];

      if (position_angle_is_absolute) {
        // Correct for projection and rotation effects: phase shift u, v, w to
        // position of the source for evaluating the gaussian
        const Directions &directions = sources.direction_vector;

        FillEulerMatrix(euler_matrix_source, directions.ra[source_index],
                        directions.dec[source_index]);

        xt::xtensor<double, 2> euler_matrix = xt::linalg::dot(
            xt::transpose(euler_matrix_source), euler_matrix_phasecenter);
        uvwShifted = xt::linalg::dot(euler_matrix, xtStationUvw);
      } else {
        uvwShifted = xtStationUvw;
      }

      const double major_axis = sources.major_axis[source_index];
      const double minor_axis = sources.minor_axis[source_index];

      // Take care of the conversion of axis lengths from FWHM in radians to
      // sigma.
      const double kUScale = major_axis * kFwhm2Sigma;
      const double kVScale = minor_axis * kFwhm2Sigma;

      uint_fast32_t p, q;

      for (uint_fast32_t bl = 0; bl < baselines.size(); ++bl) {
        std::tie(p, q) = baselines[bl];

        if (p == q) [[unlikely]]
          continue; // Skip auto-correlations

        const double u = uvwShifted(0, q) - uvwShifted(0, p);
        const double v = uvwShifted(1, q) - uvwShifted(1, p);

        // Rotate (u, v) by the position angle and scale with the major
        // and minor axis lengths (FWHM in rad).
        const double uPrime =
            kUScale * (u * kCosPhi[source_index] - v * kSinPhi[source_index]);
        const double vPrime =
            kVScale * (u * kSinPhi[source_index] + v * kCosPhi[source_index]);

        // Compute uPrime^2 + vPrime^2 and pre-multiply with -2.0 * PI^2
        // / C^2.
        const double uvPrime =
            (-2.0 * M_PI * M_PI) * (uPrime * uPrime + vPrime * vPrime);

        float *const angular_smear_terms_ptr =
            &(angular_smear_terms.unchecked(source_index, bl, 0));

        // Pre-compute the frequency-dependent part for all channels
        for (size_t ch = 0; ch < nchannels; ++ch) {
          const double lambda2 =
              frequencies(ch) * frequencies(ch) * kInvCSqr * uvPrime;
          angular_smear_terms_ptr[ch] = expf(static_cast<float>(lambda2));
        }
      }
    }
  }
}

void PredictPlanExecCPU::FillEulerMatrix(xt::xtensor<double, 2> &mat,
                                         const double ra, const double dec) {
  double sinra = sin(ra);
  double cosra = cos(ra);
  double sindec = sin(dec);
  double cosdec = cos(dec);

  mat(0, 0) = cosra;
  mat(1, 0) = -sinra;
  mat(2, 0) = 0;
  mat(0, 1) = -sinra * sindec;
  mat(1, 1) = -cosra * sindec;
  mat(2, 1) = cosdec;
  mat(0, 2) = sinra * cosdec;
  mat(1, 2) = cosra * cosdec;
  mat(2, 2) = sindec;
}

template <class E, class S, bool DO_ANGULAR_CORRECTION>
void PredictPlanExecCPU::ProcessPolarizationComponentSingle(
    E &buffer, const S &stoke_spectrum) {
  // Use preallocated storage (outside for loops)

  const auto &shift_data = GetShiftData(); // Cache reference
  const size_t n_sources = shift_data.shape(0);

  const int nthreads =
      std::min(GetMaxNumThreads(), static_cast<int>(n_sources));

#pragma omp parallel if (parallelize_over_sources) num_threads(nthreads)
  {
    std::vector<double> temp_prod_real(nchannels);
    std::vector<double> temp_prod_imag(nchannels);

    xt::xtensor<float, 2, xt::layout_type::row_major> local_smear_terms;
    local_smear_terms.resize({n_sources, nchannels});

    for (uint32_t pol = 0; pol < ((compute_stokes_I_only) ? 1 : 4); ++pol) {
      auto stoke_buffer =
          xt::view(buffer, pol, xt::all(), xt::all(), xt::all());
      auto local_buffer = xt::zeros_like(stoke_buffer);
#pragma omp for schedule(dynamic)
      for (uint_fast32_t s = 0; s < n_sources; s++) {
        uint_fast32_t p, q;
        for (uint_fast32_t bl = 0; bl < baselines.size(); ++bl) {
          std::tie(p, q) = baselines[bl];

          if (p == q) [[unlikely]]
            continue; // Skip auto-correlations

          if (correct_frequency_smearing) {
            auto source_smear_terms = xt::view(local_smear_terms, s, xt::all());
            const float phase_diff =
                static_cast<float>(station_phases(p, s) - station_phases(q, s));
            ComputeSmearTermsSingle(xt::view(station_phases, p, xt::all()),
                                    xt::view(station_phases, q, xt::all()),
                                    source_smear_terms, phase_diff);
          }

          // Compute visibilities.
          // The following loop can be parallelized, but because there is
          // already a parallelization over sources, this is not necessary. It
          // used to be parallelized with a #pragma omp parallel for, but since
          // the outer loop was also parallelized with omp, this had no effect
          // (since omp by default doesn't parallelize nested loops). After the
          // change to a ThreadPool, this would parallize each separate source,
          // which started hundreds of threads on many-core machines.

          for (uint_fast32_t ch = 0; ch < nchannels; ++ch) {
            // Note the notation:
            // Each complex number is represented as (x + j*y)
            // x: real part, y: imaginary part
            // Subscripts _p and _q represent stations p and q, respectively
            // Subscript _c represents channel (SpectrumBuffer)
            const float x_p = (shift_data(s, p, Complex::REAL, ch));
            const float y_p = (shift_data(s, p, Complex::IMAG, ch));
            const float x_q = (shift_data(s, q, Complex::REAL, ch));
            const float y_q = (shift_data(s, q, Complex::IMAG, ch));
            const double x_c = stoke_spectrum(pol, s, Complex::REAL, ch);
            const double y_c = stoke_spectrum(pol, s, Complex::IMAG, ch);

            // Compute baseline phase shift.
            // Compute visibilities.
            double q_conj_p_real = (x_p) * (x_q) + (y_p) * (y_q);
            double q_conj_p_imag = (x_p) * (y_q) - (x_q) * (y_p);

            temp_prod_real[ch] = q_conj_p_real * (x_c)-q_conj_p_imag * (y_c);
            temp_prod_imag[ch] = q_conj_p_real * (y_c) + q_conj_p_imag * (x_c);

            if constexpr (DO_ANGULAR_CORRECTION) {
              // Apply angular smearing terms
              const float angular_smear =
                  angular_smear_terms.unchecked(s, bl, ch);

              temp_prod_real[ch] *= angular_smear;
              temp_prod_imag[ch] *= angular_smear;
            }
          } // Channels.
          if (correct_frequency_smearing) {
            for (uint_fast32_t ch = 0; ch < nchannels; ++ch) {
              const float smear_term = local_smear_terms.unchecked(s, ch);
              local_buffer.unchecked(bl, Complex::REAL, ch) +=
                  smear_term * temp_prod_real[ch];
              local_buffer.unchecked(bl, Complex::IMAG, ch) +=
                  smear_term * temp_prod_imag[ch];
            }
          } else {
            for (uint_fast32_t ch = 0; ch < nchannels; ++ch) {
              local_buffer.unchecked(bl, Complex::REAL, ch) +=
                  temp_prod_real[ch];
              local_buffer.unchecked(bl, Complex::IMAG, ch) +=
                  temp_prod_imag[ch];
            }
          }
        }
      }
#pragma omp critical
      {
        // Copy the local buffer to the main buffer
        stoke_buffer += local_buffer;
      }
    }
#pragma omp barrier
  }
}

template <bool DO_ANGULAR_CORRECTION>
template <class E, class S, class Tag, class Arch>
void PredictPlanExecCPU::ComputePolarizationDispatch<
    DO_ANGULAR_CORRECTION>::operator()(Arch, E &buffer, const S &spectrum,
                                       Tag) const {
  using b_type = xsimd::batch<double, Arch>;

  size_t inc = b_type::size;
  const size_t n_channels = parent.nchannels;
  // size for which the vectorization is possible
  size_t vec_size = n_channels - n_channels % inc;
  const auto &shift_data = parent.GetShiftData(); // Cache reference
  const size_t n_sources = shift_data.shape(0);
  const size_t n_baselines = parent.baselines.size();

  const int nthreads =
      std::min(parent.GetMaxNumThreads(), static_cast<int>(n_sources));
  using zeros_view_type = decltype(xt::zeros_like(
      xt::view(buffer, 0, xt::all(), xt::all(), xt::all())));

  std::vector<std::unique_ptr<zeros_view_type>> thread_buffers;
  thread_buffers.resize(nthreads);

#pragma omp parallel if (parent.parallelize_over_sources) num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    thread_buffers[tid] = std::make_unique<zeros_view_type>(
        xt::zeros_like(xt::view(buffer, 0, xt::all(), xt::all(), xt::all())));
    auto &local_buffer = thread_buffers[tid];
    xt::xtensor<float, 2, xt::layout_type::row_major> local_smear_terms(
        std::array<size_t, 2>{n_sources, n_channels});

    for (uint32_t pol = 0; pol < ((parent.compute_stokes_I_only) ? 1 : 4);
         ++pol) {
      local_buffer->fill(0.0);
      auto stoke_buffer =
          xt::view(buffer, pol, xt::all(), xt::all(), xt::all());
#pragma omp for schedule(dynamic)
      for (uint_fast32_t s = 0; s < n_sources; ++s) {
        const double *const spectrum_real_ptr =
            &(spectrum.unchecked(pol, s, Complex::REAL, 0));
        const double *const spectrum_imag_ptr =
            &(spectrum.unchecked(pol, s, Complex::IMAG, 0));

        uint_fast32_t p, q;

        for (uint_fast32_t bl = 0; bl < n_baselines; ++bl) {
          std::tie(p, q) = parent.baselines[bl];

          if (p == q) [[unlikely]]
            continue; // Skip auto-correlations

          const double *station_phases_p_ptr =
              &parent.station_phases.unchecked(p, 0);
          const double *station_phases_q_ptr =
              &parent.station_phases.unchecked(q, 0);

          if (parent.correct_frequency_smearing) {
            const float phase_diff = static_cast<float>(
                station_phases_p_ptr[s] - station_phases_q_ptr[s]);
            auto source_smear_terms = xt::view(local_smear_terms, s, xt::all());

            parent.ComputeSmearTermsSingle(
                xt::view(parent.station_phases, p, xt::all()),
                xt::view(parent.station_phases, q, xt::all()),
                source_smear_terms, phase_diff);
          }

          const float *const shift_data_p_ptr_real =
              &shift_data.unchecked(s, p, Complex::REAL, 0);
          const float *const shift_data_p_ptr_imag =
              &shift_data.unchecked(s, p, Complex::IMAG, 0);
          const float *const shift_data_q_ptr_real =
              &shift_data.unchecked(s, q, Complex::REAL, 0);
          const float *const shift_data_q_ptr_imag =
              &shift_data.unchecked(s, q, Complex::IMAG, 0);

          double *const output_real_ptr =
              &(local_buffer->unchecked(bl, Complex::REAL, 0));
          double *const output_imag_ptr =
              &(local_buffer->unchecked(bl, Complex::IMAG, 0));

          const float *const angular_smear_ptr =
              &(parent.angular_smear_terms.unchecked(s, bl, 0));

          const float *const local_smear_terms_ptr =
              &(local_smear_terms.unchecked(s, 0));

          for (uint_fast32_t i = 0; i < vec_size; i += inc) {
            // Load shift data for p and q
            const b_type x_p = b_type::load(shift_data_p_ptr_real + i, Tag());
            const b_type y_p = b_type::load(shift_data_p_ptr_imag + i, Tag());
            const b_type x_q = b_type::load(shift_data_q_ptr_real + i, Tag());
            const b_type y_q = b_type::load(shift_data_q_ptr_imag + i, Tag());

            const b_type spectrum_real =
                b_type::load(spectrum_real_ptr + i, Tag());
            const b_type spectrum_imag =
                b_type::load(spectrum_imag_ptr + i, Tag());

            const b_type real_part =
                xsimd::add(xsimd::mul(x_p, x_q), xsimd::mul(y_p, y_q));
            const b_type imag_part =
                xsimd::sub(xsimd::mul(x_p, y_q), xsimd::mul(x_q, y_p));

            // Compute the product with the spectrum
            b_type temp_real = xsimd::sub(xsimd::mul(real_part, spectrum_real),
                                          xsimd::mul(imag_part, spectrum_imag));
            b_type temp_imag = xsimd::add(xsimd::mul(real_part, spectrum_imag),
                                          xsimd::mul(imag_part, spectrum_real));

            // Apply angular smearing terms if required
            if constexpr (DO_ANGULAR_CORRECTION) {
              alignas(alignof(b_type)) double angular_tmp[b_type::size];

              // Convert float angular terms to double
              for (uint_fast8_t k = 0; k < b_type::size; ++k) {
                angular_tmp[k] = static_cast<double>(angular_smear_ptr[i + k]);
              }

              // Load as a double batch and multiply
              const b_type angular_batch = b_type::load_unaligned(angular_tmp);
              temp_real = xsimd::mul(temp_real, angular_batch);
              temp_imag = xsimd::mul(temp_imag, angular_batch);
            }

            const b_type buffer_real = b_type::load(output_real_ptr + i, Tag());
            const b_type buffer_imag = b_type::load(output_imag_ptr + i, Tag());

            if (parent.correct_frequency_smearing) {
              // Apply smearing terms
              const b_type smear_terms_temp =
                  b_type::load_unaligned(local_smear_terms_ptr + i);

              xsimd::store(output_real_ptr + i,
                           xsimd::add(xsimd::mul(temp_real, smear_terms_temp),
                                      buffer_real),
                           Tag());

              xsimd::store(output_imag_ptr + i,
                           xsimd::add(xsimd::mul(temp_imag, smear_terms_temp),
                                      buffer_imag),
                           Tag());
            } else {
              xsimd::store(output_real_ptr + i,
                           xsimd::add(temp_real, buffer_real), Tag());

              xsimd::store(output_imag_ptr + i,
                           xsimd::add(temp_imag, buffer_imag), Tag());
            }
          }
          // Remaining part that cannot be vectorized
          for (uint_fast32_t i = vec_size; i < n_channels; ++i) {
            // Load shift data for p and q
            const float x_p = shift_data_p_ptr_real[i];
            const float y_p = shift_data_p_ptr_imag[i];
            const float x_q = shift_data_q_ptr_real[i];
            const float y_q = shift_data_q_ptr_imag[i];

            const double real_part = x_p * x_q + y_p * y_q;
            const double imag_part = x_p * y_q - x_q * y_p;

            double temp_real = real_part * spectrum_real_ptr[i] -
                               imag_part * spectrum_imag_ptr[i];
            double temp_imag = real_part * spectrum_imag_ptr[i] +
                               imag_part * spectrum_real_ptr[i];

            if constexpr (DO_ANGULAR_CORRECTION) {
              // Apply angular smearing terms
              const double angular_smear =
                  static_cast<double>(angular_smear_ptr[i]);
              temp_real *= angular_smear;
              temp_imag *= angular_smear;
            }

            if (parent.correct_frequency_smearing) {
              // Apply smearing terms
              const double smear_terms_temp = local_smear_terms.unchecked(s, i);

              output_real_ptr[i] += temp_real * smear_terms_temp;
              output_imag_ptr[i] += temp_imag * smear_terms_temp;
            } else {
              output_real_ptr[i] += temp_real;
              output_imag_ptr[i] += temp_imag;
            }
          }
        }
      }

#pragma omp critical
      {
        // Copy the local buffer to the main buffer
        if (local_buffer)
          stoke_buffer += *local_buffer;
      }
    }
  }
}

template <class E, class S, bool DO_ANGULAR_CORRECTION>
void PredictPlanExecCPU::ProcessPolarizationComponentXtensor(
    E &buffer_expr, const S &spectrum) {
  if constexpr (DO_ANGULAR_CORRECTION) {
    xsimd::dispatch(compute_pol_gaussian_dispatch_)(buffer_expr, spectrum,
                                                    xsimd::unaligned_mode());
  } else {
    xsimd::dispatch(compute_pol_dispatch_)(buffer_expr, spectrum,
                                           xsimd::unaligned_mode());
  }
}

void PredictPlanExecCPU::Compute(const PointSourceCollection &sources,
                                 Buffer4D &buffer) {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  ValidateBeforeComputation(sources, buffer);
  ProcessPolarizationComponents(buffer, sources.evaluated_spectra);
}

void PredictPlanExecCPU::ComputeWithTarget(const PointSourceCollection &sources,
                                           Buffer4D &buffer,
                                           const computation_strategy strat) {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  ValidateBeforeComputation(sources, buffer);
  ProcessPolarizationComponents(strat, buffer, sources.evaluated_spectra);
}

void PredictPlanExecCPU::ComputeWithTarget(
    const GaussianSourceCollection &sources, Buffer4D &buffer,
    const computation_strategy strat) {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  ValidateBeforeComputation(sources, buffer);
  ProcessPolarizationComponentsGaussian(strat, buffer,
                                        sources.evaluated_spectra);
}

void PredictPlanExecCPU::Compute(const GaussianSourceCollection &sources,
                                 Buffer4D &buffer) {
#ifdef ENABLE_TRACY_PROFILING
  ZoneScoped;
#endif
  ValidateBeforeComputation(sources, buffer);
  ProcessPolarizationComponentsGaussian(buffer, sources.evaluated_spectra);
}

void PredictPlanExecCPU::SetNumThreads(int num_threads) {
  omp_set_num_threads(num_threads);
}

int PredictPlanExecCPU::GetNumOMPThreads() const {
  return omp_get_num_threads();
}

int PredictPlanExecCPU::GetMaxNumThreads() const { return max_num_threads_; }

} // namespace predict
