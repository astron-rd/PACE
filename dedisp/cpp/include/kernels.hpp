#pragma once

#include <complex>

namespace dedisp {

void fourier_domain_dedisperse(size_t dm_count, size_t n_frequencies,
                               size_t n_channels, float time_resolution,
                               float *spin_frequencies,
                               float *dispersion_measures, float *delays,
                               size_t stride_in, size_t stride_out,
                               std::complex<float>* input,
                               std::complex<float>* output);

} // namespace dedisp
