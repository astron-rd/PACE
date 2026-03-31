#include <chrono>
#include <iostream>
#include <random>

#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor-fftw/basic.hpp>
#include <xtensor-fftw/helper.hpp>

#include "fddplan.hpp"
#include "kernels.hpp"
#include "metadata.hpp"
#include "utilities.hpp"

void test_transpose_data() {
  // Generate a small float array
  constexpr size_t n_samples = 5;
  constexpr size_t n_channels = 2;
  std::vector<size_t> input_shape = {n_samples, n_channels};
  xt::xarray<float> raw_data = xt::arange<float>(0, n_samples * n_channels).reshape(input_shape);
  std::cout << raw_data << std::endl;

  // Quanitise the float values
  std::cout << "Quantise data..." << std::endl;
  xt::xarray<uint8_t> data(input_shape);
  for (size_t s = 0; s < data.shape(0); ++s) {
    for (size_t c = 0; c < data.shape(1); ++c) {
      data(s, c) = dedisp::quantise(raw_data(s, c));
    }
  }
  std::cout << data << std::endl;

  // Apply transpose_data()
  constexpr size_t n_samples_padded = 8;
  std::vector<size_t> padded_shape = {n_channels, n_samples_padded};
  xt::xarray<float> padded_data = xt::zeros<float>(padded_shape);
  std::cout << padded_data << std::endl;

  std::cout << "Transpose data..." << std::endl;
  constexpr float byte_offset = 127.5f;
  dedisp::transpose_data<uint8_t, float>(n_channels, n_samples, n_channels,
                         n_samples_padded, byte_offset, 1,
                         data.data(), padded_data.data());

  std::cout << padded_data << std::endl;
}

void test_r2c_fft() {
  // Define FFT input
  constexpr size_t n_samples = 8;
  constexpr size_t n_channels = 2;
  std::vector<size_t> shape = {n_channels, n_samples};
  xt::xarray<float> data = xt::arange<float>(0, n_samples * n_channels).reshape(shape);
  std::cout << data << std::endl;

  // Output storage
  const size_t n_fft_bins = n_samples / 2 + 1;
  std::vector<size_t> scratch_shape = {n_channels, n_fft_bins};
  xt::xarray<std::complex<float>> scratch = xt::zeros<std::complex<float>>(scratch_shape);
  std::cout << scratch << std::endl;

  // Grab a single row of time samples from a single channel
  constexpr size_t channel_index = 0;
  xt::xarray<float> samples = xt::eval(xt::row(data, channel_index));
  std::cout << samples << std::endl;

  // Perform the FFT
  samples = xt::fftw::fftshift(samples);
  xt::view(scratch, channel_index, xt::all()) = xt::fftw::rfft(samples);
  std::cout << "xt::fftw::rfft() =\n" << xt::fftw::rfft(samples) << std::endl;
  std::cout << scratch << std::endl;
}

void test_c2r_fft() {
  // Define iFFT input
  constexpr size_t n_dms = 2;
  constexpr size_t n_samples = 8;
  const size_t n_fft_bins = n_samples / 2 + 1;
  std::vector<size_t> shape = {n_dms, n_fft_bins};
  xt::xarray<std::complex<float>> data = xt::arange<float>(0, n_samples * n_fft_bins).reshape(shape);
  std::cout << data << std::endl;

  // Allocate output storage
  std::vector<size_t> scratch_shape = {n_dms, n_samples};
  xt::xarray<float> scratch = xt::zeros<float>(scratch_shape);
  std::cout << scratch << std::endl;

  // Extract a single DM-row with frequency-domain samples
  constexpr size_t dm_index = 0;
  xt::xarray<std::complex<float>> samples = xt::eval(xt::row(data, dm_index));
  std::cout << samples << std::endl;

  samples = xt::fftw::fftshift(samples);
  xt::view(scratch, dm_index, xt::all()) = xt::fftw::irfft(samples);
  std::cout << "xt::fftw::rfft() = " << xt::fftw::irfft(samples) << std::endl;
  std::cout << scratch << std::endl;
}

void test_fdd_kernel() {
  // Set obs params
  constexpr size_t n_samples = 8;
  constexpr size_t n_channels = 3;
  constexpr size_t n_dms = 2;

  constexpr float time_res = 0.1f;

  const size_t n_spin = n_samples / 2 + 1;
  const size_t n_fft_bins = n_samples / 2 + 1;

  // Mock DM table, delays, and spin frequencies
  std::vector<size_t> dm_shape = {n_dms};
  xt::xarray<float> dm_table(dm_shape);
  dm_table(0) = 10.0f;
  dm_table(1) = 11.0f;

  std::vector<size_t> delay_shape = {n_channels};
  xt::xarray<float> delay_table(delay_shape);
  for (size_t channel = 0; channel < n_channels; ++channel) {
    delay_table(channel) = 1 / time_res * (channel + 1);
  }

  std::vector<size_t> spin_shape = {n_spin};
  xt::xarray<float> spin_table(spin_shape);
  for (size_t i = 0; i < n_spin; ++i) {
    spin_table(i) = i * (1.0f / (n_samples * time_res) / 100);
  }

  std::cout << "DM table = " << dm_table << std::endl;
  std::cout << "Delay table = " << delay_table << std::endl;
  std::cout << "Spin table = " << spin_table << std::endl;

  // Mock freq and dm scratch arrays
  const std::vector<size_t> input_shape = {n_channels, n_fft_bins};
  // xt::xarray<std::complex<float>> input = xt::arange<float>(0, n_channels * n_fft_bins).reshape(input_shape);
  xt::xarray<std::complex<float>> input = xt::ones<std::complex<float>>(input_shape);


  std::cout << "\nKernel input =\n" << input << '\n' << std::endl;

  const std::vector<size_t> output_shape = {n_dms, n_fft_bins};
  xt::xarray<std::complex<float>> output(output_shape); // = xt::zeros<std::complex<float>>(output_shape);

  const size_t stride = n_fft_bins;
  std::cout << "Executing FDD kernel..." << std::endl;
  dedisp::fourier_domain_dedisperse(
      n_dms, n_spin, n_channels, time_res, spin_table.data(), dm_table.data(), delay_table.data(),
      stride, stride, input.data(), output.data()
  );

  std::cout << "\nKernel output =\n" << output << std::endl;
}

/// This is a collection of simple tests to mock the steps
/// in FDDPlan::execute(), with the aim to verify its correctness.
int main() {
  std::cout << "TEST (1) -- transpose_data" << std::endl;
  test_transpose_data();
  std::cout << "TEST (1) -- Finished\n" << std::endl;

  std::cout << "TEST (2) -- real-to-complex fft" << std::endl;
  test_r2c_fft();
  std::cout << "TEST (2) -- Finished\n" << std::endl;

  std::cout << "TEST (3) -- complex-to-real fft" << std::endl;
  test_c2r_fft();
  std::cout << "TEST (3) -- Finished\n" << std::endl;

  std::cout << "TEST (4) -- FDD kernel" << std::endl;
  test_fdd_kernel();
  std::cout << "TEST (4) -- Finished\n" << std::endl;
}
