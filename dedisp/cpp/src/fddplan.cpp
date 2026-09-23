#include <cmath>
#include <iostream>

#include <xtensor-wrappers/plan_batch.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/io/xio.hpp>

#include "h5cpp/dataspace/simple.hpp"
#include "h5cpp/datatype/datatype.hpp"
#include "h5cpp/datatype/type_trait.hpp"
#include "h5cpp/file/file.hpp"
#include "h5cpp/file/functions.hpp"
#include "h5cpp/node/group.hpp"

#include "fddplan.hpp"
#include "kernels.hpp"
#include "utilities.hpp"

namespace dedisp {

FDDPlan::FDDPlan(size_t n_channels, float time_resolution, float peak_frequency,
                 float frequency_resolution)
    : dm_count_{0}, n_channels_{n_channels}, max_delay_{0},
      time_resolution_{time_resolution}, peak_frequency_{peak_frequency},
      frequency_resolution_{frequency_resolution} {
  // Generate the delay table without the DM factor, which is applied during
  // dedispersion.
  generate_delay_table();
}

xt::xarray<float> FDDPlan::execute(const xt::xarray<uint8_t> &input) {
  const size_t n_samples =
      input.shape(0); // input has dimensions samples x channel
  const size_t n_spin_frequencies = n_samples / 2 + 1;
  const size_t n_output_samples = n_samples - max_delay_;

  const bool use_zero_padding = true;
  const size_t n_samples_fft =
      use_zero_padding ? round_up(n_samples + 1, 16384) : n_samples;
  const size_t n_samples_padded = round_up(n_samples_fft + 1, 1024);
  const size_t n_fft_frequency_bins = n_samples_padded / 2 + 1;

#ifdef DEDISP_DEBUG
  std::cout << "n_samples            = " << n_samples << '\n';
  std::cout << "n_samples_fft        = " << n_samples_fft << '\n';
  std::cout << "n_samples_padded     = " << n_samples_padded << '\n';
  std::cout << "n_fft_frequency_bins = " << n_fft_frequency_bins << '\n';
#endif

#ifdef DEDISP_BENCHMARK
  auto init_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto preprocessing_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto dedispersion_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto postprocessing_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto output_timer = std::make_unique<dedisp::benchmark::Timer>();
#endif

  // 1. Generate spin table
  std::cout << "(1) Generate the spin frequency table." << std::endl;

#ifdef DEDISP_BENCHMARK
  init_timer->start();
#endif

  generate_spin_frequency_table(n_spin_frequencies, n_samples);

#ifdef DEDISP_BENCHMARK
  init_timer->pause();
#endif

#ifdef DEDISP_DEBUG
  std::cout << spin_frequency_table_ << std::endl;
#endif

  // 2. Allocate scratch buffers and build the FFT plans, rebuilding only when
  //    the transform shapes change.
  std::cout << "(2) Allocate scratch and (re)build FFT plans." << std::endl;

#ifdef DEDISP_BENCHMARK
  init_timer->start();
#endif

  setup_fft(n_samples_padded, n_fft_frequency_bins);

#ifdef DEDISP_BENCHMARK
  init_timer->pause();
#endif

  // 3. Transpose data (convert input bytes to floats)
  std::cout << "(3) Transpose data: int -> float." << std::endl;

#ifdef DEDISP_BENCHMARK
  preprocessing_timer->start();
#endif

  // Transpose from (channels, samples) to (samples, channels) row-major layout
  // so each channel is a contiguous FFT row, and zero the padding region
  // beyond n_samples so the (longer) FFT sees a clean input.
  transposed_input_.fill(0.0f);

  constexpr float byte_offset = 127.5;
  transpose_data<uint8_t, float>(n_channels_, n_samples, n_channels_,
                                 n_samples_padded, byte_offset, n_channels_,
                                 input.data(), transposed_input_.data());

#ifdef DEDISP_BENCHMARK
  preprocessing_timer->pause();
#endif

#ifdef DEDISP_DEBUG_HDF5
  hdf5::file::File output_file = hdf5::file::create("intermediate.h5");
  hdf5::node::Group root_node = output_file.root();

  {
    hdf5::datatype::Datatype datatype =
        hdf5::datatype::TypeTrait<float>::create();
    const std::vector<hsize_t> dims(transposed_input_.shape().begin(),
                                    transposed_input_.shape().end());
    auto dataspace = hdf5::dataspace::Simple(dims);
    auto signal_dataset =
        root_node.create_dataset("transpose", datatype, dataspace);

    signal_dataset.write(*transposed_input_.data(), datatype, dataspace);
  }
#endif

  // 4. Real-to-complex FFT: time series data to frequency domain
  // Batched over frequency (one FFT per channel), parallelised over the batch.
  std::cout << "(4) Forward FFT: real-to-complex." << std::endl;

#ifdef DEDISP_BENCHMARK
  preprocessing_timer->start();
#endif

  rfft_plan_.execute();

#ifdef DEDISP_BENCHMARK
  preprocessing_timer->pause();
#endif

#ifdef DEDISP_DEBUG_HDF5
  {
    hdf5::datatype::Datatype datatype =
        hdf5::datatype::TypeTrait<float>::create();
    const std::vector<hsize_t> dims(rfft_plan_.output().shape().begin(),
                                    rfft_plan_.output().shape().end());
    auto dataspace = hdf5::dataspace::Simple(dims);
    auto signal_dataset =
        root_node.create_dataset("fdd-fft-r2c", datatype, dataspace);

    signal_dataset.write(*rfft_plan_.output().data(), datatype, dataspace);
  }
#endif

  // 5. Run dedispersion algorithm (CPU reference or optimised version)
  std::cout << "(5) Run dedispersion algorithm." << std::endl;

  if (dm_count_ > 0) {
    // Zero the half-complex bins that the kernel does not touch so the inverse
    // FFT has a clean input.
    dm_scratch_.fill(std::complex<float>{0.0f, 0.0f});

#ifdef DEDISP_BENCHMARK
    dedispersion_timer->start();
#endif

    const size_t in_out_stride = n_fft_frequency_bins;
    dedisp::fourier_domain_dedisperse(
        dm_count_, n_spin_frequencies, n_channels_, time_resolution_,
        spin_frequency_table_.data(), dm_table_.data(), delay_table_.data(),
        in_out_stride, in_out_stride,
        const_cast<std::complex<float> *>(rfft_plan_.output().data()),
        dm_scratch_.data());

#ifdef DEDISP_BENCHMARK
    dedispersion_timer->pause();
#endif

#ifdef DEDISP_DEBUG_HDF5
    {
      hdf5::datatype::Compound datatype =
          hdf5::datatype::Compound::create(sizeof(std::complex<float>));
      datatype.insert("r", 0,
                      hdf5::datatype::TypeTrait<float>::create(float()));
      datatype.insert("i", alignof(float),
                      hdf5::datatype::TypeTrait<float>::create(float()));
      const std::vector<hsize_t> dims(dm_scratch_.shape().begin(),
                                      dm_scratch_.shape().end());
      auto dataspace = hdf5::dataspace::Simple(dims);
      auto signal_dataset =
          root_node.create_dataset("fdd-dedisp", datatype, dataspace);

      signal_dataset.write(*dm_scratch_.data(), datatype, dataspace);
    }
#endif

    // 6. Complex-to-real FFT: frequency domain back to time series data
    // Batched over DM (one FFT per DM trial), parallelised over the batch.
    std::cout << "(6) Inverse FFT: complex-to-real." << std::endl;

#ifdef DEDISP_BENCHMARK
    postprocessing_timer->start();
#endif

    irfft_plan_.execute();

#ifdef DEDISP_BENCHMARK
    postprocessing_timer->pause();
#endif
  }

#ifdef DEDISP_BENCHMARK
  init_timer->start();
#endif

  const std::vector<size_t> computed_shape = {n_output_samples, dm_count_};
  xt::xarray<float> computed_data(computed_shape);

#ifdef DEDISP_BENCHMARK
  init_timer->pause();
#endif

#ifdef DEDISP_DEBUG
  std::cout << "output samps = " << computed_data.shape(0)
            << "; DM count = " << computed_data.shape(1) << '\n';
#endif
#ifdef DEDISP_BENCHMARK
  output_timer->start();
#endif
  // xt::fftw::irfft normalises by the transform length; the c2r plan does
  // not, so reproduce that 1/n_samples_padded scaling here.
  const xt::xarray<float> &dm_data = irfft_plan_.output();
  const float n_scale = static_cast<float>(n_samples_padded);
  for (size_t s = 0; s < n_output_samples; ++s) {
    for (size_t d = 0; d < dm_count_; ++d) {
      xt::view(computed_data, s, d) = xt::view(dm_data, d, s) / n_scale;
    }
  }

#ifdef DEDISP_BENCHMARK
  output_timer->pause();
  std::cout << std::endl;
  std::cout << "Initialization time : " << init_timer->duration() << " sec."
            << std::endl;
  std::cout << "Preprocessing time  : " << preprocessing_timer->duration()
            << " sec." << std::endl;
  std::cout << "Dedispersion time   : " << dedispersion_timer->duration()
            << " sec." << std::endl;
  std::cout << "Postprocessing time : " << postprocessing_timer->duration()
            << " sec." << std::endl;
  std::cout << "Output copy time    : " << output_timer->duration() << " sec."
            << std::endl;
  std::cout << std::endl;
#endif

  return computed_data;
}

void FDDPlan::show() const {
  std::cout << "FDD Plan Summary" << std::endl;
  std::cout << "  nr channels:          " << n_channels_ << std::endl;
  std::cout << "  nr dm trials:         " << dm_count_ << std::endl;
  std::cout << "  max delay:            " << max_delay_ * time_resolution_
            << " s (" << max_delay_ << " samples)" << std::endl;
  std::cout << "  time resolution:      " << time_resolution_ << " s"
            << std::endl;
  std::cout << "  frequency resolution: " << frequency_resolution_ << " MHz"
            << std::endl;
  std::cout << "  peak frequency:       " << peak_frequency_ << " MHz"
            << std::endl;
}

void FDDPlan::generate_dm_list(float dm_start, float dm_end, float pulse_width,
                               float tolerance) {
  // Fill the DM list
  const double negative_frequency_resolution = -frequency_resolution_;
  const double time_resolution = time_resolution_ * 1e6;
  const double f = (peak_frequency_ +
                    ((n_channels_ / 2) - 0.5) * negative_frequency_resolution) *
                   1e-3;
  const double a = 8.3 * negative_frequency_resolution / (f * f * f);
  const double a_squared = a * a;
  const double b_squared =
      a_squared * (double)(n_channels_ * n_channels_ / 16.0);
  const double tolerance_squared = tolerance * tolerance;
  const double c =
      (time_resolution * time_resolution + pulse_width * pulse_width) *
      (tolerance_squared - 1.0);

  std::vector<float> dm_list = {dm_start};
  while (dm_list.back() < dm_end) {
    const double previous_dm = dm_list.back();
    const double previous_dm_squared = previous_dm * previous_dm;
    const double k = c + tolerance_squared * a_squared * previous_dm_squared;
    const double dm = ((b_squared * previous_dm +
                        std::sqrt(-a_squared * b_squared * previous_dm_squared +
                                  (a_squared + b_squared) * k)) /
                       (a_squared + b_squared));
    dm_list.push_back(dm);
  }

  dm_count_ = dm_list.size();

  // Store the DM table in memory
  dm_table_ = xt::adapt(dm_list, {dm_count_});

  // Calculate and store the maximum delay
  const float max_dm = dm_table_(dm_count_ - 1);
  const float max_delay = delay_table_(n_channels_ - 1);
  max_delay_ = static_cast<size_t>(max_dm * max_delay + 0.5);
}

void FDDPlan::generate_linear_dm_list(float dm_start, float dm_end,
                                      float dm_step) {
  assert(dm_step > 0);

  // Linearly fill the DM list
  std::vector<float> dm_list = {dm_start};
  while (dm_list.back() < dm_end) {
    dm_list.push_back(dm_list.back() + dm_step);
  }

  dm_count_ = dm_list.size();

  // Store the DM table in memory
  dm_table_ = xt::adapt(dm_list, {dm_count_});

  // Calculate and store the maximum delay
  const float max_dm = dm_table_(dm_count_ - 1);
  const float max_delay = delay_table_(n_channels_ - 1);
  max_delay_ = static_cast<size_t>(max_dm * max_delay + 0.5);
}

void FDDPlan::generate_delay_table() {
  delay_table_.resize({n_channels_});

  for (size_t channel = 0; channel < n_channels_; ++channel) {
    const float inverse_channel_frequency =
        1.0f / (peak_frequency_ - channel * frequency_resolution_);
    const float inverse_peak_frequency = 1.0f / peak_frequency_;

    delay_table_(channel) =
        4.148741601e3 / time_resolution_ *
        (inverse_channel_frequency * inverse_channel_frequency -
         inverse_peak_frequency * inverse_peak_frequency);
  }
}

void FDDPlan::generate_spin_frequency_table(size_t n_spin_frequencies,
                                            size_t n_samples) {
  spin_frequency_table_.resize({n_spin_frequencies});

#ifdef DEDISP_USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < n_spin_frequencies; ++i) {
    spin_frequency_table_(i) = i * (1.0f / (n_samples * time_resolution_));
  }
}

void FDDPlan::setup_fft(size_t n_samples_padded, size_t n_fft_frequency_bins) {
  const bool shape_changed = n_samples_padded != plan_n_samples_padded_;
  const bool dm_count_changed = dm_count_ != plan_dm_count_;

  if (shape_changed) {
    transposed_input_.resize({n_channels_, n_samples_padded});
  }

  if (shape_changed || dm_count_changed) {
    dm_scratch_.resize({dm_count_, n_fft_frequency_bins});

    if (dm_count_ > 0) {
      // Batched real-to-complex over channels: one rfft per channel of length
      // n_samples_padded, rows (channels) contiguous in transposed_input_.
      xt::fftw::batch_layout r2c;
      r2c.howmany = n_channels_;
      r2c.n = {static_cast<int>(n_samples_padded)};
      rfft_plan_ =
          xt::fftw::make_batch_rfft_plan(transposed_input_.data(), r2c);

      // Batched complex-to-real over DM trials: one irfft per DM of length
      // n_samples_padded from half-complex input; the input rows are n/2+1
      // elements apart (tightly packed), which is what layout.idist encodes.
      xt::fftw::batch_layout c2r;
      c2r.howmany = dm_count_;
      c2r.n = {static_cast<int>(n_samples_padded)};
      c2r.idist = static_cast<int>(n_fft_frequency_bins);
      irfft_plan_ = xt::fftw::make_batch_irfft_plan(dm_scratch_.data(), c2r);
    } else {
      irfft_plan_ = xt::fftw::batch_plan<float, float>{};
    }

    plan_n_samples_padded_ = n_samples_padded;
    plan_dm_count_ = dm_count_;
  }
}

} // namespace dedisp
