#include <cstdint>
#include <filesystem>
#include <iostream>

#include "cxxopts.hpp"

#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>

#include "h5cpp/dataspace/simple.hpp"
#include "h5cpp/datatype/datatype.hpp"
#include "h5cpp/datatype/type_trait.hpp"
#include "h5cpp/file/file.hpp"
#include "h5cpp/file/functions.hpp"
#include "h5cpp/node/group.hpp"

#include "fddplan.hpp"
#include "metadata.hpp"
#include "utilities.hpp"

cxxopts::Options configure_cli_options(const char *argv[]) {
  cxxopts::Options options(argv[0], "Fourier Domain Dedispersion");

  const std::string kSpectrum = "signal.h5";
  const std::string kFilename = "fdd.h5";

  constexpr float kDmStart = 2.0f;
  constexpr float kDmEnd = 100.0f;
  constexpr float kDmTolerance = 1.25f;
  constexpr float kPulseWidth = 4.0f;

  options.add_options()(
      "spectrum", "Path to the HDF5 file containing the dynamic spectrum.",
      cxxopts::value<std::filesystem::path>()->default_value(kSpectrum))(
      "dm-start", "Start of the dispersion measure search interval",
      cxxopts::value<float>()->default_value(std::to_string(kDmStart)))(
      "dm-end", "End of the dispersion measure search interval",
      cxxopts::value<float>()->default_value(std::to_string(kDmEnd)))(
      "dm-tolerance", "Smearing tolerance",
      cxxopts::value<float>()->default_value(std::to_string(kDmTolerance)))(
      "pulse-width", "Expected pulse width in milliseconds",
      cxxopts::value<float>()->default_value(std::to_string(kPulseWidth)))(
      "file",
      "Filename for the HDF5 dataset containing the output of the dedispersion "
      "plan.",
      cxxopts::value<std::filesystem::path>()->default_value(kFilename))(
      "h,help", "Print help");

  return options;
}

cxxopts::ParseResult parse_arguments(int argc, const char *argv[]) {
  cxxopts::Options options = configure_cli_options(argv);

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(EXIT_SUCCESS);
  }

  return result;
}

template <typename T>
xt::xarray<T> load_dataset_to_xtensor(hdf5::node::Dataset &dataset) {
  hdf5::datatype::Datatype datatype = dataset.datatype();
  hdf5::dataspace::Simple dataspace = dataset.dataspace();
  auto dim = dataspace.current_dimensions();

  xt::xarray<T> data = xt::xarray<T>::from_shape(dim);
  dataset.read(*data.data(), datatype, dataspace);
  return data;
}

int main(int argc, const char *argv[]) {
  const cxxopts::ParseResult cli_options = parse_arguments(argc, argv);

// Observation details
const dedisp::ObservationInfo observation{30.0f,    // duration
                                          250.0e-6, // integrationxtime
                                          1581.0f,  // max. frequency
                                          100.0f,   // bandwidth
                                          1024      // channel count
                                         };  

  // Mock signal parameters: RMS noise floor, DM, pulse arrival time, and signal
  // amplitude.
  constexpr float default_intensity = 25.0f;
  const dedisp::SignalInfo mock_signal{25.0f, 41.159f, 3.14159f,
                                       default_intensity};

  // Dedispersion plan constraints: start DM, end DM, pulse width (ms), smearing
  // tolerance.
  const dedisp::DedispersionConstraints constraints(cli_options);

  const float frequency_resolution =
      observation.bandwidth / observation.channels; // MHz
  const size_t n_samples = observation.duration / observation.sampling_period;

  auto input_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto plan_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto prep_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto exec_timer = std::make_unique<dedisp::benchmark::Timer>();

  std::cout << "Reading input from HDF5..." << std::endl;
  input_timer->start();

  xt::xarray<uint8_t> input;
  {
    using namespace hdf5;

    const std::filesystem::path h5_file_path =
        cli_options["spectrum"].as<std::filesystem::path>();
    if (!std::filesystem::exists(h5_file_path)) {
      std::cout << "Error: " << h5_file_path << " does not exist\n";
      return 0;
    }

    file::File input_file = file::open(h5_file_path);
    node::Group root_node = input_file.root();

    node::Dataset signal_ds = root_node.get_dataset("dynspec");
    input = load_dataset_to_xtensor<uint8_t>(signal_ds);
  }

  input_timer->pause();
  std::cout << input << std::endl;
  std::cout << "> runtime: " << input_timer->duration() << " seconds "
            << std::endl;

  // Initialise and execute the FDD plan
  std::cout << "Initialising FDD Plan..." << std::endl;
  plan_timer->start();
  dedisp::FDDPlan fdd_plan(observation.channels, observation.sampling_period,
                           observation.peak_frequency, frequency_resolution);
  plan_timer->pause();
  std::cout << "Generated delay table: ";
  std::cout << fdd_plan.get_delay_table() << std::endl;
  std::cout << "> runtime: " << plan_timer->duration() << " seconds "
            << std::endl;

  std::cout << "Generate DM list..." << std::endl;
  prep_timer->start();
  fdd_plan.generate_dm_list(constraints.dm_start, constraints.dm_end,
                            constraints.pulse_width, constraints.tolerance);
  prep_timer->pause();
  std::cout << fdd_plan.get_dm_table() << std::endl;
  std::cout << "> runtime: " << prep_timer->duration() << " seconds "
            << std::endl;

  std::cout << "Execute FDD Plan..." << std::endl;
  exec_timer->start();
  xt::xarray<float> output = fdd_plan.execute(input);
  exec_timer->pause();
  std::cout << "> runtime: " << exec_timer->duration() << " seconds "
            << std::endl;

  const double total_runtime = input_timer->duration() +
                               plan_timer->duration() + prep_timer->duration() +
                               exec_timer->duration();
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "FDD test finished; total runtime = " << total_runtime
            << std::endl;

  fdd_plan.show();

  std::cout << '\n' << "Dedispersion report" << std::endl;
  const float raw_mean = xt::mean<float>(input)();
  const float raw_std = xt::stddev<float>(input)();
  std::cout << "  Raw RMS:        " << raw_mean << "     (expected: 0.000449)"
            << std::endl;
  std::cout << "  Raw StdDev:     " << raw_std << "     (expected: 25.001390)"
            << std::endl;

  const float input_mean = xt::mean<float>(input)();
  const float input_std = xt::stddev<float>(input)();
  std::cout << "  Input RMS:      " << input_mean
            << "     (expected: 127.500458)" << std::endl;
  std::cout << "  Input StdDev:   " << input_std << "     (expected: 25.003016)"
            << std::endl;

  const float output_mean = xt::mean<float>(output)();
  const float output_std = xt::stddev<float>(output)();
  std::cout << "  Output RMS:     " << output_mean
            << "     (expected: 0.000360)" << std::endl;
  std::cout << "  Output StdDev:  " << output_std << "     (expected: 0.748115)"
            << std::endl;

  const xt::xarray<float> dm_table = fdd_plan.get_dm_table();

#ifdef DEDISP_DEBUG
  const size_t n_samples_computed = n_samples - fdd_plan.max_delay();
  int n_candidates = 0;
  for (size_t s = 0; s < n_samples_computed; ++s) {
    for (size_t d = 0; d < fdd_plan.dm_count(); ++d) {
      const float value = mock_output(s, d);
      if (value - output_mean > 6.0f * output_std) {
        printf(
            "  DM trial %u (%.3f pc/cm^3), Samp %u (%.6f s): %f (%.2f sigma)\n",
            d, dm_table(d), s, s * observation.sampling_period, value,
            (value - output_mean) / output_std);
        ++n_candidates;
        if (n_candidates > 100) {
          break;
        }
      }
    }
    if (n_candidates > 100) {
      break;
    }
  }
  std::cout << "\nFound " << n_candidates << " DM candidates.\n" << std::endl;
#endif

  {
    using namespace hdf5;

    const std::filesystem::path output_file_path =
        cli_options["file"].as<std::filesystem::path>();

    file::File output_file = file::create(output_file_path);
    node::Group root_node = output_file.root();

    {
      datatype::Datatype datatype = datatype::TypeTrait<float>::create();
      const std::vector<hsize_t> dims(output.shape().begin(),
                                      output.shape().end());
      auto dataspace = dataspace::Simple(dims);
      auto signal_dataset =
          root_node.create_dataset("fddresult", datatype, dataspace);

      signal_dataset.write(*output.data(), datatype, dataspace);

      signal_dataset.attributes.create_from("computed_samples",
                                            output.shape()[0]);
      signal_dataset.attributes.create_from("integration_time",
                                            observation.sampling_period);

      std::cout << "Output is written to dataset fddresult in "
                << output_file_path.string() << "." << std::endl;
    }

    {
      datatype::Datatype datatype = datatype::TypeTrait<float>::create();
      const std::vector<hsize_t> dims(dm_table.shape().begin(),
                                      dm_table.shape().end());
      auto dataspace = dataspace::Simple(dims);
      auto signal_dataset =
          root_node.create_dataset("dispersion_measures", datatype, dataspace);

      signal_dataset.write(*dm_table.data(), datatype, dataspace);
    }
  }
}
