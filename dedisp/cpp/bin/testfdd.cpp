#include <chrono>
#include <iostream>
#include <random>

#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/core/xmath.hpp>

#include "fddplan.hpp"
#include "metadata.hpp"
#include "utilities.hpp"

int main() {
  // Observation details: duration, integration time, max. frequency, bandwidth,
  // and channel count.
  const dedisp::ObservationInfo observation{30.0f, 250.0e-6, 1581.0f, 100.0f,
                                            1024};

  // Mock signal parameters: RMS noise floor, DM, pulse arrival time, and signal
  // amplitude.
  const dedisp::SignalInfo mock_signal{25.0f, 41.159f, 3.14159f, 200.0f};

  // Dedispersion plan constraints: start DM, end DM, pulse width (ms), smearing
  // tolerance.
  const dedisp::DedispersionConstraints constraints{2.0f, 100.0f, 4.0f, 1.25f};

  const float frequency_resolution =
      -1.0 * observation.bandwidth /
      observation.channels; // MHz   (This must be negative!)
  const size_t n_samples = observation.duration / observation.sampling_period;

  auto mock_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto plan_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto prep_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto exec_timer = std::make_unique<dedisp::benchmark::Timer>();

  std::cout << "Generating mock input..." << std::endl;
  mock_timer->start();
  xt::xarray<float> mock_input =
      dedisp::simulate_dispersed_signal(mock_signal, observation);

  // Quantise the input signal. Note that this actually clips the signal...
  // TODO: don't clip?
  xt::xarray<uint8_t> quantised_mock_input(mock_input.shape());
  for (size_t c = 0; c < mock_input.shape(0); ++c) {
    for (size_t s = 0; s < mock_input.shape(1); ++s) {
      quantised_mock_input(c, s) = dedisp::quantise(mock_input(c, s));
    }
  }
  mock_timer->pause();
  std::cout << mock_input << std::endl;
  std::cout << "> runtime: " << mock_timer->duration() << " seconds "
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
  xt::xarray<float> mock_output = fdd_plan.execute(mock_input);
  exec_timer->pause();
  std::cout << "> runtime: " << exec_timer->duration() << " seconds "
            << std::endl;

  const double total_runtime = mock_timer->duration() + plan_timer->duration() +
                               prep_timer->duration() + exec_timer->duration();
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "FDD test finished; total runtime = " << total_runtime
            << std::endl;

  fdd_plan.show();

  std::cout << '\n' << "Dedispersion report" << std::endl;
  const float raw_mean = xt::mean<float>(mock_input)();
  const float raw_std  = xt::stddev<float>(mock_input)();
  std::cout << "  Raw RMS    : " << raw_mean << std::endl;
  std::cout << "  Raw StdDev : " << raw_std << std::endl;

  const float input_mean = xt::mean<float>(quantised_mock_input)();
  const float input_std  = xt::stddev<float>(quantised_mock_input)();
  std::cout << "  Input RMS    : " << input_mean << std::endl;
  std::cout << "  Input StdDev : " << input_std << std::endl;

  const float output_mean = xt::mean<float>(mock_output)();
  const float output_std  = xt::stddev<float>(mock_output)();
  std::cout << "  Output RMS    : " << output_mean << std::endl;
  std::cout << "  Output StdDev : " << output_std << std::endl;

  const size_t n_samples_computed = n_samples - fdd_plan.max_delay();
  const xt::xarray<float> dm_table = fdd_plan.get_dm_table();

  // TODO: limit output to 100 like the original dedisp code.
  // int n_candidates = 0;
  // for (size_t d = 0; d < fdd_plan.dm_count(); ++d) {
  //   for (size_t s = 0; s < n_samples_computed; ++s) {
  //     const float value = mock_output(d, s);
  //     std::cout << "  Checking DM trial " << d << " x " << s << " => " << value - output_mean << " > " << 6.0f * output_std << std::endl;
  //     if (value - output_mean > 6.0f * output_std) {
  //       // printf(
  //       //     "  DM trial %u (%.3f pc/cm^3), Samp %u (%.6f s): %f (%.2f sigma)\n",
  //       //     d, dm_table(d), s, s * observation.sampling_period, value,
  //       //     (value - output_mean) / output_std);
  //       ++n_candidates;
  //       if (n_candidates > 100) {
  //         break;
  //       }
  //     }
  //   }
  //   if (n_candidates > 100) {
  //     break;
  //   }
  // }
  // std::cout << "\nFound " << n_candidates << " DM candidates." << std::endl;


  const std::string fn_in{"fddin.npy"};
  xt::dump_npy(fn_in, quantised_mock_input);
  std::cout << "\nInput is written to " << fn_in << "." << std::endl;

  const std::string fn_out{"fddout.npy"};
  xt::dump_npy(fn_out, mock_output);
  std::cout << "Output is written to " << fn_out << "." << std::endl;
}

