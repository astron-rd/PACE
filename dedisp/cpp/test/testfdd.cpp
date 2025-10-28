#include <chrono>
#include <iostream>
#include <random>

#include <xtensor/io/xio.hpp>

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
  const dedisp::SignalInfo mock_signal{25.0f, 41.159f, 3.14159f, 25.0f};

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
  std::cout << "> runtime: " << exec_timer->duration() << " seconds " <<
  std::endl;

  fdd_plan.show();

  const double total_runtime = mock_timer->duration() + plan_timer->duration() +
                               prep_timer->duration() + exec_timer->duration();
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "FDD test finished: total runtime = " << total_runtime
            << std::endl;
}