#include <chrono>
#include <iostream>
#include <random>

#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>

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
  const dedisp::SignalInfo signal_properties{25.0f, 41.159f, 3.14159f, 25.0f};

  // Dedispersion plan constraints: start DM, end DM, pulse width (ms), smearing
  // tolerance.
  const dedisp::DedispersionConstraints constraints{2.0f, 100.0f, 4.0f, 1.25f};

  const float frequency_resolution =
      -1.0 * observation.bandwidth /
      observation.channels; // MHz   (This must be negative!)
  const size_t n_samples = observation.duration / observation.sampling_period;

  auto timer = std::make_unique<dedisp::benchmark::Timer>();
  std::cout << "Simulating a dispersed signal..." << std::endl;
  timer->start();
  xt::xarray<float> signal =
      dedisp::simulate_dispersed_signal(signal_properties, observation);

  // Quantise the input signal. Note that this actually clips the signal...
  // TODO: don't clip?
  xt::xarray<uint8_t> quantised_signal(signal.shape());
  for (size_t s = 0; s < signal.shape(0); ++s) {
    for (size_t c = 0; c < signal.shape(1); ++c) {
      quantised_signal(s, c) = dedisp::quantise(signal(s, c));
    }
  }
  timer->pause();
  std::cout << quantised_signal << std::endl;
  std::cout << "> runtime: " << timer->duration() << " seconds. "
            << std::endl;

  const std::string filename{"signal.npy"};
  xt::dump_npy(filename, quantised_signal);

  std::cout << "The simulated signal has been written to " << filename << "." << std::endl;
}
