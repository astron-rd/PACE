#include <chrono>
#include <iostream>
#include <random>

#include <xtensor/io/xio.hpp>

#include "fddplan.hpp"
#include "h5cpp/dataspace/simple.hpp"
#include "h5cpp/datatype/datatype.hpp"
#include "h5cpp/datatype/type_trait.hpp"
#include "h5cpp/file/file.hpp"
#include "h5cpp/file/functions.hpp"
#include "h5cpp/node/group.hpp"
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

  // Quantise the input signal.
  xt::xarray<uint8_t> quantised_signal(signal.shape());
  for (size_t s = 0; s < signal.shape(0); ++s) {
    for (size_t c = 0; c < signal.shape(1); ++c) {
      quantised_signal(s, c) = dedisp::quantise(signal(s, c));
    }
  }
  timer->pause();
  std::cout << quantised_signal << std::endl;
  std::cout << "> runtime: " << timer->duration() << " seconds. " << std::endl;

  hdf5::file::File output_file = hdf5::file::create("signal.h5");
  hdf5::node::Group root_node = output_file.root();

  hdf5::datatype::Datatype datatype =
      hdf5::datatype::TypeTrait<uint8_t>::create();

  const std::vector<hsize_t> dims(signal.shape().begin(), signal.shape().end());
  auto dataspace = hdf5::dataspace::Simple(dims);
  auto signal_dataset = root_node.create_dataset("signal", datatype, dataspace);

  signal_dataset.write(*signal.data(), datatype, dataspace);

  std::cout << "The simulated signal has been written to signal.h5."
            << std::endl;
}
