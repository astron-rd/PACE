#include <chrono>
#include <iostream>
#include <random>

#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>

#include "fddplan.hpp"
#include "filterbank.hpp"
#include "utilities.hpp"

int main() {
  auto mock_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto plan_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto prep_timer = std::make_unique<dedisp::benchmark::Timer>();
  auto exec_timer = std::make_unique<dedisp::benchmark::Timer>();

  std::cout << "Reading SIGPROC file..." << std::endl;
  mock_timer->start();
  const std::string file_path = "/var/scratch/veldhuis/data/pks_frb110220.fil";
  FilterbankFile fil(file_path);

  // Observation details: duration, integration time, max. frequency, bandwidth,
  // and channel count.
  FilterbankHeader &header = const_cast<FilterbankHeader &>(fil.header());
  std::cout << header << '\n';

  const dedisp::ObservationInfo observation{
      static_cast<float>(header.nsamples * header.tsamp),
      static_cast<float>(header.tsamp), static_cast<float>(header.fch1),
      -1.0f * static_cast<float>(header.foff * header.nchans),
      static_cast<size_t>(header.nchans)};

  // Dedispersion plan constraints: start DM, end DM, pulse width (ms), smearing
  // tolerance.
  const dedisp::DedispersionConstraints constraints{900.0f, 1000.0f, 4.0f,
                                                    1.25f};

  const float frequency_resolution = header.foff;
  const size_t n_samples = header.nsamples;

  std::array<size_t, 2> shape = {static_cast<size_t>(fil.header().nsamples),
                                 static_cast<size_t>(fil.header().nchans)};
  std::cout << fil.header().nsamples << " / " << fil.header().nchans << '\n';
  xt::xarray<uint8_t> fil_input =
      xt::adapt(fil.data_ptr(), fil.data_size(), xt::no_ownership(), shape);

  mock_timer->pause();
  std::cout << fil_input << std::endl;
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
  xt::xarray<float> output = fdd_plan.execute(fil_input);
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
  const float raw_mean = xt::mean<float>(fil_input)();
  const float raw_std = xt::stddev<float>(fil_input)();
  std::cout << "  Raw RMS    : " << raw_mean << std::endl;
  std::cout << "  Raw StdDev : " << raw_std << std::endl;

  const float output_mean = xt::mean<float>(output)();
  const float output_std = xt::stddev<float>(output)();
  std::cout << "  Output RMS    : " << output_mean << std::endl;
  std::cout << "  Output StdDev : " << output_std << std::endl;

  const xt::xarray<float> dm_table = fdd_plan.get_dm_table();

#ifdef DEDISP_DEBUG
  const size_t n_samples_computed = n_samples - fdd_plan.max_delay();
  int n_candidates = 0;
  for (size_t d = 0; d < fdd_plan.dm_count(); ++d) {
    for (size_t s = 0; s < n_samples_computed; ++s) {
      const float value = output(d, s);
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
  std::cout << "\nFound " << n_candidates << " DM candidates." << std::endl;
#endif

  const std::string fn_in{"fddin_fil.npy"};
  xt::dump_npy(fn_in, fil_input);
  std::cout << "\nInput is written to " << fn_in << "." << std::endl;

  const std::string fn_out{"fddout_fil.npy"};
  xt::dump_npy(fn_out, output);
  std::cout << "Output is written to " << fn_out << "." << std::endl;

  const std::string fn_dm_table{"dmtable_fil.npy"};
  xt::dump_npy(fn_dm_table, dm_table);
  std::cout << "Trial DMs are written to " << fn_dm_table << "." << std::endl;
}
