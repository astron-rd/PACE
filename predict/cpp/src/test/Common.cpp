#include <cstdlib>
#include <predict/test/Common.h>

#include <random>

bool do_randomized_run = false;
int randomized_run_seed = 0;

constexpr std::pair<double, double> point_offset_range{-0.1, 0.1};

namespace predict {

std::unique_ptr<PredictRun> MakePredictRun(const Direction &reference_point,
                                           const Direction &offset_point,
                                           size_t n_stations, size_t n_channels,
                                           bool stokes_i_only,
                                           bool correct_freq_smearing) {
  const size_t nbaselines = n_stations * (n_stations - 1) / 2;

  std::unique_ptr<PredictRun> predict_run = std::make_unique<PredictRun>(
      ((stokes_i_only) ? 1u : 4u), n_channels, n_stations, nbaselines);

  predict_run->offset_source = offset_point;
  predict_run->plan.reference = reference_point;

  predict_run->plan.compute_stokes_I_only = stokes_i_only;
  predict_run->plan.correct_frequency_smearing = correct_freq_smearing;

  predict_run->Initialize();

  return predict_run;
}

void SetUpRandomizedSource(Direction &test_reference_point,
                           Direction &test_offset_point, int seed) {
  static std::mt19937 gen(seed);

  std::uniform_real_distribution<> ra_dist(0.0, 2 * M_PI);
  std::uniform_real_distribution<> dec_dist(-M_PI / 2, M_PI / 2);
  std::uniform_real_distribution<> offset_dist(point_offset_range.first,
                                               point_offset_range.second);

  double ra = ra_dist(gen);
  double dec = dec_dist(gen);
  double offset_ra = offset_dist(gen);
  double offset_dec = offset_dist(gen);

  test_reference_point.ra = ra;
  test_reference_point.dec = dec;
  test_offset_point.ra =
      std::clamp(test_reference_point.ra + offset_ra, 0.0, 2 * M_PI);
  test_offset_point.dec =
      std::clamp(test_reference_point.dec + offset_dec, -M_PI / 2, M_PI / 2);
}

void SetUpFixedSource(Direction &test_reference_point,
                      Direction &test_offset_point) {
  // Set fixed source parameters
  test_reference_point.ra = 0.5;
  test_reference_point.dec = 0.1;
  test_offset_point.ra = test_reference_point.ra + 0.02;
  test_offset_point.dec = test_reference_point.dec + 0.02;
}

} // namespace predict