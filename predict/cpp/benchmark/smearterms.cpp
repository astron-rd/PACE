
#include <benchmark/benchmark.h>
#include <xtensor/xtensor.hpp>

#include <predict/Direction.h>
#include <predict/Directions.h>

#include <predict/PointSource.h>
#include <predict/Spectrum.h>
#include <predict/test/Common.h>
#include <random>

using namespace predict;
const Direction test_reference_point{0.5, 0.1};
const Direction test_offset_point{test_reference_point.ra + 0.02,
                                  test_reference_point.dec + 0.02};

const std::vector<int64_t> n_frequencies = {128, 256, 512};
const std::vector<int64_t> n_stations = {12, 32, 52};

const double kHz = 1.e3;
const double MHz = 1.e6;
namespace {} // namespace

class SmearTermsBenchmark : public benchmark::Fixture {
public:
  void SetUp(benchmark::State &state) override {
    size_t n_sources = state.range(0);
    size_t n_frequencies = state.range(1);
    smear_terms.resize({n_sources, n_frequencies});

    static std::mt19937 gen(-5);
    channel_widths.resize({n_frequencies});
    station_phases_p.resize({n_sources});
    station_phases_q.resize({n_sources});

    std::uniform_real_distribution<> phase_dist(-M_PI, M_PI);
    for (int64_t s = 0; s < n_sources; s++) {
      station_phases_p(s) = phase_dist(gen);
      station_phases_q(s) = phase_dist(gen);
    }
    for (int64_t ch = 0; ch < n_frequencies; ch++) {
      channel_widths[ch] = 1.e5;
    }
  }
  void TearDown(benchmark::State &) override {}

protected:
  xt::xtensor<float, 2> smear_terms;
  xt::xtensor<double, 1> station_phases_p;
  xt::xtensor<double, 1> station_phases_q;
  xt::xtensor<float, 1> channel_widths;
  predict::PredictPlanExecCPU::SmearTermsDispatch smear_terms_dispatch_;
};

BENCHMARK_DEFINE_F(SmearTermsBenchmark, SmearTermBenchmarkSIMD)
(benchmark::State &state) {
  for (auto _ : state) {
    for (size_t s = 0; s < station_phases_p.size(); s++) {
      const float phase_diff =
          static_cast<float>(station_phases_p(s) - station_phases_q(s));

      xsimd::dispatch(smear_terms_dispatch_)(
          channel_widths, station_phases_p, station_phases_q, smear_terms,
          phase_diff, xsimd::unaligned_mode());
    }
  }
}

BENCHMARK_REGISTER_F(SmearTermsBenchmark, SmearTermBenchmarkSIMD)
    ->ArgsProduct({n_stations, n_frequencies})
    ->ArgNames({"#stations", "#frequencies"})
    ->Unit(benchmark::kMillisecond);
