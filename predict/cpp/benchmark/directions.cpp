#include <benchmark/benchmark.h>
#include <predict/Predict.h>
#include <random>
#include <xtensor/xtensor.hpp>

#include <predict/Directions.h>
#include <predict/test/Common.h>
using predict::Direction;
using predict::Directions;

extern bool do_randomized_run;
extern int randomized_run_seed;

const std::vector<int64_t> nstations = {24, 48};
const std::vector<int64_t> nsources = {64, 128};

namespace {
Direction test_reference_point{0.5, 0.1};
Direction test_offset_point{test_reference_point.ra + 0.02,
                            test_reference_point.dec + 0.02};
} // namespace

class DirectionsBenchmark : public benchmark::Fixture {
public:
  void SetUp(benchmark::State &state) override {
    n_directions = state.range(0);
    lmn = xt::xtensor<double, 2>({n_directions, 3});
    directions = std::make_unique<Directions>();
    directions->Reserve(n_directions);
    for (size_t i = 0; i < n_directions; i++) {
      if (do_randomized_run) {
        SetUpRandomizedSource(test_reference_point, test_offset_point,
                              randomized_run_seed);
      } else {
        SetUpFixedSource(test_reference_point, test_offset_point);
      }
      directions->Add(test_offset_point);
    }
  }
  void TearDown(benchmark::State &) override { directions.reset(); }

protected:
  std::unique_ptr<Directions> directions;
  size_t n_directions;
  xt::xtensor<double, 2> lmn;
};

BENCHMARK_DEFINE_F(DirectionsBenchmark, DirectionsOneByOne)
(benchmark::State &state) {
  for (auto _ : state) {
    directions->RaDec2Lmn<Directions::computation_strategy::SINGLE>(
        test_reference_point, lmn);
  }
}

BENCHMARK_DEFINE_F(DirectionsBenchmark, DirectionsMulti)
(benchmark::State &state) {
  for (auto _ : state) {
    directions->RaDec2Lmn<Directions::computation_strategy::MULTI>(
        test_reference_point, lmn);
  }
}

BENCHMARK_REGISTER_F(DirectionsBenchmark, DirectionsOneByOne)
    ->ArgsProduct({{1024, 2048, 4096, 8192}})
    ->ArgNames({
        "#dir",
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DirectionsBenchmark, DirectionsMulti)
    ->ArgsProduct({{1024, 2048, 4096, 8192}})
    ->ArgNames({
        "#dir",
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DirectionsBenchmark, DirectionsMultiSIMD)
    ->ArgsProduct({{1024, 2048, 4096, 8192}})
    ->ArgNames({
        "#dir",
    })
    ->Unit(benchmark::kMillisecond);