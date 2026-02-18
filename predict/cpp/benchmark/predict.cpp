#include <predict/Directions.h>
#include <predict/GaussianSourceCollection.h>
#include <random>

#include <benchmark/benchmark.h>
#include <predict/GaussianSource.h>
#include <predict/PointSource.h>
#include <predict/test/Common.h>

using predict::computation_strategy;
using predict::Direction;
using predict::GaussianSource;
using predict::GaussianSourceCollection;
using predict::MakePredictRun;
using predict::PointSource;
using predict::PointSourceCollection;
using predict::PredictRun;

extern bool do_randomized_run;
extern int randomized_run_seed;

const std::vector<int64_t> nstations = {24, 48};
const std::vector<int64_t> nsources = {64, 128};

Direction test_reference_point{0.5, 0.1};
Direction test_offset_point{test_reference_point.ra + 0.02,
                            test_reference_point.dec + 0.02};

class PredictBenchmark : public benchmark::Fixture {
public:
  void SetUp(benchmark::State &state) override {
    const int64_t stations = state.range(0);
    const int64_t channels = state.range(1);
    const bool is_fullstokes = state.range(2);
    const bool is_freqsmear = state.range(3);

    if (do_randomized_run) {
      SetUpRandomizedSource(test_reference_point, test_offset_point,
                            randomized_run_seed);
    } else {
      SetUpFixedSource(test_reference_point, test_offset_point);
    }

    predict_run = predict::MakePredictRun(test_reference_point,
                                          test_offset_point, stations, channels,
                                          !is_fullstokes, is_freqsmear);
  }

  void TearDown(benchmark::State &) override { predict_run.reset(); }

protected:
  std::unique_ptr<PredictRun> predict_run;
};

BENCHMARK_DEFINE_F(PredictBenchmark, PointSource)
(benchmark::State &state) {
  PointSourceCollection sources;
  sources.Add(predict_run->makeSource<PointSource>());

  for (auto _ : state) {
    predict_run->RunWithStrategy(sources, computation_strategy::SINGLE);
  }
}

BENCHMARK_REGISTER_F(PredictBenchmark, PointSource)
    ->ArgsProduct({
        nstations,
        nsources,
        {false, true},
        {false, true},
    })
    ->ArgNames({"stat", "chan", "fullstokes", "freqsmear"})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(PredictBenchmark, PointSourceSIMD)
    ->ArgsProduct({
        nstations,
        nsources,
        {false, true},
        {false, true},
    })
    ->ArgNames({"stat", "chan", "fullstokes", "freqsmear"})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(PredictBenchmark, PointSourceXSIMD)
(benchmark::State &state) {
  PointSourceCollection sources;
  sources.Add(predict_run->makeSource<PointSource>());

  for (auto _ : state) {
    predict_run->RunWithStrategy(sources, computation_strategy::XSIMD);
  }
}

BENCHMARK_REGISTER_F(PredictBenchmark, PointSourceXSIMD)
    ->ArgsProduct({
        nstations,
        nsources,
        {false, true},
        {false, true},
    })
    ->ArgNames({"stat", "chan", "fullstokes", "freqsmear"})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(PredictBenchmark, GaussianSource)
(benchmark::State &state) {
  GaussianSourceCollection sources;
  sources.Add(predict_run->makeSource<GaussianSource>());

  for (auto _ : state) {
    predict_run->Run(sources);
  }
}

BENCHMARK_REGISTER_F(PredictBenchmark, GaussianSource)
    ->ArgsProduct({
        nstations,
        nsources,
        {false, true},
        {false, true},
    })
    ->ArgNames({"stat", "chan", "fullstokes", "freqsmear"})
    ->Unit(benchmark::kMillisecond);
